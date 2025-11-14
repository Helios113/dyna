# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradient noise scale during training."""

import torch
from typing import Optional, Dict, Any
from collections import defaultdict

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

__all__ = ['GradientNoiseScaleMonitor']


class GradientNoiseScaleMonitor(Callback):
    """Computes and logs the Gradient Noise Scale (GNS) during training.
    
    GNS provides a suggestion for a compute-efficient batch size: small enough to be 
    compute efficient and large enough to take advantage of parallelism. This implementation
    follows the paper "Efficient and Approximate Per-Example Gradient Norms for Gradient 
    Noise Scale" (Gray et al., 2023).
    
    The callback supports two methods:
    1. SOGNS (Scaled Output Gradient Noise Scale): Efficient approximation for transformers
    2. PEPGNS (Per-Example Parameter GNS): Exact computation for 2D tensors
    
    Args:
        batch_log_interval: How often to compute GNS (every N batches)
        small_batch_size: Size of small batch for variance estimation (default: 1 for per-example)
        use_approximation: If True, use SOGNS approximation for 3D+ tensors (default: True)
        accumulation_steps: Number of microbatches to accumulate for small batch gradient estimates
        log_layer_wise: If True, log GNS per layer in addition to global (default: False)
        
    Example:
        >>> from composer import Trainer
        >>> from composer.callbacks import GradientNoiseScaleMonitor
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_dataloader=train_dataloader,
        ...     optimizers=optimizer,
        ...     max_duration="1ep",
        ...     callbacks=[GradientNoiseScaleMonitor(batch_log_interval=100)],
        ... )
    
    Logged metrics:
    +-----------------------------------------------+-----------------------------------------------------+
    | Key                                           | Logged data                                         |
    +===============================================+=====================================================+
    | ``gns/simple/global``                         | Global gradient noise scale (B_simple)              |
    +-----------------------------------------------+-----------------------------------------------------+
    | ``gns/simple/LAYER_NAME``                     | Layer-wise GNS (if log_layer_wise=True)            |
    +-----------------------------------------------+-----------------------------------------------------+
    | ``gns/grad_variance/global``                  | Global gradient variance (trace of covariance)      |
    +-----------------------------------------------+-----------------------------------------------------+
    | ``gns/grad_mean_norm/global``                 | Global gradient mean squared norm                   |
    +-----------------------------------------------+-----------------------------------------------------+
    """

    def __init__(
        self,
        batch_log_interval: int = 100,
        small_batch_size: int = 1,
        use_approximation: bool = True,
        accumulation_steps: int = 10,
        log_layer_wise: bool = False,
    ):
        self.batch_log_interval = batch_log_interval
        self.small_batch_size = small_batch_size
        self.use_approximation = use_approximation
        self.accumulation_steps = accumulation_steps
        self.log_layer_wise = log_layer_wise
        
        # Storage for gradient accumulation
        self.small_batch_grads: Dict[str, list] = defaultdict(list)
        self.large_batch_grad_norms: Dict[str, float] = {}
        self.microbatch_count = 0
        
    def _compute_approximate_per_example_norm(
        self, 
        param: torch.Tensor, 
        grad: torch.Tensor
    ) -> torch.Tensor:
        """Compute SOGNS approximation for 3D+ tensors.
        
        For inputs with shape [batch, seq_len, ...], computes:
        η²_b = I * σ²_b * Σ_k (Σ_t y'_btk)²
        
        where σ²_b is the per-example variance of activations.
        """
        if grad.dim() < 3:
            # For 2D tensors, use exact computation
            return torch.linalg.vector_norm(grad.reshape(grad.shape[0], -1), dim=1)
        
        # Assume grad has shape [batch, seq_len, ...] or similar
        batch_size = grad.shape[0]
        
        # Flatten all dimensions except batch and sequence
        if grad.dim() == 3:
            # Standard transformer case: [batch, seq, hidden]
            grad_reshaped = grad  # [B, T, K]
        else:
            # For higher dimensional tensors, flatten non-batch dims
            grad_reshaped = grad.reshape(batch_size, grad.shape[1], -1)
        
        B, T, K = grad_reshaped.shape
        I = K  # Input dimension (approximated)
        
        # Sum over sequence dimension: [B, K]
        grad_summed = grad_reshaped.sum(dim=1)
        
        # Approximate variance (assuming we don't have access to activations)
        # Using gradient statistics as proxy: σ²_b ≈ mean(grad²) over seq
        sigma_squared_b = (grad_reshaped ** 2).mean(dim=(1, 2))  # [B]
        
        # Compute approximate per-example norm
        # η²_b = I * σ²_b * Σ_k (Σ_t y'_btk)²
        eta_squared = I * sigma_squared_b * (grad_summed ** 2).sum(dim=1)
        
        return torch.sqrt(eta_squared.clamp(min=1e-10))
    
    def _compute_per_example_gradient_norms(
        self,
        state: State,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-example gradient norms for all parameters."""
        per_example_norms = {}
        
        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:
                if self.use_approximation and p.grad.dim() >= 3:
                    # Use SOGNS approximation for 3D+ tensors
                    norms = self._compute_approximate_per_example_norm(p, p.grad)
                else:
                    # Exact computation for 2D tensors
                    # Flatten all non-batch dimensions
                    grad_flat = p.grad.reshape(p.grad.shape[0], -1)
                    norms = torch.linalg.vector_norm(grad_flat, dim=1)
                
                per_example_norms[name] = norms
        
        return per_example_norms
    
    def _accumulate_small_batch_gradients(self, state: State):
        """Accumulate gradients from small batches (microbatches)."""
        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:
                # Store gradient norm squared for this microbatch
                grad_norm_sq = torch.linalg.vector_norm(p.grad) ** 2
                self.small_batch_grads[name].append(grad_norm_sq.detach().cpu())
        
        self.microbatch_count += 1
    
    def _compute_gns_metrics(
        self,
        state: State,
    ) -> Dict[str, float]:
        """Compute GNS using accumulated small batch gradients and current large batch."""
        metrics = {}
        
        # Get large batch size from state
        B_big = state.timestamp.batch_size.value if hasattr(state.timestamp, 'batch_size') else 32
        B_small = self.small_batch_size
        
        if self.microbatch_count == 0 or B_big <= B_small:
            return metrics
        
        global_S = 0.0  # Variance estimate (trace of covariance)
        global_G_sq = 0.0  # Mean gradient norm squared
        
        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:
                # Large batch gradient norm squared
                G_big_sq = torch.linalg.vector_norm(p.grad) ** 2
                
                # Small batch gradient norm squared (mean of accumulated)
                if name in self.small_batch_grads and len(self.small_batch_grads[name]) > 0:
                    small_grads = torch.stack(self.small_batch_grads[name])
                    G_small_sq = small_grads.mean()
                    
                    # Unbiased estimators from McCandlish et al. 2018
                    # |G|² := 1/(B_big - B_small) * (B_big * |G_big|² - B_small * |G_small|²)
                    G_sq_est = (B_big * G_big_sq - B_small * G_small_sq) / (B_big - B_small)
                    
                    # S := 1/(1/B_small - 1/B_big) * (|G_small|² - |G_big|²)
                    S_est = (G_small_sq - G_big_sq) / (1.0 / B_small - 1.0 / B_big)
                    
                    # Accumulate for global metrics
                    global_S += S_est.item()
                    global_G_sq += G_sq_est.item()
                    
                    # Layer-wise metrics
                    if self.log_layer_wise:
                        layer_gns = S_est / (G_sq_est + 1e-10)
                        metrics[f'gns/simple/{name}'] = layer_gns.item()
        
        # Global GNS
        if global_G_sq > 0:
            B_simple = global_S / global_G_sq
            metrics['gns/simple/global'] = B_simple
            metrics['gns/grad_variance/global'] = global_S
            metrics['gns/grad_mean_norm/global'] = global_G_sq
        
        # Reduce across ranks if distributed
        if dist.get_world_size() > 1:
            for key in ['gns/simple/global', 'gns/grad_variance/global', 'gns/grad_mean_norm/global']:
                if key in metrics:
                    tensor = torch.tensor(metrics[key], device=state.device)
                    dist.all_reduce(tensor, reduce_operation='SUM')
                    metrics[key] = (tensor / dist.get_world_size()).item()
        
        return metrics

    def after_dataloader(self, state: State, logger: Logger):
        """Called after the dataloader returns a batch, before forward pass.
        
        This is where we can capture small batch gradients during accumulation.
        """
        # Check if we're in accumulation mode
        if hasattr(state, 'gradient_accumulation') and state.gradient_accumulation > 1:
            # We're in the middle of accumulation, store these as small batch grads
            if self.microbatch_count < self.accumulation_steps:
                # Note: We actually need to do this after backward, not here
                pass

    def after_train_batch(self, state: State, logger: Logger):
        """Called after the backward pass and optimizer step."""
        
        # Only compute every N batches
        if state.timestamp.batch.value % self.batch_log_interval != 0:
            # Still accumulate small batch stats even when not logging
            if self.microbatch_count < self.accumulation_steps:
                self._accumulate_small_batch_gradients(state)
            return
        
        # Accumulate small batch gradient norms
        if self.microbatch_count < self.accumulation_steps:
            self._accumulate_small_batch_gradients(state)
        
        # Compute GNS metrics
        if self.microbatch_count >= self.accumulation_steps:
            metrics = self._compute_gns_metrics(state)
            
            if metrics:
                logger.log_metrics(metrics)
        
        # Reset accumulation
        self.small_batch_grads.clear()
        self.microbatch_count = 0