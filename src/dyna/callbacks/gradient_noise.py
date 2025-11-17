# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradient statistics and noise across microbatches."""

import torch
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
import wandb
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

__all__ = ['GradientNoiseMonitor']


class GradientNoiseMonitor(Callback):
    """Computes and logs gradient mean, variance, and SNR statistics across microbatches.
    
    This callback tracks the accumulated gradients after each microbatch is added and computes
    the Signal-to-Noise Ratio (SNR) as mean/std across different batch sizes.
    
    The SNR (reciprocal of coefficient of variation) shows how the gradient signal
    quality improves with larger batch sizes.
    
    Args:
        microbatch_size: Size of each microbatch (e.g., 16)
        log_interval: Log statistics every N training steps
        track_layers: Optional list of layer name patterns to track. If None, tracks all.
    
    Example:
        >>> from composer import Trainer
        >>> from composer.callbacks import GradientNoiseMonitor
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_dataloader=train_dataloader,
        ...     optimizers=optimizer,
        ...     max_duration="1ep",
        ...     callbacks=[GradientNoiseMonitor(
        ...         microbatch_size=16,
        ...         log_interval=10
        ...     )],
        ... )
    
    Logged metrics:
        - grad_snr/{layer_name}/bs_{batch_size}: SNR = mean/std for each layer
        - grad_mean/{layer_name}/bs_{batch_size}: Mean of accumulated gradients
        - grad_std/{layer_name}/bs_{batch_size}: Std of accumulated gradients
        - grad_snr/global/bs_{batch_size}: Global SNR across all parameters
    """

    def __init__(
        self,
        microbatch_size: int,
        log_interval: int = 1,
        track_layers: Optional[List[str]] = None
    ):
        self.microbatch_size = microbatch_size
        self.log_interval = log_interval
        self.track_layers = track_layers
        
        # Storage for gradient statistics after each microbatch
        # {layer_name: [(mean, variance, batch_size), ...]}
        self.reset_accumulation()
        
    def reset_accumulation(self):
        """Reset accumulated gradient statistics."""
        self.gradient_stats: Dict[str, List[Tuple[float, float, int]]] = defaultdict(list)
        self.current_batch_size = 0
        
    def _should_track_layer(self, name: str) -> bool:
        """Check if we should track this layer."""
        if self.track_layers is None:
            return True
        return any(pattern in name for pattern in self.track_layers)
    
    def after_backward(self, state: State, logger: Logger):
        """Store mean and variance of accumulated gradients after each microbatch."""
        # Increment the current accumulated batch size
        self.current_batch_size += self.microbatch_size
        print(f"Accumulated batch size: {self.current_batch_size}", flush=True)
        # Store statistics for each parameter
        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad and self._should_track_layer(name):
                # Compute mean and variance of the accumulated gradient
                grad_mean = p.grad.mean().item()
                grad_var = p.grad.var().item()
                
                # Store (mean, variance, batch_size)
                self.gradient_stats[name].append((grad_mean, grad_var, self.current_batch_size))
    
    def batch_end(self, state: State, logger: Logger):
        """Compute and log SNR statistics across all accumulated batch sizes."""
        if state.timestamp.batch.value % self.log_interval != 0:
            self.reset_accumulation()
            return
        
        if len(self.gradient_stats) == 0:
            self.reset_accumulation()
            return
        
        # Collect data for plotting
        layer_data = {}  # {layer_name: {'batch_sizes': [], 'snr': [], 'mean': [], 'std': []}}
        
        # Compute SNR for each layer at each batch size
        for name, stats_list in self.gradient_stats.items():
            layer_data[name] = {
                'batch_sizes': [],
                'snr': [],
                "var" : []
            }
            
            for mean_val, var_val, batch_size in stats_list:
                print(f"Layer: {name}, Batch Size: {batch_size}, Mean: {mean_val}, Var: {var_val}", flush=True)
                # Compute std from variance
                std_val = var_val ** 0.5
                
                # Compute SNR as mean/std (reciprocal of coefficient of variation)
                if std_val > 1e-10:
                    snr = abs(mean_val) / std_val
                else:
                    snr = float('inf') if abs(mean_val) > 1e-10 else 0.0
                
                layer_data[name]['batch_sizes'].append(batch_size)
                layer_data[name]['snr'].append(snr if snr != float('inf') else 1e6)
                layer_data[name]['var'].append(var_val if var_val != float('inf') else 1e6)
                
                
        
        # Compute global statistics across all parameters at each batch size
        batch_sizes = sorted(set(bs for stats_list in self.gradient_stats.values() 
                                for _, _, bs in stats_list))
        
        global_data = {
            'batch_sizes': [],
            'snr': [],
            'var': []
        }
        
        for batch_size in batch_sizes:
            all_means = []
            all_vars = []
            
            for name, stats_list in self.gradient_stats.items():
                # Find stats for this batch size
                for mean_val, var_val, bs in stats_list:
                    if bs == batch_size:
                        all_means.append(mean_val)
                        all_vars.append(var_val)
                        break
            
            if len(all_means) > 0:
                # Compute global statistics
                global_mean = sum(all_means) / len(all_means)
                global_var = sum(all_vars) / len(all_vars)
                global_std = global_var ** 0.5
                
                if global_std > 1e-10:
                    global_snr = abs(global_mean) / global_std
                else:
                    global_snr = float('inf') if abs(global_mean) > 1e-10 else 0.0
                
                global_data['batch_sizes'].append(batch_size)
                global_data['snr'].append(global_snr if global_snr != float('inf') else 1e6)
                global_data['var'].append(global_var if global_var != float('inf') else 1e6)
                
        
        # Create plots
        self._create_and_log_plots(layer_data, global_data, logger, state)
        
        # Reset for next minibatch
        self.reset_accumulation()
    
    def _create_and_log_plots(self, layer_data: Dict, global_data: Dict, logger: Logger, state: State):
        """Create matplotlib plots and log to wandb."""
        
        metrics_dict = {}
        
        # Plot 1: Global SNR vs Batch Size
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(global_data['batch_sizes'], global_data['snr'], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('SNR (Signal-to-Noise Ratio)', fontsize=12)
        ax.set_title('Gradient SNR vs Batch Size (Global)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        plt.tight_layout()
        
        # Convert to wandb image and store in metrics
        metrics_dict['gradient_noise/global_snr'] = self._fig_to_wandb_image(fig)
        # Plot 1: Global SNR vs Batch Size
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(global_data['batch_sizes'], global_data['var'], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Var', fontsize=12)
        ax.set_title('Gradient SNR vs Batch Size (Global)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # ax.set_xscale('log', base=2)
        # ax.set_yscale('log')
        plt.tight_layout()
        
        # Convert to wandb image and store in metrics
        metrics_dict['gradient_noise/global_var'] = self._fig_to_wandb_image(fig)
        
        
        # Plot 3: SNR for bottom layers (by final SNR)
        # Sort layers by their final SNR value (lowest first)
        print("layer_data", layer_data, flush=True)
        layer_final_snr = {name: data['snr'][-1] for name, data in layer_data.items()}
        bottom_layers = sorted(layer_final_snr.items(), key=lambda x: x[1], reverse=False)[:10]
        
        if len(bottom_layers) > 0:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            for layer_name, _ in bottom_layers:
                data = layer_data[layer_name]
                # Truncate long layer names for legend
                short_name = layer_name.split('.')[-1] if len(layer_name) > 30 else layer_name
                ax.plot(data['batch_sizes'], data['snr'], 'o-', label=short_name, alpha=0.7)
            
            ax.set_xlabel('Batch Size', fontsize=12)
            ax.set_ylabel('SNR (Signal-to-Noise Ratio)', fontsize=12)
            ax.set_title('Gradient SNR vs Batch Size (Bottom 10 Layers)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.tight_layout()
            
            metrics_dict['gradient_noise/bottom_layers'] = self._fig_to_wandb_image(fig)
        
        # Log all metrics at once
        logger.log_metrics(metrics_dict)
    
    def _fig_to_wandb_image(self, fig) -> wandb.Image:
        """Convert matplotlib figure to wandb Image."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)

        # Create wandb image from PIL Image
        pil_img = Image.open(buf)
        img = wandb.Image(pil_img)
        plt.close(fig)
        return img