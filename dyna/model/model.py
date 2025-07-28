import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from composer.models import HuggingFaceModel
from flash_attn.ops.triton.layer_norm import RMSNorm
from llmfoundry.utils.builders import build_metric
from omegaconf import DictConfig
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from dyna.model.cvmm import CVMMSel, cvmm, cvmm_prepare_sel2

# Constants
CROSS_ENTROPY_IGNORE_INDEX = -100
DEFAULT_CAUSAL_LM_TRAIN_METRICS = [
    "language_cross_entropy",
    "language_perplexity", 
    "token_accuracy",
]

# Type aliases for better readability
KVCache = Optional[Dict[str, torch.Tensor]]
MultilayerKVCache = Optional[Dict[int, KVCache]]


@dataclass
class AttentionMask:
    """Container for attention mask components."""
    src_length_mask: Optional[torch.Tensor]
    position_mask: Optional[torch.Tensor]


@dataclass
class MoEUTOutput:
    """Output container for MoEUT model."""
    outputs: torch.Tensor
    reg_loss: torch.Tensor
    cache: MultilayerKVCache


def get_targets(labels: torch.Tensor) -> torch.Tensor:
    """Shift labels for causal language modeling."""
    targets = torch.roll(labels, shifts=-1)
    targets[:, -1] = CROSS_ENTROPY_IGNORE_INDEX
    return targets


def compute_loss_from_logits(
    outputs: CausalLMOutputWithPast,
    shift_labels: bool,
    labels: torch.Tensor,
    loss_fn: torch.nn.Module,
) -> torch.Tensor:
    """Compute cross-entropy loss from logits and labels."""
    targets = get_targets(labels) if shift_labels else labels
    
    losses = loss_fn(
        outputs.logits.view(-1, outputs.logits.size(-1)),
        targets.view(-1),
    )
    
    if torch.all(targets == loss_fn.ignore_index):
        loss = losses.sum()
    else:
        loss = losses.sum() / (targets != loss_fn.ignore_index).sum()
    
    return loss


def round_up_to_multiple_of_256(n: int) -> int:
    """Return the smallest number divisible by 256 that is >= n."""
    if n <= 0:
        return 256
    return ((n - 1) // 256 + 1) * 256


def log_mean(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Compute log of mean along specified dimension."""
    return x.logsumexp(dim) - math.log(x.shape[dim])


def entropy_l(l: torch.Tensor) -> torch.Tensor:
    """Compute entropy from log probabilities."""
    return -(l * l.exp()).sum(-1)


def entropy_reg(sel: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute entropy regularization term."""
    sel = F.log_softmax(sel, dim=-1)
    sel = log_mean(sel, dim)
    return -entropy_l(sel).mean()


class MoEUTConfig(PretrainedConfig):
    """Configuration class for MoEUT model."""
    
    model_type = "moeut"

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_ffn_experts: int,
        n_att_experts: int,
        d_head: Optional[int] = None,
        group_size: int = 2,
        ff_k: int = 8,
        att_k: int = 2,
        ff_expert_dropout: float = 0.0,
        att_expert_dropout: float = 0.0,
        ff_expert_size: int = 128,
        dropout: float = 0.0,
        entropy_reg: float = 0.01,
        att_entropy_reg: float = 0.001,
        shift_labels: bool = True,
        scale_add: bool = True,
        prot_emb: bool = False,
        shared_expert_number: int = 1,
        **kwargs,
    ):
        super().__init__(**{"model_type": self.model_type})
        
        # Model architecture
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        
        # Expert configuration
        self.n_ffn_experts = n_ffn_experts
        self.n_att_experts = n_att_experts
        self.ff_expert_size = ff_expert_size
        self.shared_expert_number = shared_expert_number
        
        # Routing configuration
        self.group_size = group_size
        self.ff_k = ff_k
        self.att_k = att_k
        
        # Dropout and regularization
        self.ff_expert_dropout = ff_expert_dropout
        self.att_expert_dropout = att_expert_dropout
        self.dropout = dropout
        self.entropy_reg = entropy_reg
        self.att_entropy_reg = att_entropy_reg
        
        # Training configuration
        self.shift_labels = shift_labels
        self.scale_add = scale_add
        self.prot_emb = prot_emb
        
        # Store additional kwargs for forward compatibility
        for k, v in kwargs.items():
            setattr(self, k, v)


class SigmaMoE(torch.nn.Module):
    """Sigma Mixture of Experts layer for feed-forward networks."""
    
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        expert_size: int,
        k: int,
        activation=F.relu,
        v_dim: Optional[int] = None,
        expert_dropout: float = 0.0,
        shared_expert_number: int = 1,
    ):
        super().__init__()
        
        # Model dimensions
        self.k_dim = d_model
        self.v_dim = v_dim if v_dim is not None else d_model
        self.d_model = d_model
        
        # Expert configuration
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.shared_expert_number = min(shared_expert_number, n_experts)
        self.routed_expert_number = n_experts - self.shared_expert_number
        
        # Routing configuration
        self.n_heads = k
        self.activation = activation
        self.expert_dropout = expert_dropout
        
        # Bias tracking for load balancing
        self.bias = torch.nn.Parameter(torch.zeros(n_experts), requires_grad=False)
        self.bias_update_lr = 0.001
        
        # Expert parameters
        self.keys = torch.nn.Parameter(
            torch.empty(self.n_experts, self.k_dim, self.expert_size)
        )
        self.values = torch.nn.Parameter(
            torch.empty(self.n_experts, self.expert_size, self.v_dim)
        )
        self.expert_sel = torch.nn.Parameter(
            torch.empty(self.n_experts, self.k_dim)
        )
        
        # Register shared expert indices
        self.register_buffer(
            'shared_expert',
            torch.arange(n_experts - self.shared_expert_number, n_experts, dtype=torch.long)
        )

    def reset_parameters(self, std_scale: float) -> None:
        """Initialize parameters with proper scaling."""
        torch.nn.init.normal_(self.expert_sel, 0, std_scale / math.sqrt(self.k_dim))
        torch.nn.init.normal_(self.keys, 0, std_scale / math.sqrt(self.k_dim))
        torch.nn.init.normal_(
            self.values, 0, std_scale / math.sqrt(self.n_experts * self.expert_size)
        )
        self.renorm_keep_std(self.expert_sel, dim=1)

    def renorm_keep_std(self, weight: torch.Tensor, dim: int = 0) -> None:
        """Renormalize weights while keeping standard deviation."""
        with torch.no_grad():
            std = weight.std()
            weight.div_(weight.norm(dim=dim, keepdim=True))
            weight.mul_(std / weight.std())

    def _compute_expert_selection(
        self, 
        input_tensor: torch.Tensor, 
        sel_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute expert selection scores and indices."""
        # Compute selection scores
        sel = F.linear(
            sel_input if sel_input is not None else input_tensor, 
            self.expert_sel, 
            None
        )
        sel = F.sigmoid(sel)
        
        # Apply expert dropout during training
        if self.training and self.expert_dropout > 0:
            mask = torch.rand_like(sel) < self.expert_dropout
            sel = sel.masked_fill(mask, float("-inf"))
        
        # Select top-k routed experts
        _, sel_index = torch.topk(
            sel[:, :, :self.routed_expert_number] + self.bias[:self.routed_expert_number],
            self.routed_expert_number,
            dim=-1,
            sorted=False
        )
        
        # Add shared experts to selection
        if self.shared_expert_number > 0:
            shared_shape = sel_index.shape[:-1] + (self.shared_expert_number,)
            shared_expert_expanded = self.shared_expert.view(
                *([1] * (sel_index.dim() - 1)), -1
            ).expand(shared_shape)
            sel_index = torch.cat([sel_index, shared_expert_expanded], dim=-1)
        
        sel_val = torch.gather(sel, -1, sel_index)
        
        # Update bias for load balancing during training
        if self.training:
            self._update_load_balancing_bias(sel_index)
        
        return sel_val, sel_index

    def _update_load_balancing_bias(self, sel_index: torch.Tensor) -> None:
        """Update bias for load balancing."""
        with torch.no_grad():
            c_i = torch.bincount(sel_index.flatten(), minlength=self.n_experts)
            c_i_avg = torch.mean(c_i, dtype=torch.float32)
            
            self.bias[:self.routed_expert_number] += self.bias_update_lr * torch.sign(
                -c_i[:self.routed_expert_number] + c_i_avg
            )

    def forward(
        self, 
        input_tensor: torch.Tensor, 
        sel_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the MoE layer."""
        # Get expert selection
        sel_val, sel_index = self._compute_expert_selection(input_tensor, sel_input)
        
        # Prepare selection indices for CVMM operations
        sel_indices = cvmm_prepare_sel2(sel_index.int())
        
        # Up-projection: input * expert_keys
        scores = cvmm(input_tensor, sel_indices, self.keys)
        scores = self.activation(scores)
        
        # Down-projection: scores * expert_values
        sel_indices = sel_indices.clone()
        sel_indices.reduction_weight = sel_val
        sel_indices.sel_index = sel_indices.out_index
        sel_indices.out_index = None
        
        out = cvmm(scores, sel_indices, self.values)
        
        return out.view(*input_tensor.shape[:-1], self.v_dim)


class SwitchHeadCore(torch.nn.Module):
    """Core attention mechanism with expert routing."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_experts: int,
        dropout: float = 0.0,
        d_head: Optional[int] = None,
        expert_dropout: float = 0.0,
        moe_k: int = 2,
        shared_expert_number: int = 1,
    ):
        super().__init__()
        
        # Model configuration
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head or (d_model // n_heads)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else lambda x: x
        
        # Expert configuration
        self.n_experts = n_experts
        self.shared_expert_number = min(shared_expert_number, n_experts)
        self.routed_expert_number = n_experts - self.shared_expert_number
        self.moe_k = moe_k
        self.expert_dropout = expert_dropout
        
        # Bias tracking
        self.bias_update_lr = 0.001
        
        # Query and Key projections (shared)
        self.q = torch.nn.Linear(self.d_model, self.d_head * self.n_heads, bias=False)
        self.k = torch.nn.Linear(self.d_model, self.d_head * self.n_heads, bias=False)
        
        # Expert-specific parameters
        self._init_expert_parameters()
        
        # Shared expert indices
        self.register_buffer(
            'shared_expert',
            torch.arange(n_experts - self.shared_expert_number, n_experts, dtype=torch.long)
        )
        
        # Attention scale
        self.register_buffer(
            "scale",
            torch.full([1], 1.0 / math.sqrt(self.d_head)),
            persistent=False,
        )
        
        # Tracking variables for visualization
        self.attention_to_visualize = []
        self.selections_to_visualize = {}
        self.sel_hist = []

    def _init_expert_parameters(self) -> None:
        """Initialize expert-specific parameters."""
        if self.n_experts > 1:
            # Value and output projections for multiple experts
            self.v = torch.nn.Parameter(
                torch.empty(self.n_heads * self.n_experts, self.d_model, self.d_head)
            )
            self.o = torch.nn.Parameter(
                torch.empty(self.n_heads * self.n_experts, self.d_head, self.d_model)
            )
            
            # Expert selection parameters
            self.sel_v = torch.nn.Parameter(
                torch.empty(self.n_heads * self.n_experts, self.d_model)
            )
            self.sel_o = torch.nn.Parameter(
                torch.empty(self.n_heads * self.n_experts, self.d_model)
            )
            
            # Bias parameters for load balancing
            self.bias_v = torch.nn.Parameter(torch.zeros(self.n_experts), requires_grad=False)
            self.bias_o = torch.nn.Parameter(torch.zeros(self.n_experts), requires_grad=False)
        else:
            # Single expert case
            self.v = torch.nn.Parameter(
                torch.empty(self.n_heads * self.d_head, self.d_model)
            )
            self.o = torch.nn.Parameter(
                torch.empty(self.d_model, self.n_heads * self.d_head)
            )
            self.sel_o = torch.nn.Parameter(
                torch.empty(self.n_heads * self.n_experts, self.d_model)
            )

    def reset_parameters(self, std_scale: float) -> None:
        """Initialize all parameters with proper scaling."""
        # Initialize selection parameters
        if self.n_experts > 1:
            torch.nn.init.normal_(self.sel_v, 0, std_scale / math.sqrt(self.d_model))
            self.renorm_rows(self.sel_v)
        
        torch.nn.init.normal_(self.sel_o, 0, std_scale / math.sqrt(self.d_model))
        self.renorm_rows(self.sel_o)
        
        # Initialize projection parameters
        torch.nn.init.normal_(self.k.weight, 0, std_scale / math.sqrt(self.d_model))
        torch.nn.init.normal_(self.q.weight, 0, std_scale / math.sqrt(self.d_model))
        torch.nn.init.normal_(self.v, 0, std_scale / math.sqrt(self.d_model))
        torch.nn.init.normal_(self.o, 0, std_scale / math.sqrt(self.n_heads * self.d_head))

    def renorm_rows(self, x: torch.Tensor) -> None:
        """Renormalize rows while preserving standard deviation."""
        with torch.no_grad():
            std_t = x.std(dim=-1, keepdim=True)
            x.div_(x.norm(dim=-1, keepdim=True))
            x.mul_(std_t / x.std())

    def project_to_torch_order(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor to PyTorch attention format."""
        return x.view(*x.shape[:-1], self.n_heads, -1).transpose(-2, -3)

    def _create_attention_mask(
        self, 
        src_len: int, 
        mask: Tuple[torch.Tensor, torch.Tensor], 
        skip_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Create attention mask tensor."""
        if mask is None or (mask[0] is None and mask[1] is None):
            return None
        
        # Process position and length masks
        if mask[0] is not None:
            n_pad = src_len - mask[0].shape[-1]
            pm = F.pad(mask[0], (n_pad, 0), "constant", value=False) if n_pad > 0 else mask[0]
        
        # Combine masks
        if mask[0] is None:
            m = mask[1].unsqueeze(-2).unsqueeze(-2)
        elif mask[1] is None:
            m = pm
        else:
            m = mask[1].unsqueeze(-2).unsqueeze(-2) | pm
        
        # Create efficient batched mask
        lengths = skip_mask.sum(dim=1)
        max_len = round_up_to_multiple_of_256(lengths.max().item())
        attention_mask = torch.zeros(
            skip_mask.shape[0], self.n_heads, max_len, src_len,
            dtype=torch.bool, device=self.v.device
        )
        
        for i in range(skip_mask.shape[0]):
            idx = skip_mask[i].nonzero(as_tuple=False).squeeze(-1)
            n = idx.numel()
            if n > 0:
                attention_mask[i, :, :n] = m[idx]
        
        return attention_mask

    def _get_expert_selection(
        self, 
        input_tensor: torch.Tensor, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor] = None
    ) -> Tuple[CVMMSel, torch.Tensor]:
        """Get expert selection indices and weights."""
        # Compute selection scores
        sel = F.linear(input_tensor, weight).float()
        sel_raw = sel.view(*sel.shape[:-1], self.n_heads, -1)
        sel = sel_raw.sigmoid()
        
        with torch.no_grad():
            # Apply expert dropout
            if self.expert_dropout > 0 and self.training:
                mask = torch.rand_like(sel) < self.expert_dropout
                sel2 = sel.masked_fill(mask, float("-inf"))
            else:
                sel2 = sel
            
            # Select routed experts
            routed_k = max(1, self.moe_k - self.shared_expert_number)
            bias_term = bias[:self.routed_expert_number] if bias is not None else None
            
            _, sel_index = torch.topk(
                (sel2[:, :, :, :self.routed_expert_number] + bias_term) 
                if bias_term is not None else sel2[:, :, :, :self.routed_expert_number],
                routed_k, dim=-1, sorted=False
            )
            
            # Add shared experts
            if self.shared_expert_number > 0:
                shared_shape = sel_index.shape[:-1] + (self.shared_expert_number,)
                shared_expert_expanded = self.shared_expert.view(
                    *([1] * (sel_index.dim() - 1)), -1
                ).expand(shared_shape)
                sel_index = torch.cat([sel_index, shared_expert_expanded], dim=-1)
            
            # Update bias for load balancing
            if self.training and bias is not None:
                c_i = torch.bincount(sel_index.flatten(), minlength=self.n_experts)
                c_i_avg = torch.mean(c_i, dtype=torch.float32)
                bias[:self.routed_expert_number] += self.bias_update_lr * torch.sign(
                    -c_i[:self.routed_expert_number] + c_i_avg
                )
        
        # Get selection values and create CVMM selection object
        sel_val = torch.gather(
            sel.view(*sel.shape[:-2], -1), 
            -1, 
            sel_index.view(*sel_index.shape[:-2], -1)
        ).view(*sel_index.shape)
        
        # Create shifted indices for expert matrix operations
        sel_index_shifted = (
            torch.arange(self.n_heads, device=sel_index.device, dtype=sel_index.dtype)
            * self.n_experts
        ).unsqueeze(-1) + sel_index
        
        return cvmm_prepare_sel2(sel_index_shifted.flatten(-2, -1), sel_val), sel_index

    def attend(
        self,
        pos_offset: int,
        positions: torch.Tensor,
        v: torch.Tensor,
        k: torch.Tensor,
        q: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Attention computation - to be implemented by subclasses."""
        raise NotImplementedError()

    def forward(
        self,
        q_src: torch.Tensor,
        k_src: torch.Tensor,
        v_src: torch.Tensor,
        mask: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: KVCache = None,
        pos_offset: torch.Tensor = None,
        positions: torch.Tensor = None,
        skip_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        """Forward pass through the attention layer."""
        # Apply scaling to queries and keys
        scale = self.scale.sqrt()
        q = self.q(q_src) * scale.type_as(q_src)
        k = self.k(k_src) * scale.type_as(k_src)
        
        # Handle expert routing for values and outputs
        if self.n_experts > 1:
            v_sel, v_sel_index = self._get_expert_selection(k_src, self.sel_v, self.bias_v)
            o_sel, o_sel_index = self._get_expert_selection(q_src, self.sel_o, self.bias_o)
            v = cvmm(v_src, v_sel, self.v).transpose(-2, -3)
        else:
            o_gate = F.sigmoid(F.linear(q_src, self.sel_o))
            v = self.project_to_torch_order(F.linear(v_src, self.v))
        
        # Project to attention format
        q = self.project_to_torch_order(q)
        k = self.project_to_torch_order(k)
        
        # Handle KV cache
        if kv_cache is not None:
            v = torch.cat([kv_cache["v"], v], dim=-2) if "v" in kv_cache else v
            k = torch.cat([kv_cache["k"], k], dim=-2) if "k" in kv_cache else k
            kv_cache = {"v": v, "k": k}
        
        # Apply dropout and attention
        q = self.dropout(q)
        attention_mask = self._create_attention_mask(v.shape[-2], mask, skip_mask)
        res = self.attend(pos_offset, positions, v, k, q, attention_mask)
        res = res.transpose(-2, -3)
        
        # Apply output projection
        if self.n_experts > 1:
            o_sel.sel_index = o_sel.out_index // o_sel.reduction_weight.shape[-1]
            o_sel.reduction_weight = o_sel.reduction_weight.flatten(-2)
            out = cvmm(res, o_sel, self.o)
        else:
            res = res * o_gate[..., None]
            out = F.linear(res.flatten(-2), self.o)
        
        return out, kv_cache


class RotaryPosEncoding(torch.nn.Module):
    """Rotary Position Encoding (RoPE) implementation."""
    
    def __init__(self, d_model: int, base: int = 10000, seq_dim: int = 1):
        super().__init__()
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for efficiency
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        self.seq_dim = torch.tensor(seq_dim)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=x1.ndim - 1)

    def apply_rot(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply rotary position encoding to input tensor."""
        if positions is None:
            sin, cos = self.get_sincos_cached(x)
        else:
            sin, cos = self.get_sincos_positions(positions, x)
        
        sin = sin.narrow(self.seq_dim, offset, x.shape[self.seq_dim])
        cos = cos.narrow(self.seq_dim, offset, x.shape[self.seq_dim])
        
        return (x * cos) + (self.rotate_half(x) * sin)

    def get_sincos_cached(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached sin/cos values or compute new ones."""
        seq_len = x.shape[self.seq_dim]
        
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[self.seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            
            tgt_shape = [1] * x.ndim
            tgt_shape[self.seq_dim] = seq_len
            tgt_shape[-1] = x.shape[-1]
            
            self.cos_cached = emb.cos().view(*tgt_shape)
            self.sin_cached = emb.sin().view(*tgt_shape)
        
        return self.sin_cached, self.cos_cached

    def get_sincos_positions(
        self, positions: torch.Tensor, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sin/cos values for specific positions."""
        freqs = torch.einsum("ki,j->kij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(positions.device)
        
        tgt_shape = [1] * q.ndim
        tgt_shape[0] = q.shape[0]
        tgt_shape[1] = 1
        tgt_shape[self.seq_dim] = q.shape[self.seq_dim]
        tgt_shape[-1] = q.shape[-1]
        
        return emb.sin().view(*tgt_shape), emb.cos().view(*tgt_shape)

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        pos_offset: int = 0, 
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors."""
        return (
            self.apply_rot(q, pos_offset, positions), 
            self.apply_rot(k, 0, None)
        )


class SwitchHeadRope(SwitchHeadCore):
    """Attention head with Rotary Position Encoding."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_experts: int,
        dropout: float = 0.0,
        d_head: Optional[int] = None,
        expert_dropout: float = 0.0,
        moe_k: int = 2,
        rotate_fraction: float = 0.5,
        rope_base: float = 10000,
        shared_expert_number: int = 1,
    ):
        super().__init__(
            d_model, n_heads, n_experts, dropout, d_head, expert_dropout, moe_k, shared_expert_number
        )
        
        # RoPE configuration
        self.n_rotate = int(rotate_fraction * self.d_head)
        if self.n_rotate > 0:
            self.pe = RotaryPosEncoding(self.n_rotate, seq_dim=-2, base=rope_base)

    def _apply_rope(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        offset: int, 
        positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position encoding to queries and keys."""
        if self.n_rotate < self.d_head:
            # Split rotated and non-rotated parts
            r_k, nr_k = k[..., :self.n_rotate], k[..., self.n_rotate:]
            r_q, nr_q = q[..., :self.n_rotate], q[..., self.n_rotate:]
            
            # Apply RoPE to rotated parts
            r_q, r_k = self.pe(r_q, r_k, torch.tensor(offset), positions)
            
            # Concatenate back
            return (
                torch.cat([r_q, nr_q], dim=-1),
                torch.cat([r_k, nr_k], dim=-1)
            )
        else:
            # Apply RoPE to entire tensors
            return self.pe(q, k, torch.tensor(offset), positions)

    def attend(
        self,
        pos_offset: int,
        positions: torch.Tensor,
        v: torch.Tensor,
        k: torch.Tensor,
        q: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute attention with RoPE."""
        # Apply rotary position encoding
        if self.n_rotate > 0:
            q, k = self._apply_rope(q, k, pos_offset or 0, positions)
        
        # Compute scaled dot-product attention
        return F.scaled_dot_product_attention(q, k, v, scale=1.0, attn_mask=mask)


class MoEUTLayer(torch.nn.Module):
    """Single layer of the MoEUT model with early exit capability."""
    
    def __init__(self, config: MoEUTConfig):
        super().__init__()
        
        # Get shared expert configuration
        shared_expert_number = getattr(config, 'shared_expert_number', 1)
        
        # Initialize attention and FFN components
        self.attention = SwitchHeadRope(
            config.d_model,
            config.n_heads,
            config.n_att_experts,
            d_head=config.d_head,
            moe_k=config.att_k,
            expert_dropout=config.att_expert_dropout,
            shared_expert_number=shared_expert_number,
        )
        
        self.ffn = SigmaMoE(
            config.d_model,
            config.n_ffn_experts,
            config.ff_expert_size,
            k=config.ff_k,
            expert_dropout=config.ff_expert_dropout,
            shared_expert_number=shared_expert_number,
        )
        
        # Layer normalization
        self.attn_pre = RMSNorm(config.d_model)
        self.attn_post = RMSNorm(config.d_model)
        self.ffn_pre = RMSNorm(config.d_model)
        self.ffn_post = RMSNorm(config.d_model)
        
        # Configuration
        self.drop = torch.nn.Dropout(config.dropout)
        self.scale_add = config.scale_add
        self.group_size = config.group_size

    def _check_early_exit(
        self, 
        x: torch.Tensor, 
        router: torch.nn.Parameter, 
        cum_sum: torch.Tensor, 
        tau: torch.Tensor, 
        layer_index: int
    ) -> Tuple[torch.Tensor, bool]:
        """Check if tokens should exit early and create skip mask."""
        # Compute exit scores
        s_exit = F.sigmoid(F.linear(x, router))
        cum_sum += s_exit
        
        # Create skip mask (True = continue processing)
        skip_mask = cum_sum < tau
        
        # Handle sequence-level early exit
        last_token_idx = x.shape[1] - 1
        last_tokens_exit = ~skip_mask[:, last_token_idx]
        
        # Mark entire sequences as done if last token exits
        for batch_idx in last_tokens_exit.nonzero(as_tuple=True)[0]:
            skip_mask[batch_idx, :] = False
        
        # Check if all sequences are done
        continue_processing = not torch.all(skip_mask == False)
        
        return skip_mask, continue_processing

    def _pack_sequence(
        self, 
        x: torch.Tensor, 
        skip_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Pack sequences for efficient processing."""
        lengths = skip_mask.sum(dim=1)
        max_len = round_up_to_multiple_of_256(lengths.max().item())
        
        # Initialize output tensors
        packed_x = torch.zeros(x.shape[0], max_len, x.shape[-1], device=x.device)
        positions = torch.zeros(x.shape[0], max_len, dtype=torch.long, device=x.device)
        
        # Get valid positions
        batch_idx, seq_idx = skip_mask.nonzero(as_tuple=True)
        
        if batch_idx.numel() > 0:
            # Compute packing indices efficiently
            batch_counts = torch.bincount(batch_idx, minlength=x.shape[0])
            cumsum_counts = torch.cumsum(
                torch.cat([torch.tensor([0], device=x.device), batch_counts[:-1]]),
                dim=0,
            )
            
            local_indices = torch.arange(batch_idx.numel(), device=x.device) - cumsum_counts[batch_idx]
            valid_mask = local_indices < max_len
            
            if valid_mask.any():
                # Pack data efficiently
                packed_x[batch_idx[valid_mask], local_indices[valid_mask]] = x[
                    batch_idx[valid_mask], seq_idx[valid_mask]
                ]
                positions[batch_idx[valid_mask], local_indices[valid_mask]] = seq_idx[valid_mask]
        
        return packed_x, positions, max_len

    def _apply_residual_update(
        self, 
        x: torch.Tensor, 
        update: torch.Tensor, 
        skip_mask: torch.Tensor, 
        cum_sum: torch.Tensor, 
        tau: torch.Tensor, 
        layer_index: int, 
        e: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply residual connection with optional scaling."""
        if self.scale_add:
            if e is not None:
                x = x - e
            
            # Apply scaled update
            scale_factor = layer_index / (layer_index + 1)
            update_factor = cum_sum[skip_mask].unsqueeze(1) * tau / (layer_index + 1)
            
            x[skip_mask] = (
                scale_factor * x[skip_mask] + 
                update.view(-1, update.shape[-1])[:skip_mask.sum()] * update_factor
            )
            
            if e is not None:
                x = x + e
        else:
            # Simple residual connection
            x = update + x
        
        return x

    def forward(
        self,
        x: torch.Tensor,
        layer_index: int,
        e: Optional[torch.Tensor] = None,
        router: Optional[torch.nn.Parameter] = None,
        cum_sum: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: KVCache = None,
    ) -> Tuple[torch.Tensor, KVCache, bool, int]:
        """Forward pass through the layer with early exit."""
        # Update layer index for group processing
        layer_index = self.group_size * layer_index + 1
        
        # Check for early exit
        skip_mask, continue_processing = self._check_early_exit(
            x, router, cum_sum, tau, layer_index
        )
        
        if not continue_processing:
            return x, kv_cache, False, 0
        
        # === ATTENTION BLOCK ===
        # Pre-normalization and packing
        xnorm = self.attn_pre(x)
        packed_x, positions, max_len = self._pack_sequence(xnorm, skip_mask)
        
        # Attention computation
        att, kv_cache = self.attention(
            packed_x, xnorm, xnorm, mask,
            kv_cache=kv_cache, pos_offset=0, positions=positions, skip_mask=skip_mask
        )
        
        # Apply residual connection
        attn_out = self.drop(self.attn_post(att))
        x = self._apply_residual_update(x, attn_out, skip_mask, cum_sum, tau, layer_index, e)
        layer_index += 1
        
        # === FFN BLOCK ===
        # Pre-normalization and packing
        xnorm = self.ffn_pre(x)
        packed_x, positions, _ = self._pack_sequence(xnorm, skip_mask)
        
        # FFN computation
        ffn_out = self.ffn(packed_x, packed_x)
        
        # Apply residual connection
        ffn_processed = self.drop(self.ffn_post(ffn_out))
        
        if self.scale_add:
            if e is not None:
                x = x - e
                x = self._apply_residual_update(x, ffn_processed, skip_mask, cum_sum, tau, layer_index, None)
                x = x + e
            else:
                scale_factor = layer_index / (layer_index + 1)
                x = scale_factor * x + ffn_processed / (layer_index + 1)
        else:
            x = ffn_processed + x
        
        return x, kv_cache, True, max_len


class MoEUTPretrainedModel(PreTrainedModel):
    """Base class for MoEUT pretrained models."""
    
    config_class = MoEUTConfig
    base_model_prefix: str = "moeut"
    is_parallelizable: bool = False
    main_input_name: str = "input_ids"
    load_tf_weights = None


class MoEUT(MoEUTPretrainedModel):
    """MoEUT transformer model with early exit capability."""
    
    def __init__(self, config: MoEUTConfig):
        super().__init__(config)
        
        # Model configuration
        self.entropy_reg = config.entropy_reg
        self.att_entropy_reg = config.att_entropy_reg
        self.group_size = config.group_size
        self.n_repeats = 4
        self.d_model = config.d_model
        
        # Model components
        self.layers = torch.nn.ModuleList([
            MoEUTLayer(config) for _ in range(config.group_size)
        ])
        
        # Early exit parameters
        self.router = torch.nn.Parameter(torch.empty(self.d_model))
        self.tau = torch.nn.Parameter(torch.ones(1))
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize all model parameters."""
        scale = math.sqrt(2 / (self.n_repeats * len(self.layers)))
        torch.nn.init.normal_(self.router, 0, scale / math.sqrt(self.d_model))
        
        # Initialize tracking variables
        self._layer_index_abs = 0
        self._seq_len_evolve = []
        
        # Initialize layer parameters
        for layer in self.modules():
            if isinstance(layer, (SwitchHeadCore, SigmaMoE)):
                layer.reset_parameters(scale)
            elif isinstance(layer, RMSNorm):
                layer.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        e: Optional[torch.Tensor] = None,
        mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: MultilayerKVCache = None,
    ) -> CausalLMOutputWithPast:
        """Forward pass through the model."""
        # Initialize state
        new_cache = {}
        self._seq_len_evolve = []
        cum_sum = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        
        # Process layers with early exit
        layer_index_abs = 0
        continue_processing = True
        
        while continue_processing:
            for layer in self.layers:
                # Forward through layer
                x, _, continue_processing, seq_lengths = layer(
                    x, layer_index_abs, e, self.router, cum_sum, self.tau, mask, kv_cache=None
                )
                
                # Track sequence lengths for analysis
                self._seq_len_evolve.append(copy.deepcopy(seq_lengths))
                
                if not continue_processing:
                    break
                
                layer_index_abs += 1
        
        # Store layer usage for callbacks
        self._layer_index_abs = layer_index_abs
        
        return CausalLMOutputWithPast(
            loss=None,
            logits=x,
            past_key_values=new_cache if kv_cache is not None else None
        )


class MoEUTLM(MoEUTPretrainedModel):
    """MoEUT Language Model with embedding and output layers."""
    
    def __init__(self, config: MoEUTConfig):
        super().__init__(config)
        
        # Core transformer
        self.transformer = MoEUT(config)
        
        # Model configuration
        self.n_layers = config.n_layers
        self.prot_emb = config.prot_emb
        
        # Input/output layers
        self.embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size)
        self.out_norm = RMSNorm(config.d_model)
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embedding and transformer parameters."""
        torch.nn.init.kaiming_normal_(
            self.embedding.weight, mode="fan_in", nonlinearity="linear"
        )
        self.transformer.reset_parameters()

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal attention mask."""
        return torch.triu(
            torch.ones(size, size, dtype=torch.bool, device=self.lm_head.weight.device),
            diagonal=1,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        src_len_mask: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        """Forward pass through the language model."""
        # Validate inputs
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        
        # Get embeddings
        if input_ids is not None:
            x = self.embedding(input_ids)
        else:
            x = inputs_embeds
        
        # Generate default attention mask if needed
        if attention_mask is None:
            attention_mask = self._generate_causal_mask(x.shape[-2])
        
        # Prepare protected embeddings if enabled
        e = x.clone() if self.prot_emb else None
        
        # Forward through transformer
        outputs = self.transformer(x, e, (attention_mask, src_len_mask), {})
        
        # Apply output projection
        outputs.logits = self.lm_head(self.out_norm(outputs.logits))
        
        return outputs


class ComposerMoEUT(HuggingFaceModel):
    """Composer-compatible MoEUT model wrapper."""
    
    def __init__(
        self,
        config: MoEUTConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        # Create model
        model = MoEUTLM(config)
        
        # Configuration
        self.vocab_size = config.vocab_size
        self.shift_labels = config.shift_labels
        
        # Metrics
        train_metrics = [
            build_metric(metric, {}) 
            for metric in DEFAULT_CAUSAL_LM_TRAIN_METRICS
        ]
        
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics={},
            shift_labels=config.shift_labels,
            allow_embedding_resizing=True,
        )

    def forward(self, batch) -> CausalLMOutputWithPast:
        """Forward pass through the model."""
        return self.model(
            input_ids=batch.get("input_ids", None),
            inputs_embeds=batch.get("inputs_embeds", None),
            attention_mask=batch.get("attention_mask", None),
        )

    def loss(self, outputs: CausalLMOutputWithPast, batch) -> torch.Tensor:
        """Compute training loss."""
        loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=CROSS_ENTROPY_IGNORE_INDEX,
            reduction="mean",
        )
        
        labels = batch["input_ids"] if "input_ids" in batch else batch["labels"]
        
        return compute_loss_from_logits(outputs, self.shift_labels, labels, loss_fn)
