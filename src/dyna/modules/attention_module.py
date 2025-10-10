from __future__ import annotations

import math

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from . import DynaModule


def log_mean(x: Float[Tensor, "*batch dim"], dim: int = 0) -> Float[Tensor, "*batch"]:
    """Compute log of mean along specified dimension."""
    return x.logsumexp(dim) - math.log(x.shape[dim])


def entropy_l(log_probs: Float[Tensor, "*batch dim"]) -> Float[Tensor, "*batch"]:
    """Compute entropy from log probabilities."""
    return -(log_probs * log_probs.exp()).sum(-1)


def entropy_reg(sel: Float[Tensor, "*batch n_experts"], dim: int) -> Float[Tensor, ""]:
    """Compute entropy regularization term."""
    sel = torch.nn.functional.log_softmax(sel, dim=-1)
    sel = log_mean(sel, dim)
    return -entropy_l(sel).mean()


class AttentionModule(DynaModule):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        base: int = 10000,
        seq_dim: int = 1,
    ) -> None:
        """Base attention module with RoPE (Rotary Position Encoding) support.

        This module provides core attention functionality with optimized rotary
        position encoding, including sin/cos caching for improved performance.
        It serves as a base class for attention implementations.

        Args:
            d_model: Model dimension for computing rotary frequencies.
            n_heads: Number of attention heads.
            d_head: Dimension of each attention head.
            base: Base value for rotary position encoding frequency computation
                (default: 10000).
            seq_dim: Dimension index representing the sequence length in tensors
                (default: 1).
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for efficiency
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        self.seq_dim = torch.tensor(seq_dim)

        self.n_heads = n_heads
        self.d_head = d_head

    def attend(
        self,
        v: Float[Tensor, "batch n_heads seq d_head"],
        k: Float[Tensor, "batch n_heads seq d_head"],
        q: Float[Tensor, "batch n_heads seq d_head"],
        attention_mask: Bool[Tensor, "batch n_heads seq seq"],
        position_mask: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch n_heads seq d_head"]:
        """Compute attention with RoPE for constant length q k v tensors.

        Computes attention with multiple sequences per sample in the batch.
        FlashAttention requires q, k, v to be padded to the same length, so we
        use cu_seqlens to indicate the start of each sequence.

        Args:
            v: Value tensor of shape (batch, n_heads, seq, d_head).
            k: Key tensor of shape (batch, n_heads, seq, d_head).
            q: Query tensor of shape (batch, n_heads, seq, d_head).
            attention_mask: Attention mask tensor.
            position_mask: Position indices for RoPE of shape (batch, seq).

        Returns:
            Attention output tensor of shape (batch, n_heads, seq, d_head).
        """
        # Apply rotary position encoding
        # Remove debug print that could cause issues

        q, k = self._apply_rope(q, k, position_mask)

        # Explicitly remove scaling, as we scale in the body in the transformer
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, scale=1.0
        )

    def project_to_torch_order(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor to PyTorch attention format."""
        return x.view(*x.shape[:-1], self.n_heads, self.d_head).transpose(-2, -3)

    def apply_rot_optimized(
        self,
        x: torch.Tensor,  # [batch, n_heads, seq, d_head]
        positions: torch.Tensor,  # [batch, seq]
    ) -> torch.Tensor:
        """Optimized rotary position encoding application."""
        sin, cos = self.get_sincos_positions(positions, x)

        # Get sequence length once
        seq_len = x.shape[self.seq_dim]

        # Use slice instead of narrow (more readable, same performance)
        sin = sin[..., :seq_len, :]
        cos = cos[..., :seq_len, :]

        # Apply rotation in one line using the optimized rotate_half
        return x * cos + self.rotate_half_optimized(x) * sin

    def rotate_half_optimized(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized rotation of the second half of the last dimension."""
        # Avoid redundant shape calculation and use tensor.chunk for cleaner split
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=x1.ndim - 1)

    def get_sincos_positions(
        self,
        positions: Int[Tensor, "batch seq"],
        q: Float[Tensor, "batch n_heads seq d_head"],
    ) -> tuple[
        Float[Tensor, "batch 1 seq d_head"], Float[Tensor, "batch 1 seq d_head"]
    ]:
        """Get sin/cos values for specific positions."""
        seq_len = q.shape[self.seq_dim]

        # Check if we can reuse cached values
        if (
            self.seq_len_cached < seq_len
            or self.cos_cached is None
            or self.sin_cached is None
            or self.cos_cached.device != q.device
        ):
            # Create position indices for the maximum sequence length we might need
            max_pos = positions.max().item() + 1
            pos_idx = torch.arange(max_pos, device=q.device)

            # Compute frequencies
            freqs = torch.einsum("i,j->ij", pos_idx, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            # Cache sin/cos for future use
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
            self.seq_len_cached = max_pos

        # Extract sin/cos values for the specific positions
        cos_vals = self.cos_cached[positions]  # [batch, seq, d_head]
        sin_vals = self.sin_cached[positions]  # [batch, seq, d_head]

        # Reshape to match expected output format
        tgt_shape = [1] * q.ndim
        tgt_shape[0] = q.shape[0]
        tgt_shape[1] = 1
        tgt_shape[self.seq_dim] = q.shape[self.seq_dim]
        tgt_shape[-1] = q.shape[-1]

        return sin_vals.view(*tgt_shape), cos_vals.view(*tgt_shape)

    def _apply_rope(
        self,
        q: Float[Tensor, "batch n_heads seq d_head"],
        k: Float[Tensor, "batch n_heads seq d_head"],
        position_mask_full: Int[Tensor, "batch seq"],
    ) -> tuple[
        Float[Tensor, "batch n_heads seq d_head"],
        Float[Tensor, "batch n_heads seq d_head"],
    ]:
        return (
            self.apply_rot_optimized(q, position_mask_full),
            self.apply_rot_optimized(k, position_mask_full),
        )


# class DummyAttention(AttentionModule):
#     def __init__(
#         self,
#         d_model: int,
#         n_heads: int,
#         d_head: int,
#         dropout: float = 0.0,
#         rotate_fraction: float = 1.0,
#         rope_base: float = 10000,
#     ):
#         super().__init__()
#         # Model configuration
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.d_head = d_head
#         self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else lambda x: x

#         # Query and Key projections (shared)
#         self.q = torch.nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
#         self.k = torch.nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
#         self.v = torch.nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
#         self.o = torch.nn.Linear(self.n_heads * self.d_head, self.d_model, bias=False)

#         # RoPE configuration
#         self.n_rotate = int(rotate_fraction * self.d_head)
#         if self.n_rotate > 0:
#             self.pe = RotaryPosEncoding(self.n_rotate, seq_dim=-2, base=rope_base)

#         # This might be it?
#         # The scaled dot prod has a scale of 1 because k and v were rescaled
#         # # Attention scale
#         # self.register_buffer(
#         #     "scale",
#         #     torch.full([1], 1.0 / math.sqrt(self.d_head)),
#         #     persistent=False,
#         # )

#     @torch.no_grad
#     def reset_parameters(self, std_scale: float) -> None:
#         # Initialize projection parameters
#         torch.nn.init.normal_(self.k.weight, 0, std_scale / math.sqrt(self.d_model))
#         torch.nn.init.normal_(self.q.weight, 0, std_scale / math.sqrt(self.d_model))
#         torch.nn.init.normal_(self.v.weight, 0, std_scale / math.sqrt(self.d_model))
#         # FIX: Use proper scaling for output projection
#         torch.nn.init.normal_(
#             self.o.weight, 0, std_scale / math.sqrt(self.n_heads * self.d_head)
#         )

#     def forward(
#         self,
#         q_src: Float[Tensor, "batch seq d_model"],
#         k_src: Float[Tensor, "batch seq d_model"],
#         v_src: Float[Tensor, "batch seq d_model"],
#         mask: tuple[Bool[Tensor, "batch seq seq"], Int[Tensor, "batch seq"]],
#         continue_mask: None | Bool[Tensor, "batch seq"],
#     ) -> tuple[
#         Float[Tensor, "batch seq d_model"],
#         tuple[None, None],
#     ]:

#         return q_src, (None, None)

#     def get_reg_loss(self) -> torch.Tensor:
#         """Return zero for regularization loss since DummyAttention doesn't use expert routing."""
#         return torch.tensor(0.0, device=self.q.weight.device, dtype=self.q.weight.dtype)
