from collections.abc import Callable

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from . import DynaModule


def custom_softmax(x, dim=-1):
    """Custom softmax implementation with numerical stability.

    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax

    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_exp = torch.exp((x - x_max).to(torch.float32)).to(x.dtype)

    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    out = x_exp / x_sum
    if out.isnan().any():
        print("NaN detected in output", flush=True)

        # Find locations of NaNs
        nan_mask = out.isnan()
        nan_indices = torch.nonzero(nan_mask, as_tuple=False)

        print(f"\nNumber of NaN values: {nan_mask.sum()}", flush=True)
        print(f"NaN locations (first 10): {nan_indices[:10]}", flush=True)

        # For each NaN location, print the corresponding x_exp and x_sum values
        for i, idx in enumerate(nan_indices[:10]):  # Print first 10 cases
            idx_tuple = tuple(idx.tolist())
            print(f"\n--- NaN case {i + 1} at index {idx_tuple} ---", flush=True)
            print(f"out value: {out[idx_tuple]}", flush=True)
            print(f"x_exp value: {x_exp[idx_tuple]}", flush=True)

            # Get the corresponding x_sum (which is summed along dim)
            # Create index for x_sum (which has keepdim=True)
            sum_idx = list(idx_tuple)
            sum_idx[dim] = 0  # x_sum has size 1 along the summed dimension
            sum_idx_tuple = tuple(sum_idx)
            print(f"x_sum value: {x_sum[sum_idx_tuple]}", flush=True)

            # Also print the original x values along that dimension
            slice_idx = list(idx_tuple)
            slice_idx[dim] = slice(None)  # Get all values along dim
            print(f"x values along dim: {x[tuple(slice_idx)]}", flush=True)
            print(f"x_exp values along dim: {x_exp[tuple(slice_idx)]}", flush=True)

        exit()
    return out


def custom_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = 1.0,
    training: bool = True,
    custom_softmax_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Custom implementation of scaled dot product attention.

    Args:
        query: Query tensor of shape (batch, num_heads, seq_len_q, head_dim)
        key: Key tensor of shape (batch, num_heads, seq_len_k, head_dim)
        value: Value tensor of shape (batch, num_heads, seq_len_k, head_dim)
        attention_mask: Optional attention mask. Can be 2D or 4D.
                        Use float('-inf') for positions to mask out.
        dropout_p: Dropout probability (default: 0.0)
        is_causal: If True, applies causal (lower triangular) masking
        training: Whether in training mode (affects dropout)
        custom_softmax_fn: Custom softmax function to use

    Returns:
        Output tensor of shape (batch, num_heads, seq_len_q, head_dim)
    """
    if custom_softmax_fn is None:
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            scale=scale,
        )

    # Get dimensions
    batch_size, num_heads, seq_len_q, head_dim = query.shape
    seq_len_k = key.shape[2]

    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply causal mask if specified
    if attention_mask is not None:
        attn_scores = attn_scores.masked_fill(attention_mask, float("-inf"))
    elif is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

    # Apply additional attention mask if provided

    # Apply custom softmax
    attn_weights = custom_softmax(attn_scores, dim=-1)

    # Apply dropout if specified
    if dropout_p > 0.0 and training:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)

    # Compute attention output: Attention @ V
    output = torch.matmul(attn_weights, value)

    return output


class AttentionModule(DynaModule):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        base: int = 10000,
        nope_pos: bool = False,
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
            nope_pos: Whether to use nope position encoding.
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Cache for efficiency
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        self.seq_dim: int = 2

        self.n_heads = n_heads
        self.d_head = d_head
        self.nope_pos = nope_pos
        # self.softmax_fn = custom_softmax
        self.softmax_fn = None
        if self.softmax_fn is not None:
            print("!!!! We are using custom softmax function !!!!", flush=True)

    def attend(
        self,
        v: Float[Tensor, "batch n_heads seq d_head"],
        k: Float[Tensor, "batch n_heads seq d_head"],
        q: Float[Tensor, "batch n_heads seq d_head"],
        attention_mask: Bool[Tensor, "batch 1 seq seq"],
        sequence_length: Int[Tensor, "batch seq"],
        sqrt_attention_scale: bool = False,
        scale_qk: bool = False,
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
            sequence_length: Position indices for RoPE of shape (batch, seq).
            nope_pos: where to apply NoPe position encoding
            manual_scale: whether to manually scale the attention scores

        Returns:
            Attention output tensor of shape (batch, n_heads, seq, d_head).
        """
        if not self.nope_pos:
            q, k = self._apply_rope(q, k, sequence_length)
        if sqrt_attention_scale:
            scale = 1 / torch.sqrt(torch.tensor(self.d_head, device=self.device))
        elif scale_qk == False:
            scale = 1 / self.d_head
        else:
            scale = 1

        return custom_scaled_dot_product_attention(
            q,
            k,
            v,
            attention_mask=attention_mask,
            scale=scale,
            custom_softmax_fn=self.softmax_fn,
        )

    def update_inv_freq(self, base: int) -> None:
        self.inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.d_head, 2, device=self.device).float()
                / self.d_head
            )
        )
        self.seq_len_cached = 0

    def project_to_torch_order(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor to PyTorch attention format."""
        return x.view(*x.shape[:-1], self.n_heads, self.d_head).transpose(-2, -3)

    def apply_rot(
        self,
        x: Float[Tensor, "batch n_heads seq d_head"],
        positions: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch n_heads seq d_head"]:
        """Optimized rotary position encoding application."""
        sin, cos = self.get_sincos_positions(x, positions)

        # Get sequence length once
        seq_len = x.shape[self.seq_dim]

        # Use slice instead of narrow (more readable, same performance)
        sin = sin[..., :seq_len, :]
        cos = cos[..., :seq_len, :]
        return x * cos + self.rotate_half_optimized(x) * sin

    def rotate_half_optimized(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized rotation of the second half of the last dimension."""
        # Avoid redundant shape calculation and use tensor.chunk for cleaner split
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=x1.ndim - 1)

    def get_sincos_positions(
        self,
        x: Float[Tensor, "batch n_heads seq d_head"],
        positions: Int[Tensor, "batch seq"],
    ) -> tuple[
        Float[Tensor, "batch 1 seq d_head"],
        Float[Tensor, "batch 1 seq d_head"],
    ]:
        """Get sin/cos values for specific positions."""
        seq_len = x.shape[self.seq_dim]

        # Check if we can reuse cached values
        if (
            self.seq_len_cached < seq_len
            or self.cos_cached is None
            or self.sin_cached is None
            or self.cos_cached.device != x.device
        ):
            # Create position indices for the maximum sequence length we might need
            max_pos = positions.max().item() + 1
            pos_idx = torch.arange(max_pos, device=x.device)

            # Compute frequencies
            freqs: Float[Tensor, "max_pos d_head"] = torch.einsum(
                "i,j->ij", pos_idx, self.inv_freq
            )

            emb = torch.cat((freqs, freqs), dim=-1)

            # Cache sin/cos for future use
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
            self.seq_len_cached = max_pos

        # Extract sin/cos values for the specific positions
        cos_vals = self.cos_cached[positions]  # [batch, seq, d_head]
        sin_vals = self.sin_cached[positions]  # [batch, seq, d_head]

        # Reshape to match expected output format
        tgt_shape = [1] * (x.ndim)
        tgt_shape[0] = x.shape[0]
        tgt_shape[1] = 1
        tgt_shape[self.seq_dim] = x.shape[self.seq_dim]
        tgt_shape[-1] = x.shape[-1]

        return sin_vals.view(*tgt_shape), cos_vals.view(*tgt_shape)

    def _apply_rope(
        self,
        q: Float[Tensor, "batch n_heads seq d_head"],
        k: Float[Tensor, "batch n_heads seq d_head"],
        sequence_length_full: Int[Tensor, "batch seq"],
    ) -> tuple[
        Float[Tensor, "batch n_heads seq d_head"],
        Float[Tensor, "batch n_heads seq d_head"],
    ]:
        return (
            self.apply_rot(q, sequence_length_full),
            self.apply_rot(k, sequence_length_full),
        )
