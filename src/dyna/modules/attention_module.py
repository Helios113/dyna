import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from . import DynaModule


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
        q: Float[Tensor, "batch n_heads seq d_head"],
        k: Float[Tensor, "batch n_heads seq d_head"],
        v: Float[Tensor, "batch n_heads seq d_head"],
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
        elif not scale_qk:
            scale = 1 / self.d_head
        else:
            scale = 1

        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0,
            scale=scale,
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
