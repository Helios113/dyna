import math

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from dyna.modules import AttentionModule


class BasicAttn(AttentionModule):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        rotate_fraction: float = 1.0,
        rope_base: int = 10000,
        nope_pos: bool = False,
    ):
        """Initialize BasicAttn with configurable parameters."""
        super().__init__(d_model, n_heads, d_head, rope_base, nope_pos)
        # Model configuration
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

        def identity_pytorch(x: torch.Tensor) -> torch.Tensor:
            return x

        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else identity_pytorch

        # Query and Key projections (shared)
        self.q = torch.nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.k = torch.nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.v = torch.nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.o = torch.nn.Linear(self.n_heads * self.d_head, self.d_model, bias=False)

        # RoPE configuration
        self.n_rotate = int(rotate_fraction * self.d_head)

    def reset_parameters(self, ffn_scale: float, attn_scale: float) -> None:
        # Initialize projection parameters
        torch.nn.init.normal_(
            self.k.weight, 0, attn_scale * (1 / math.sqrt(self.n_heads * self.d_head))
        )
        torch.nn.init.normal_(
            self.q.weight, 0, attn_scale * (1 / math.sqrt(self.n_heads * self.d_head))
        )
        torch.nn.init.normal_(
            self.v.weight, 0, attn_scale * (1 / math.sqrt(self.n_heads * self.d_head))
        )
        # FIX: Use proper scaling for output projection
        torch.nn.init.normal_(
            self.o.weight, 0, attn_scale * (1 / math.sqrt(self.d_model))
        )

    def get_reg_loss(self) -> torch.Tensor:
        """Return zero for regularization loss.

        BasicAttn doesn't use expert routing, so no regularization loss.
        """
        return torch.tensor(0.0, device=self.q.weight.device)

    def forward(
        self,
        q_src: Float[Tensor, "batch seq d_model"],
        k_src: Float[Tensor, "batch seq d_model"],
        v_src: Float[Tensor, "batch seq d_model"],
        attention_mask: Bool[Tensor, "batch 1 seq seq"],
        sequence_length: Int[Tensor, "batch seq"],
    ) -> tuple[
        Float[Tensor, "batch seq d_model"],
        tuple[None, None],
    ]:
        """Forward pass through the attention layer."""
        q: Float[Tensor, "batch seq nd_head"] = self.q(q_src)
        k: Float[Tensor, "batch seq nd_head"] = self.k(k_src)
        v: Float[Tensor, "batch seq nd_head"] = self.v(v_src)

        # Project to attention format

        q = self.project_to_torch_order(q)
        k = self.project_to_torch_order(k)
        v = self.project_to_torch_order(v)

        # Apply dropout
        # q = self.dropout(q)

        # Apply attention
        res = self.attend(q, k, v, attention_mask, sequence_length)

        # Reshape result for output projection
        res = res.transpose(-2, -3).contiguous().view(res.shape[0], res.shape[2], -1)

        # Apply output projection
        out = self.o(res)

        return out, (None, None)

    def clear_selection_history(self):
        return
