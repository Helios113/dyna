from typing import Callable
from .attention_module import AttentionModule
import torch
import math

from jaxtyping import Float, Int, Bool
from torch import Tensor

class BasicAttn(AttentionModule):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        rotate_fraction: float = 1.0,
        rope_base: int = 10000,
    ):
        super().__init__(d_model, n_heads, d_head, rope_base)
        # Model configuration
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        identity_pytorch: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else identity_pytorch

        # Query and Key projections (shared)
        self.q = torch.nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.k = torch.nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.v = torch.nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.o = torch.nn.Linear(self.n_heads * self.d_head, self.d_model, bias=False)

        # RoPE configuration
        self.n_rotate = int(rotate_fraction * self.d_head)
       
        # # Attention scale
        # self.register_buffer(
        #     "scale",
        #     torch.full([1], 1.0 / math.sqrt(self.d_head)),
        #     persistent=False,
        # )

    def reset_parameters(self, std_scale: float) -> None:
        with torch.no_grad():
            # Initialize projection parameters
            torch.nn.init.normal_(self.k.weight, 0, std_scale / math.sqrt(self.d_model))
            torch.nn.init.normal_(self.q.weight, 0, std_scale / math.sqrt(self.d_model))
            torch.nn.init.normal_(self.v.weight, 0, std_scale / math.sqrt(self.d_model))
            # FIX: Use proper scaling for output projection
            torch.nn.init.normal_(
                self.o.weight, 0, std_scale / math.sqrt(self.n_heads * self.d_head)
            )

    def get_reg_loss(self) -> torch.Tensor:
        """Return zero for regularization loss since BasicAttn doesn't use expert routing."""
        return torch.tensor(0.0, device=self.q.weight.device)

    def forward(
        self,
        q_src: Float[Tensor, "batch seq d_model"],
        k_src: Float[Tensor, "batch seq d_model"],
        v_src: Float[Tensor, "batch seq d_model"],
        mask: tuple[Bool[Tensor, "batch seq seq"], Int[Tensor, "batch seq"]],
    ) -> tuple[
        Float[Tensor, "batch seq d_model"],
        tuple[None, None],
    ]:
        """Forward pass through the attention layer."""

        q: Float[Tensor, "batch seq n_heads*d_head"] = self.q(q_src)
        k: Float[Tensor, "batch seq n_heads*d_head"] = self.k(k_src)
        v: Float[Tensor, "batch seq n_heads*d_head"] = self.v(v_src)

        # Project to attention format

        q = self.project_to_torch_order(q)
        k = self.project_to_torch_order(k)
        v = self.project_to_torch_order(v)

        # Apply dropout
        q = self.dropout(q)

        # Apply attention
        res = self.attend(v, k, q, mask[0], mask[1])
        # Reshape result for output projection
        res = res.transpose(-2, -3).contiguous().view(res.shape[0], res.shape[2], -1)

        # Apply output projection
        out = self.o(res)
        return out, (None, None)