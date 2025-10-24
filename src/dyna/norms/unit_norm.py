import torch
from jaxtyping import Float
from torch import Tensor


class UnitNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 0.5,
        device=None,
    ):
        """Dynamic Tanh module with learnable parameters."""
        super().__init__()

    def forward(
        self, x: Float[Tensor, "batch seq d_model"]
    ) -> Float[Tensor, "batch seq d_model"]:
        return torch.nn.functional.normalize(x, dim=-1)
