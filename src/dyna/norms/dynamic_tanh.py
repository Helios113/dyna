import torch
from jaxtyping import Float
from torch import Tensor


class DynamicTanh(torch.nn.Module):
    def __init__(
        self, normalized_shape: int, channels_last: bool, alpha_init_value: float = 0.5
    ):
        """Dynamic Tanh module with learnable parameters."""
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = torch.nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(
        self, x: Float[Tensor, "batch seq d_model"]
    ) -> Float[Tensor, "batch seq d_model"]:
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def extra_repr(self):
        return f"""normalized_shape={self.normalized_shape},
        alpha_init_value={self.alpha_init_value},
        channels_last={self.channels_last}"""
