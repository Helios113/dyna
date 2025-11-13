from collections.abc import Callable

import torch

from dyna.modules import DynaModule


class BasicFFN(DynaModule):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.gelu,
    ):
        """Initialize BasicFFN with configurable parameters."""
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.activation = activation
        self.projection_up = torch.nn.Linear(self.d_model, self.d_ffn)
        self.projection_down = torch.nn.Linear(self.d_ffn, self.d_model)

    def forward(
        self,
        token_stream: torch.Tensor,
        selection_input: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:  # Match return type with SigmaMoE
        up_output = self.projection_up(token_stream)
        activation_output = self.activation(up_output)
        down_output = self.projection_down(activation_output)
        return (down_output, None)

    def reset_parameters(self, ffn_scale: float, attn_scale: float) -> None:
        torch.nn.init.normal_(self.projection_up.weight, 0, ffn_scale)
        torch.nn.init.normal_(self.projection_down.weight, 0, ffn_scale)
        torch.nn.init.zeros_(self.projection_up.bias)
        torch.nn.init.zeros_(self.projection_down.bias)

    def get_reg_loss(self) -> torch.Tensor:
        """Return zero for regularization loss.

        BasicFFN doesn't use expert routing, so no regularization loss.
        """
        return torch.tensor(0.0, device=self.projection_up.weight.device)

    def clear_selection_history(self):
        return
