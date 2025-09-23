from typing import Callable
from dyna.modules.dyna_module import DynaModule
import torch
import math

class BasicFFN(DynaModule):
    def __init__(
        self,
        d_model: int,
        d_expert_ffn: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.gelu,
    ):
        super().__init__() # pyright: ignore[reportUnknownMemberType]
        self.d_model = d_model
        self.d_expert_ffn = d_expert_ffn
        self.activation = activation
        self.projection_up = torch.nn.Linear(self.d_model, self.d_expert_ffn)
        self.projection_down = torch.nn.Linear(self.d_expert_ffn, self.d_model)

    def forward(
        self, token_stream: torch.Tensor, selection_input: torch.Tensor
    ) -> tuple[torch.Tensor, None]:  # Match return type with SigmaMoE
        output = self.projection_down(self.activation(self.projection_up(token_stream)))
        return output, None  # Return None for the selection index to match SigmaMoE

    def reset_parameters(self, std_scale: float) -> None:
        with torch.no_grad():
            """Initialize parameters with proper scaling."""
            torch.nn.init.normal_(
                self.projection_up.weight, 0, std_scale / math.sqrt(self.d_model)
            )
            torch.nn.init.normal_(
                self.projection_down.weight, 0, std_scale / math.sqrt(self.d_expert_ffn)
            )
            torch.nn.init.zeros_(self.projection_up.bias)
            torch.nn.init.zeros_(self.projection_down.bias)

    def get_reg_loss(self) -> torch.Tensor:
        """Return zero for regularization loss since BasicFFN doesn't use expert routing."""
        return torch.tensor(0.0, device=self.projection_up.weight.device)
