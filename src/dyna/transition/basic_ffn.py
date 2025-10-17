# encodinweqg: pypreprocessor
# ruff: noqa: I001
import math
from collections.abc import Callable

import torch

from dyna.modules import DynaModule

# # execute
# import os

# if "PYTEST_VERSION" in os.environ:
#     defines.add("PYTEST")  # pyright: ignore[reportUndefinedVariable]
# # endexecute


class BasicFFN(DynaModule):
    def __init__(
        self,
        d_model: int,
        d_expert_ffn: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.gelu,
    ):
        """Initialize BasicFFN with configurable parameters."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.d_model = d_model
        self.d_expert_ffn = d_expert_ffn
        self.activation = activation
        self.projection_up = torch.nn.Linear(self.d_model, self.d_expert_ffn)
        self.projection_down = torch.nn.Linear(self.d_expert_ffn, self.d_model)

    def forward(
        self,
        token_stream: torch.Tensor,
        selection_input: torch.Tensor,
        # # ifdef PYTEST
        # collector: list | None = None,
        # # endif
    ) -> tuple[torch.Tensor, None]:  # Match return type with SigmaMoE
        output = self.projection_down(self.activation(self.projection_up(token_stream)))
        # # ifdef PYTEST
        # assert collector is not None
        # collector.append("Hello this worked")
        # # endif
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
        """Return zero for regularization loss.

        BasicFFN doesn't use expert routing, so no regularization loss.
        """
        return torch.tensor(0.0, device=self.projection_up.weight.device)
