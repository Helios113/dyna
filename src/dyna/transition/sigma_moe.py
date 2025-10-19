import math
from collections.abc import Callable

import torch
from jaxtyping import Float, Int
from torch import Tensor

from dyna.kernel.cvmm import cvmm, cvmm_prepare_sel2
from dyna.modules import entropy_reg
from dyna.modules.dyna_module import DynaModule


class SigmaMoE(DynaModule):
    """Sigma Mixture of Experts layer for feed-forward networks."""

    def __init__(
        self,
        d_model: int,
        n_experts_ffn: int,
        d_expert_ffn: int,
        n_expert_shared_ffn: int,
        k_ffn: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.gelu,
        dropout_expert: float = 0.0,
        use_bias: bool = True,
    ):
        """Initialize SigmaMoE with configurable parameters."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.d_model = d_model
        self.n_experts_ffn = n_experts_ffn
        self.d_expert_ffn = d_expert_ffn
        self.use_bias = use_bias
        self.n_expert_shared_ffn = min(n_expert_shared_ffn, n_experts_ffn)
        self.n_expert_routed_ffn = n_experts_ffn - self.n_expert_shared_ffn
        self.k_ffn = k_ffn
        self.activation = activation
        self.dropout_expert = dropout_expert
        self.bias_ffn = (
            torch.nn.Parameter(torch.zeros(n_experts_ffn), requires_grad=False)
            if self.n_expert_routed_ffn > 0
            else None
        )
        # Bias tracking for load balancing
        self.bias_update_lr = 0.001

        # Expert parameters
        self.keys: torch.Tensor = torch.nn.Parameter(
            torch.empty(self.n_experts_ffn, self.d_model, self.d_expert_ffn)
        )
        self.values: torch.Tensor = torch.nn.Parameter(
            torch.empty(self.n_experts_ffn, self.d_expert_ffn, self.d_model)
        )
        self.expert_sel: torch.Tensor = torch.nn.Parameter(
            torch.empty(self.n_experts_ffn, self.d_model)
        )

        # Register shared expert indices
        self.expert_shared = torch.nn.Parameter(
            torch.arange(
                n_experts_ffn - self.n_expert_shared_ffn,
                n_experts_ffn,
                dtype=torch.long,
            ),
            requires_grad=False,
        )

        self.sel_hist = []

    def reset_parameters(self, std_scale: float) -> None:
        """Initialize parameters with proper scaling."""
        torch.nn.init.normal_(self.keys, 0, std_scale / math.sqrt(self.d_model))
        torch.nn.init.normal_(
            self.values,
            0,
            std_scale / math.sqrt(self.n_experts_ffn * self.d_expert_ffn),
        )
        torch.nn.init.normal_(self.expert_sel, 0, std_scale / math.sqrt(self.d_model))
        self.renorm_keep_std(self.expert_sel, dim=1)

    def renorm_keep_std(self, weight: torch.Tensor, dim: int = 0) -> None:
        """Renormalize weights while keeping standard deviation."""
        with torch.no_grad():
            std = weight.std()
            weight.div_(weight.norm(dim=dim, keepdim=True))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            weight.mul_(std / weight.std())

    def _compute_expert_selection(
        self, selection_input: Float[Tensor, "batch seq d_model"]
    ) -> tuple[
        Float[Tensor, "batch seq k_experts"], Int[Tensor, "batch seq k_experts"]
    ]:
        """Compute expert selection scores and indices."""
        if self.n_experts_ffn == 1:
            return torch.ones(
                (selection_input.shape[0], selection_input.shape[1], 1),
                device=selection_input.device,  # Ensure tensor is on the same device
            ), torch.zeros(
                (selection_input.shape[0], selection_input.shape[1], 1),
                dtype=torch.int32,
                device=selection_input.device,  # Ensure tensor is on the same device
            )
        # Compute selection scores
        affinity: Float[Tensor, "batch seq n_experts"] = torch.nn.functional.sigmoid(
            torch.nn.functional.linear(
                selection_input,
                self.expert_sel,
            )
        )
        # Apply dropout
        if self.training and self.dropout_expert > 0:
            mask = torch.rand_like(affinity) < self.dropout_expert
            affinity.masked_fill_(mask, float("-inf"))

        bias_term = (
            self.bias_ffn[: self.n_expert_routed_ffn]
            if self.bias_ffn is not None
            else None
        )

        # Select top-k routed experts, but ensure k doesn't exceed available experts
        assert self.k_ffn < self.n_expert_routed_ffn
        _, selection_index = torch.topk(
            (
                (affinity[:, :, : self.n_expert_routed_ffn] + bias_term)
                if bias_term is not None
                else affinity[:, :, : self.n_expert_routed_ffn]
            ),
            self.k_ffn,
            dim=-1,
            sorted=False,
        )

        # Add shared experts
        if self.n_expert_shared_ffn > 0:
            shape_expert_shared = (
                *selection_index.shape[:-1],
                self.n_expert_shared_ffn,
            )
            expert_shared_expanded = self.expert_shared.view(
                *([1] * (selection_index.dim() - 1)), -1
            ).expand(shape_expert_shared)
            selection_index = torch.cat(
                [selection_index, expert_shared_expanded], dim=-1
            )

        # Gather affinities for selected experts
        affinity = torch.gather(affinity, -1, selection_index)

        # Update bias for load balancing during training
        if self.use_bias and self.training and self.n_expert_routed_ffn > 0:
            assert self.bias_ffn is not None  # For type checker
            with torch.no_grad():  # Prevent gradient accumulation
                c_i = torch.bincount(
                    selection_index.flatten(), minlength=self.n_experts_ffn
                )
                c_i_avg = torch.mean(c_i, dtype=torch.float32)
                self.bias_ffn[: self.n_expert_routed_ffn] = self.bias_ffn[
                    : self.n_expert_routed_ffn
                ] + self.bias_update_lr * torch.sign(
                    -c_i[: self.n_expert_routed_ffn] + c_i_avg
                )

        return affinity, selection_index

    def forward(
        self,
        token_stream: Float[Tensor, "batch seq d_model"],
        selection_input: Float[Tensor, "batch seq d_model"],
    ) -> tuple[Float[Tensor, "batch seq d_model"], Int[Tensor, "batch seq k_experts"]]:
        """Forward pass through the MoE layer."""
        # Get expert selection
        affinity, selection_index = self._compute_expert_selection(selection_input)
        if self.training:
            self.sel_hist.append(affinity)
        # Detach to avoid storing gradients

        # Prepare selection indices for CVMM operations
        selection_indices = cvmm_prepare_sel2(selection_index.int())

        scores: Float[Tensor, "batch seq k_experts d_expert"] = cvmm(
            token_stream, selection_indices, self.keys
        )
        scores = self.activation(scores)

        # Down-projection: scores * expert_values
        selection_indices.reduction_weight = affinity
        selection_indices.sel_index = selection_indices.out_index
        selection_indices.out_index = None

        out = cvmm(scores, selection_indices, self.values)

        # Clean up intermediate tensors to prevent memory leak
        del scores, selection_indices

        return out.view_as(token_stream), selection_index

    def get_reg_loss(self) -> Float[Tensor, ""]:
        # Average over time and layers
        loss = entropy_reg(torch.stack(self.sel_hist, dim=-2).flatten(-3, -2), -2)
        # Clear the history to prevent memory accumulation
        self.sel_hist.clear()
        return loss
