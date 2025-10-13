from __future__ import annotations

import math

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from dyna.cvmm import CVMMSel, cvmm, cvmm_prepare_sel2
from dyna.modules import AttentionModule


class SwitchHead(AttentionModule):
    """Core attention mechanism with expert routing."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_experts_attn: int,
        d_head: int,
        dropout: float = 0.0,
        dropout_expert: float = 0.0,
        k_attn: int = 2,
        n_expert_shared_attn: int = 0,
        rotate_fraction: float = 1,
        rope_base: int = 10000,
    ):
        """Initialize SwitchHead with expert routing configuration."""
        super().__init__(d_model, n_heads, d_head, rope_base)
        # Model configuration
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

        def identity_pytorch(x: torch.Tensor) -> torch.Tensor:
            return x

        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else identity_pytorch

        # Expert configuration
        self.n_experts_attn = n_experts_attn
        self.dropout_expert = dropout_expert
        self.k_attn = k_attn
        self.n_expert_shared_attn = min(n_expert_shared_attn, n_experts_attn)
        self.n_expert_routed_attn = n_experts_attn - self.n_expert_shared_attn

        # Bias tracking
        self.bias_update_lr = 0.001

        # Query and Key projections (shared)
        self.q = torch.nn.Linear(self.d_model, self.d_head * self.n_heads, bias=False)
        self.k = torch.nn.Linear(self.d_model, self.d_head * self.n_heads, bias=False)

        # Expert-specific parameters
        self._init_expert_parameters()
        # Shared expert indices
        self.register_buffer(
            "expert_shared",
            torch.arange(
                n_experts_attn - self.n_expert_shared_attn,
                n_experts_attn,
                dtype=torch.long,
            ),
        )

        # Attention scale
        self.register_buffer(
            "scale",
            torch.full([1], 1.0 / math.sqrt(self.d_head)),
            persistent=False,
        )

        # Tracking variables for visualization
        self.selections_to_visualize = {}
        # self.sel_hist:list[torch.Tensor]

        self.call_h = 0

    def _init_expert_parameters(self) -> None:
        """Initialize expert-specific parameters."""
        # Value and output projections for multiple experts
        self.v = torch.nn.Parameter(
            torch.empty(self.n_heads * self.n_experts_attn, self.d_model, self.d_head)
        )
        self.o = torch.nn.Parameter(
            torch.empty(self.n_heads * self.n_experts_attn, self.d_head, self.d_model)
        )

        # Expert selection parameters
        self.sel_v = torch.nn.Parameter(
            torch.empty(self.n_heads * self.n_experts_attn, self.d_model)
        )
        self.sel_o = torch.nn.Parameter(
            torch.empty(self.n_heads * self.n_experts_attn, self.d_model)
        )

        # Bias parameters for load balancing
        self.bias_v = torch.nn.Parameter(
            torch.zeros(self.n_experts_attn), requires_grad=False
        )
        self.bias_o = torch.nn.Parameter(
            torch.zeros(self.n_experts_attn), requires_grad=False
        )

    def reset_parameters(self, std_scale: float) -> None:
        with torch.no_grad():
            """Initialize all parameters with proper scaling."""
            # Initialize selection parameters
            if self.n_experts_attn > 1:
                torch.nn.init.normal_(
                    self.sel_v, 0, std_scale / math.sqrt(self.d_model)
                )
                self.renorm_rows(self.sel_v)

            torch.nn.init.normal_(self.sel_o, 0, std_scale / math.sqrt(self.d_model))
            self.renorm_rows(self.sel_o)

            # Initialize projection parameters
            torch.nn.init.normal_(self.k.weight, 0, std_scale / math.sqrt(self.d_model))
            torch.nn.init.normal_(self.q.weight, 0, std_scale / math.sqrt(self.d_model))
            torch.nn.init.normal_(self.v, 0, std_scale / math.sqrt(self.d_model))
            torch.nn.init.normal_(
                self.o, 0, std_scale / math.sqrt(self.n_heads * self.d_head)
            )

    def renorm_rows(self, x: torch.Tensor) -> None:
        """Renormalize rows while preserving standard deviation."""
        with torch.no_grad():
            std_t = x.std(dim=-1, keepdim=True)
            x.div_(x.norm(dim=-1, keepdim=True))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            x.mul_(std_t / x.std())

    # def get_reg_loss(self) -> Float[Tensor, "dim"]:
    #     """Get regularization loss from selection history."""
    #     loss = torch.tensor(0.0, device=next(self.parameters()).device)
    #     if self.sel_hist:
    #         for i in range(len(self.sel_hist[0])):
    #             loss = loss + entropy_reg(
    #                 torch.stack([l[i] for l in self.sel_hist], dim=-3)
    #                 .flatten(-4, -3),
    #                 -3,
    #             )
    #     # Clear the history to prevent memory accumulation
    #     self.sel_hist.clear()
    #     return loss

    def _get_expert_selection(
        self,
        input_tensor: Float[Tensor, "batch seq d_model"],
        weight: Float[Tensor, "n_heads_x_experts d_model"],
        bias: Float[Tensor, " n_experts_attn"] | None = None,
    ) -> tuple[
        CVMMSel,
        Float[Tensor, "batch seq n_heads n_experts_attn"],
        Int[Tensor, "batch seq n_heads k_experts"],
    ]:
        """Get expert selection indices and weights."""
        # Compute selection scores - remove explicit float() cast
        affinity: Float[Tensor, "batch seq n_heads_x_experts"] = (
            torch.nn.functional.linear(input_tensor, weight)
        )
        affinity_raw: Float[Tensor, "batch seq n_heads n_experts"] = affinity.view(
            *affinity.shape[:-1], self.n_heads, -1
        )
        affinity = affinity_raw.sigmoid()

        # Apply expert dropout
        if self.dropout_expert > 0 and self.training:
            mask = torch.rand_like(affinity) < self.dropout_expert
            affinity_2 = affinity.masked_fill(mask, float("-inf"))
        else:
            affinity_2 = affinity

        # Select routed experts
        routed_k = max(1, self.k_attn - self.n_expert_shared_attn)
        bias_term = bias[: self.n_expert_routed_attn] if bias is not None else None

        _, sel_index = torch.topk(
            (
                (affinity_2[:, :, :, : self.n_expert_routed_attn] + bias_term)
                if bias_term is not None
                else affinity_2[:, :, :, : self.n_expert_routed_attn]
            ),
            routed_k,
            dim=-1,
            sorted=False,
        )

        # Add shared experts
        if self.n_expert_shared_attn > 0:
            shared_shape = sel_index.shape[:-1] + (self.n_expert_shared_attn,)
            expert_shared_expanded = self.expert_shared.view(
                ([1] * (sel_index.dim() - 1)), -1
            ).expand(*shared_shape)

            sel_index = torch.cat([sel_index, expert_shared_expanded], dim=-1)

        # Update bias for load balancing
        if self.training and bias is not None:
            with torch.no_grad():
                c_i = torch.bincount(sel_index.flatten(), minlength=self.n_experts_attn)
                c_i_avg = torch.mean(c_i, dtype=torch.float32)
                bias[: self.n_expert_routed_attn] = bias[
                    : self.n_expert_routed_attn
                ] + self.bias_update_lr * torch.sign(
                    -c_i[: self.n_expert_routed_attn] + c_i_avg
                )

        # Get selection values and create CVMM selection object
        sel_val: Float[Tensor, "batch seq n_heads k_experts"] = torch.gather(
            affinity.view(*affinity.shape[:-2], -1),
            -1,
            sel_index.view(*sel_index.shape[:-2], -1),
        ).view(*sel_index.shape)

        # Create shifted indices for expert matrix operations
        sel_index_shifted: Int[Tensor, "batch seq n_heads k_experts"] = (
            torch.arange(self.n_heads, device=sel_index.device, dtype=sel_index.dtype)
            * self.n_experts_attn
        ).unsqueeze(-1) + sel_index

        return (
            cvmm_prepare_sel2(sel_index_shifted.flatten(-2, -1), sel_val),
            affinity_raw,
            sel_index,
        )

    def forward(
        self,
        q_src: Float[Tensor, "batch seq d_model"],
        k_src: Float[Tensor, "batch seq d_model"],
        v_src: Float[Tensor, "batch seq d_model"],
        mask: tuple[Bool[Tensor, "batch seq seq"], Int[Tensor, "batch seq"]],
    ) -> tuple[
        Float[Tensor, "batch seq d_model"],
        tuple[
            Int[Tensor, "batch seq n_heads k_experts"],
            Int[Tensor, "batch seq n_heads k_experts"],
        ],
    ]:
        """Forward pass through the attention layer."""
        # Apply scaling to queries and keys
        q: Float[Tensor, "batch seq d_model"] = self.q(q_src)
        k: Float[Tensor, "batch seq d_model"] = self.k(k_src)
        v_sel_index = None
        o_sel_inedx = None

        # Handle expert routing for values and outputs
        if self.n_experts_attn > 1:
            v_sel, v_sel_r, v_sel_index = self._get_expert_selection(
                k_src, self.sel_v, self.bias_v
            )

            o_sel, o_sel_r, o_sel_inedx = self._get_expert_selection(  # pyright: ignore[reportUnusedVariable]
                q_src, self.sel_o, self.bias_o
            )
            # Commented for mem reduction
            # if self.training:
            #     self.sel_hist.append((o_sel_r, v_sel_r))
            v: Float[Tensor, "batch n_heads seq d_head"] = cvmm(
                v_src, v_sel, self.v
            ).transpose(-2, -3)

            # Clean up intermediate tensors
            del v_sel_r, v_sel
            # Project to attention format
            q: Float[Tensor, "batch n_heads seq d_head"] = self.project_to_torch_order(
                q
            )
            k: Float[Tensor, "batch n_heads seq d_head"] = self.project_to_torch_order(
                k
            )

            # Apply dropout and attention
            q = self.dropout(q)

            res: Float[Tensor, "batch n_heads seq d_head"] = self.attend(
                v, k, q, mask[0], mask[1]
            )
            res = res.transpose(-2, -3)
            o_sel.sel_index = o_sel.out_index // o_sel.reduction_weight.shape[-1]
            o_sel.reduction_weight = o_sel.reduction_weight.flatten(-2)
            out: Float[Tensor, "batch seq d_model"] = cvmm(res, o_sel, self.o)
        else:
            o_gate: Float[Tensor, "batch seq d_model"] = torch.nn.functional.sigmoid(
                torch.nn.functional.linear(q_src, self.sel_o)
            )

            v = torch.einsum("bsd,ndh->bsnh", v_src, self.v)
            v = self.project_to_torch_order(v.reshape(v.shape[0], v.shape[1], -1))
            q: Float[Tensor, "batch n_heads seq d_head"] = self.project_to_torch_order(
                q
            )
            k: Float[Tensor, "batch n_heads seq d_head"] = self.project_to_torch_order(
                k
            )

            # Apply dropout and attention
            q = self.dropout(q)

            res: Float[Tensor, "batch n_heads seq d_head"] = self.attend(
                v, k, q, mask[0], mask[1]
            )
            res = res.transpose(-2, -3)

            res = res * o_gate[..., None]

            res = res.view(
                res.shape[0],
                res.shape[1],
                self.n_heads * self.n_experts_attn,
                self.d_head,
            )
            out = torch.einsum("bsnh,nhd->bsd", res, self.o)
            v_sel_index = torch.zeros_like(res, dtype=torch.int32)
            o_sel_inedx = torch.zeros_like(res, dtype=torch.int32)

        assert isinstance(out, torch.Tensor)
        assert isinstance(v_sel_index, torch.Tensor)
        assert isinstance(o_sel_inedx, torch.Tensor)

        return out, (v_sel_index.detach().cpu(), o_sel_inedx.detach().cpu())
