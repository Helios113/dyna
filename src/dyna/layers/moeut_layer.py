import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from dyna.attention import SwitchHead
from dyna.config import DynaConfig
from dyna.modules import LayerModule, SaturationGate
from dyna.transition import SigmaMoE


class MoEUTLayer(LayerModule):
    """Single layer of the MoEUT model with configurable behavior."""

    def __init__(self, config: DynaConfig, input_reinjection: bool = False):
        """Initialize MoEUTLayer with configurable parameters."""
        super().__init__(
            config=config,
            attention_module=SwitchHead(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_head=config.d_head,
                n_experts_attn=config.n_experts_attn,
                dropout=config.dropout_expert_attn,
                dropout_expert=config.dropout_expert_attn,
                k_attn=config.k_attn,
                n_expert_shared_attn=config.n_expert_shared_attn,
                nope_pos=config.nope_pos,
                use_bias=config.use_moe_bias,
                manual_scale=config.manual_scale,
            ),
            ffn_module=SigmaMoE(
                config.d_model,
                config.n_experts_ffn,
                config.d_expert_ffn,
                k_ffn=config.k_ffn,
                dropout_expert=config.dropout_expert_ffn,
                n_expert_shared_ffn=config.n_expert_shared_ffn,
                use_bias=config.use_moe_bias,
            ),
            input_projection=(
                torch.nn.Linear(2 * config.d_model, config.d_model, bias=False)
                if input_reinjection
                else None
            ),
        )
        self.input_reinjection = input_reinjection
        if config.enable_early_exit:
            self.saturation_detector = SaturationGate(config.d_model)
        else:
            self.saturation_detector = None

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        e: None | Float[Tensor, "batch seq d_model"],
        layer_index: int,
        reinjection_embeddings: None | Float[Tensor, "batch seq d_model"],
        attention_mask: None | Bool[Tensor, "batch 1 seq seq"],
        sequence_length: None | Int[Tensor, "batch seq"],
        continue_mask: None | Int[Tensor, " size"] = None,
    ) -> tuple[
        Float[Tensor, "batch seq d_model"],
        tuple,
        Float[Tensor, "batch seq"] | None,
    ]:
        """Forward pass through the layer with configurable behavior."""
        if self.input_reinjection and reinjection_embeddings is not None:
            assert self.input_projection is not None
            x = torch.cat((x, reinjection_embeddings), dim=-1)
            x = self.input_projection(x)

        q_val, k_val, v_val = self._apply_pre_norm_attn(x)

        att_out, expert_sel_attn = self.attention(
            q_val, k_val, v_val, attention_mask, sequence_length
        )

        x = self._apply_update_to_residual(
            x,
            att_out,
            continue_mask,
            layer_index,
            self.attn_post,
            e,
        )

        ffn_out, expert_sel_ffn = self.ffn(*self._apply_pre_norm_ffn(x))

        saturation_event = None
        if self.saturation_detector is not None:
            saturation_event = self.saturation_detector(ffn_out)

        x = self._apply_update_to_residual(
            x,
            ffn_out,
            continue_mask,
            layer_index,
            self.ffn_post,
            e,
        )

        return (x, (expert_sel_attn, expert_sel_ffn), saturation_event)
