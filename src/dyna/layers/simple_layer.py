from __future__ import annotations

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from dyna.attention import BasicAttn
from dyna.config import DynaConfig
from dyna.modules import LayerModule, SaturationGate
from dyna.transition import BasicFFN


class SimpleLayer(LayerModule):
    def __init__(self, config: DynaConfig, input_reinjection: bool = False):
        """Initialize SimpleLayer with configurable parameters."""
        super().__init__(
            config,
            BasicAttn(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_head=config.d_head,
                dropout=config.dropout,
                nope_pos=config.nope_pos,
            ),
            BasicFFN(
                config.d_model,
                config.d_ffn,
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
        attention_mask: Bool[Tensor, "batch 1 seq seq"],
        sequence_length: Int[Tensor, "batch seq"],
        continue_mask: None | Int[Tensor, " size"] = None,
        # ifdef PYTEST
        collector: dict | None = None,
        # endif
    ) -> tuple[
        Float[Tensor, "batch seq d_model"],
        tuple,
        Float[Tensor, "batch seq"] | None,
        int,
    ]:
        """Forward pass through the layer with configurable behavior."""
        if self.input_reinjection and reinjection_embeddings is not None:
            assert self.input_projection is not None, "Input projection must be defined"
            # Concatenate along the feature dimension
            x = torch.cat((x, reinjection_embeddings), dim=-1)
            # Project back to original d_model dimension
            x = self.input_projection(x)

        q_val, k_val, v_val = self._apply_pre_norm_attn(x)

        att_out, expert_sel_attn = self.attention(
            q_val,
            k_val,
            v_val,
            attention_mask,
            sequence_length,
            # ifdef PYTEST
            collector=collector,
            # endif
        )

        x, layer_index = self._apply_update_to_residual(
            x,
            att_out,
            continue_mask,
            layer_index,
            self.attn_post,
            e,
        )

        ffn_inputs = self._apply_pre_norm_ffn(x)
        ffn_out, expert_sel_ffn = self.ffn(
            *ffn_inputs,
            # ifdef PYTEST
            collector=collector,
            # endif
        )

        saturation_event = None
        if self.saturation_detector is not None:
            saturation_event = self.saturation_detector(ffn_out)
        x, layer_index = self._apply_update_to_residual(
            x, ffn_out, continue_mask, layer_index, self.ffn_post, e
        )

        return (x, (expert_sel_attn, expert_sel_ffn), saturation_event, layer_index)
