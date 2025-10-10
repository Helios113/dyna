from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from beartype import beartype

# from composer.callbacks
# Add jaxtyping imports
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.normalization import RMSNorm

from dyna.config import DynaConfig, NormStructure, RescaleMethod
from .dyna_module import DynaModule

from .attention_module import AttentionModule


@beartype
class LayerModule(Module, ABC):
    def __init__(
        self,
        config: DynaConfig,
        attention_module: AttentionModule,
        ffn_module: DynaModule,
        input_projection: Module | None = None,
    ):
        """Initialize LayerModule with configurable components."""
        super().__init__()
        self.attention = attention_module
        self.ffn = ffn_module
        self.input_projection = input_projection
        # Layer normalization - configurable type
        # norm_class = RMSNorm if config.use_rms_norm else torch.nn.LayerNorm

        self.attn_pre = RMSNorm(config.d_model)
        self.attn_post = RMSNorm(config.d_model)
        self.attn_post.requires_grad_(
            config.norm_structure in [NormStructure.peri, NormStructure.post]
        )
        self.ffn_pre = RMSNorm(config.d_model)
        self.ffn_post = RMSNorm(config.d_model)
        self.ffn_post.requires_grad_(
            config.norm_structure in [NormStructure.peri, NormStructure.post]
        )

        # Configuration
        self.drop = torch.nn.Dropout(config.dropout)
        self.n_layers = config.n_layers
        self.enable_early_exit = config.enable_early_exit
        self.rescaling_method = config.rescaling_method
        self.norm_structure = config.norm_structure

    # Done
    def _apply_pre_norm_attn(
        self,
        residual_stream: Float[Tensor, "batch seq d_model"],
    ) -> tuple[
        Float[Tensor, "batch max_len d_model"],
        Float[Tensor, "batch seq d_model"],
        Float[Tensor, "batch seq d_model"],
    ]:
        if (
            self.norm_structure.value == NormStructure.peri.value
            or self.norm_structure == NormStructure.pre.value
        ):
            # Peri, Pre
            residual_stream_normed = self.attn_pre(residual_stream)
            q_val = residual_stream_normed
            k_val = residual_stream_normed
            v_val = residual_stream_normed

        elif self.norm_structure.value == NormStructure.post.value:
            q_val = residual_stream
            k_val = residual_stream
            v_val = residual_stream

        elif self.norm_structure.value == NormStructure.moeut.value:
            residual_stream_normed = self.attn_pre(residual_stream)
            q_val = residual_stream_normed
            k_val = residual_stream_normed
            v_val = residual_stream
        else:
            raise ValueError(f"{self.norm_structure} must be one of {NormStructure}")

        return q_val, k_val, v_val

    def _apply_update_to_residual(
        self,
        residual_stream: Float[Tensor, "batch seq d_model"],
        update_on_stream: Float[Tensor, "batch seq d_model"],
        continue_mask: None | Int[Tensor, "size"],
        layer_index: int,
        norm_to_use: Module,
        e: Float[Tensor, "batch seq d_model"] | None = None,
        cum_sum: Float[Tensor, "batch seq"] | None = None,
    ) -> Float[Tensor, "batch seq d_model"]:
        update = update_on_stream
        if self.norm_structure.value == NormStructure.peri.value:
            update = norm_to_use(update_on_stream)
        update = self.drop(update)

        match self.rescaling_method.value:
            case RescaleMethod.none.value:
                if self.enable_early_exit and continue_mask is not None:
                    residual_stream = torch.scatter_add(
                        residual_stream.view(-1),
                        0,
                        continue_mask,
                        update.view(-1)[continue_mask],
                    ).reshape_as(residual_stream)
                else:
                    residual_stream = residual_stream + update
            case (
                RescaleMethod.cum_avg_prot_emb.value
                | RescaleMethod.cum_avg_no_prot_emb.value
            ):
                if e is not None:
                    residual_stream = residual_stream - e
                if self.enable_early_exit and cum_sum is not None and tau is not None:
                    scale_factor = (layer_index - 1) / layer_index
                    update_factor = (
                        cum_sum[continue_mask].unsqueeze(1) * tau / layer_index
                    )

                    residual_stream[continue_mask] = (
                        scale_factor * residual_stream[continue_mask]
                        + update.view(-1, update.shape[-1])[: continue_mask.sum()]
                        * update_factor
                    )
                else:
                    # Apply to all tokens when early exit is disabled
                    scale_factor = (layer_index - 1) / layer_index
                    residual_stream = (
                        scale_factor * residual_stream + update / layer_index
                    )
                if e is not None:
                    residual_stream = residual_stream + e
            # case (
            #     RescaleMethod.sqrt_prot_emb.value | RescaleMethod.sqrt_no_prot_emb.value
            # ):
            #     if e is not None:
            #         residual_stream = residual_stream - e
            #     if self.enable_early_exit:
            #         scale_factor = math.sqrt(layer_index) - 1 / math.sqrt(layer_index)
            #         update_factor = (
            #             cum_sum[continue_mask].unsqueeze(1)
            #             * tau
            #             / math.sqrt(layer_index)
            #         )

            #         residual_stream[continue_mask] = (
            #             scale_factor * residual_stream[continue_mask]
            #             + update.view(-1, update.shape[-1])[: continue_mask.sum()]
            #             * update_factor
            #         )
            #     else:
            #         # Apply to all tokens when early exit is disabled
            #         scale_factor = (math.sqrt(layer_index) - 1) / math.sqrt(layer_index)
            #         residual_stream = scale_factor * residual_stream + update / (
            #             math.sqrt(layer_index)
            #         )
            #     if e is not None:
            #         residual_stream = residual_stream + e
            # case RescaleMethod.sqrt_scale_prot_emb.value:
            #     residual_stream = residual_stream - e
            #     if self.enable_early_exit:
            #         update_factor = cum_sum[continue_mask].unsqueeze(1) * tau
            #         residual_stream[continue_mask] = (
            #             residual_stream[continue_mask]
            #             + update.view(-1, update.shape[-1])[: continue_mask.sum()]
            #             * update_factor
            #         ) / math.sqrt(2)
            #     else:
            #         # Apply to all tokens when early exit is disabled
            #         if layer_index == 2:
            #             residual_stream = residual_stream + update
            #         else:
            #             residual_stream = (residual_stream + update) / (math.sqrt(2))

            #     residual_stream = residual_stream + e
            # case RescaleMethod.avg_prot_emb.value:
            #     residual_stream = residual_stream - e
            #     if self.enable_early_exit:
            #         update_factor = cum_sum[continue_mask].unsqueeze(1) * tau
            #         residual_stream[continue_mask] = (
            #             residual_stream[continue_mask] / 2
            #             + update.view(-1, update.shape[-1])[: continue_mask.sum()]
            #             * update_factor
            #         )
            #     else:
            #         residual_stream = residual_stream / 2 + update

            #     residual_stream = residual_stream + e
        if self.norm_structure.value == NormStructure.post.value:
            residual_stream = norm_to_use(residual_stream)

        return residual_stream

    def _apply_pre_norm_ffn(self, residual_stream: Float[Tensor, "batch seq d_model"]):
        if (
            self.norm_structure.value == NormStructure.peri.value
            or self.norm_structure.value == NormStructure.pre.value
        ):
            # Peri, Pre
            residual_stream_normed = self.ffn_pre(residual_stream)
            ffn_val_1 = residual_stream_normed
            ffn_val_2 = residual_stream_normed
        elif self.norm_structure.value == NormStructure.post.value:
            ffn_val_1 = residual_stream
            ffn_val_2 = residual_stream
        elif self.norm_structure.value == NormStructure.moeut.value:
            residual_stream_normed = self.ffn_pre(residual_stream)
            ffn_val_1 = residual_stream_normed
            ffn_val_2 = residual_stream
        else:
            raise ValueError(f"{self.norm_structure} must be one of {NormStructure}")

        return ffn_val_1, ffn_val_2

    @abstractmethod
    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        layer_index: int,
        e: Float[Tensor, "batch seq d_model"],
        mask: tuple[Bool[Tensor, "batch seq seq"], Int[Tensor, "batch seq"]],
    ) -> tuple[
        Float[Tensor, "batch seq d_model"],
        bool,
        int,
        None | Float[Tensor, "batch seq"],
        tuple,
    ]:
        """Forward pass through the layer with configurable behavior."""
        pass
