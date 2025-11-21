from abc import ABC, abstractmethod

import torch
from jaxtyping import Float, Int
from llmfoundry.models.layers.layer_builders import build_norm
from torch import Tensor
from torch.nn import Module

from dyna.config import DynaConfig, NormStructure, RescaleMethod

from .attention_module import AttentionModule
from .dyna_module import DynaModule


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
        
        PRE_NORM_STRUCTURES = [
            NormStructure.peri,
            NormStructure.pre,
            NormStructure.moeut,
            NormStructure.sandwich,
        ]
        POST_NORM_STRUCTURES = [
            NormStructure.peri,
            NormStructure.post,
            NormStructure.sandwich,
        ]
        
        self.attn_pre = build_norm(
            name=config.norms.norm_type,
            eps=config.norms.attn_eps,
            normalized_shape=config.d_model,
        )
        self.attn_post = build_norm(
            name=config.norms.norm_type,
            eps=config.norms.attn_eps,
            normalized_shape=config.d_model,
        )
        
        enable_attn_pre = config.norm_structure in PRE_NORM_STRUCTURES
        enable_attn_post = config.norm_structure in POST_NORM_STRUCTURES
        self.attn_pre.requires_grad_(enable_attn_pre)
        self.attn_post.requires_grad_(enable_attn_post)
        
        self.ffn_pre = build_norm(
            name=config.norms.norm_type,
            eps=config.norms.ffn_eps,
            normalized_shape=config.d_model,
        )
        self.ffn_post = build_norm(
            name=config.norms.norm_type,
            eps=config.norms.ffn_eps,
            normalized_shape=config.d_model,
        )
        
        enable_ffn_pre = config.norm_structure in PRE_NORM_STRUCTURES
        enable_ffn_post = config.norm_structure in POST_NORM_STRUCTURES
        self.ffn_pre.requires_grad_(enable_ffn_pre)
        self.ffn_post.requires_grad_(enable_ffn_post)
        # Configuration
        self.drop = torch.nn.Dropout(config.dropout)
        self.n_layers = config.n_layers
        self.enable_early_exit = config.enable_early_exit
        self.rescaling_method = config.rescaling_method
        self.norm_structure = config.norm_structure

        # Use the the inint block length for this value
        self.base_depth = config.base_depth
        self.current_depth = config.current_depth
        self.base_width = config.base_width
        self.current_width = config.current_width
        self.loop_hyper_params = config.loop_hyper_params
        self.cp_alpha = config.cp_alpha

    def update_inv_freq(self, base: int):
        self.attention.update_inv_freq(base)

    def _apply_pre_norm_attn(
        self,
        residual_stream: Float[Tensor, "batch seq d_model"],
    ) -> tuple[
        Float[Tensor, "batch max_len d_model"],
        Float[Tensor, "batch seq d_model"],
        Float[Tensor, "batch seq d_model"],
    ]:
        uses_pre_norm = self.norm_structure in [
            NormStructure.peri,
            NormStructure.pre,
            NormStructure.sandwich,
        ]
        
        if uses_pre_norm:
            normalized = self.attn_pre(residual_stream)
            query_input = normalized
            key_input = normalized
            value_input = normalized
        elif self.norm_structure == NormStructure.post:
            query_input = residual_stream
            key_input = residual_stream
            value_input = residual_stream
        elif self.norm_structure == NormStructure.moeut:
            normalized = self.attn_pre(residual_stream)
            query_input = normalized
            key_input = normalized
            value_input = residual_stream
        else:
            raise ValueError(f"{self.norm_structure} must be one of {NormStructure}")

        return query_input, key_input, value_input

    def _apply_update_to_residual(
        self,
        residual_stream: Float[Tensor, "batch seq d_model"],
        update_on_stream: Float[Tensor, "batch seq d_model"],
        continue_mask: None | Int[Tensor, " size"],
        layer_index: int,
        norm_to_use: Module,
        e: Float[Tensor, "batch seq d_model"] | None = None,
        cum_sum: Float[Tensor, "batch seq"] | None = None,
    ) -> tuple[Float[Tensor, "batch seq d_model"], int]:
        update = update_on_stream
        if self.norm_structure == NormStructure.peri:
            update = norm_to_use(update_on_stream)
        update = self.drop(update)
        layer_index = layer_index + 1
        match self.rescaling_method:
            case RescaleMethod.none:
                if self.enable_early_exit and continue_mask is not None:
                    residual_stream = torch.scatter_add(
                        residual_stream.view(-1),
                        0,
                        continue_mask,
                        update.view(-1)[continue_mask],
                    ).reshape_as(residual_stream)
                else:
                    residual_stream = residual_stream + update
            case RescaleMethod.cum_avg_prot_emb | RescaleMethod.cum_avg_no_prot_emb:
                if (
                    self.enable_early_exit
                    and cum_sum is not None
                    and continue_mask is not None
                ):
                    scale_factor = (layer_index - 1) / layer_index
                    update_factor: Tensor = (  # pyright: ignore[reportAssignmentType,reportRedeclaration]
                        cum_sum[continue_mask].unsqueeze(1) / layer_index
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
            case RescaleMethod.complete_p:
                # Reference: nanoGPT-mup Block.forward() implementation
                # self.residual_scaling = 1/(config.depth_multiplier ** config.depth_alpha_exp)
                scale_factor = (self.base_depth / self.current_depth) ** self.cp_alpha
                if self.enable_early_exit and continue_mask is not None:
                    residual_stream = torch.scatter_add(
                        residual_stream.view(-1),
                        0,
                        continue_mask,
                        scale_factor * update.view(-1)[continue_mask],
                    ).reshape_as(residual_stream)
                else:
                    residual_stream = residual_stream + scale_factor * update

            case RescaleMethod.complete_p_dyn:
                if (
                    self.enable_early_exit
                    and cum_sum is not None
                    and continue_mask is not None
                ):
                    scale_factor = layer_index / (layer_index - 1) ** (-self.cp_alpha)
                    update_factor: float = cum_sum[continue_mask].unsqueeze(1) * (  # pyright: ignore[reportRedeclaration]
                        layer_index / self.base_depth
                    ) ** (-self.cp_alpha)

                    residual_stream[continue_mask] = (
                        scale_factor * residual_stream[continue_mask]
                        + update.view(-1, update.shape[-1])[: continue_mask.sum()]
                        * update_factor
                    )
                else:
                    # Apply to all tokens when early exit is disabled
                    scale_factor = (layer_index - 1) / layer_index
                    update_factor: float = (layer_index / self.base_depth) ** (
                        -self.cp_alpha
                    )
                    # print(f"Scale factor: {scale_factor}, Update factor: {update_factor}", flush=True)
                    residual_stream = (
                        scale_factor * residual_stream + update * update_factor
                    )
            # case (
            #     RescaleMethod.sqrt_prot_emb |
            # RescaleMethod.sqrt_no_prot_emb
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
            #         scale_factor = (math.sqrt(layer_index) - 1) /
            #  math.sqrt(layer_index)
            #         residual_stream = scale_factor * residual_stream + update / (
            #             math.sqrt(layer_index)
            #         )
            #     if e is not None:
            #         residual_stream = residual_stream + e
            # case RescaleMethod.sqrt_scale_prot_emb:
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
            # case RescaleMethod.avg_prot_emb:
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
        if (
            self.norm_structure == NormStructure.post
            or self.norm_structure == NormStructure.sandwich
        ):
            residual_stream = norm_to_use(residual_stream)

        return residual_stream, layer_index

    def _apply_pre_norm_ffn(self, residual_stream: Float[Tensor, "batch seq d_model"]):
        if (
            self.norm_structure == NormStructure.peri
            or self.norm_structure == NormStructure.pre
            or self.norm_structure == NormStructure.sandwich
        ):
            # Peri, Pre
            residual_stream_normed = self.ffn_pre(residual_stream)
            ffn_val_1 = residual_stream_normed
            ffn_val_2 = residual_stream_normed
        elif self.norm_structure == NormStructure.post:
            ffn_val_1 = residual_stream
            ffn_val_2 = residual_stream
        elif self.norm_structure == NormStructure.moeut:
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
        e: None | Float[Tensor, "batch seq d_model"],
        reinjection_embeddings: None | Float[Tensor, "batch seq d_model"],
        attention_mask: None | Float[Tensor, "batch seq seq"],
        sequence_length: None | Int[Tensor, "batch seq"],
        layer_index: int,
        continue_mask: None | Int[Tensor, " size"] = None,
    ) -> tuple[
        Float[Tensor, "batch seq d_model"],
        tuple,
        Float[Tensor, "batch seq"] | None,
    ]:
        """Forward pass through the layer with configurable behavior."""
        pass
