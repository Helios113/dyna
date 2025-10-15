import math
from collections.abc import Callable

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import ModuleList

from dyna.config import (
    CROSS_ENTROPY_IGNORE_INDEX,
    GEIPING_METHODS,
)
from dyna.config.enums import ExecutionMode
from dyna.layers import MoEUTLayer, SimpleLayer
from dyna.model.base import DynaConfig, DynaPretrainedModel
from dyna.modules import AttentionModule, DynaModule


class DynaFormer(DynaPretrainedModel):
    """Transformer model with configurable behavior."""

    def __init__(self, config: DynaConfig):
        """Initialize DynaFormer model.

        Args:
            config (DynaConfig): Configuration object for the model.
        """
        super().__init__(config)
        # entropy calculation head
        self._temp_lm_head: Callable[[torch.Tensor], torch.Tensor] | None = None

        # Moeut regularization entropy
        self.reg_entropy = config.reg_entropy
        self.reg_entropy_attn = config.reg_entropy_attn
        self.use_reg_loss = config.use_reg_loss
        self.use_energy_per_sample = config.use_energy_per_sample
        # Size of head and tail
        self.head_size = config.head_size
        self.tail_size = config.tail_size

        # Looping behaviour
        self.n_layers = config.n_layers
        self.n_repeats = config.n_repeats
        self.repeat_residual = config.repeat_residual
        self.min_loop_layers = self.n_repeats
        self.repeats = config.n_repeats
        self.total_depth_for_init = config.total_depth_for_init

        # Execution behaviour
        self.enable_early_exit = config.enable_early_exit
        self.execution_mode = config.execution_mode
        self.gather_stats = False

        # Layer configuration
        self.body_layers: ModuleList
        self.head_layers: ModuleList | None = None
        self.tail_layers: ModuleList | None = None

        self._construct_layers(config)

    def _construct_layers(self, config):
        """Constructs the layers of the transformer model.

        We have three types of layers: body, head, and tail.
        body is always present
        head and tail are optional based on the execution mode.
        body layers can have input re-injection,
        which is the boolen parameter in the init.
        """
        match config.execution_mode:
            # MOE
            case ExecutionMode.moe:
                self.body_layers = ModuleList(
                    [MoEUTLayer(config) for _ in range(config.n_layers)]
                )
            # Transformer
            case ExecutionMode.transformer:
                self.body_layers = ModuleList(
                    [SimpleLayer(config) for _ in range(config.n_layers)]
                )
            # Geiping
            case ExecutionMode.geiping_std:
                assert self.head_size > 0
                assert self.tail_size > 0
                self.head_layers = ModuleList(
                    [SimpleLayer(config) for _ in range(self.head_size)]
                )
                self.body_layers = ModuleList(
                    [
                        SimpleLayer(config, input_reinjection=True)
                        for _ in range(config.n_layers)
                    ]
                )
                self.tail_layers = ModuleList(
                    [SimpleLayer(config) for _ in range(self.tail_size)]
                )
            # Arbit
            case ExecutionMode.arbit:
                assert self.head_size > 0
                assert self.tail_size > 0
                self.head_layers = ModuleList(
                    [SimpleLayer(config) for _ in range(self.head_size)]
                )
                self.body_layers = ModuleList(
                    [SimpleLayer(config, True) for _ in range(config.n_layers)]
                )
                self.tail_layers = ModuleList(
                    [SimpleLayer(config) for _ in range(self.tail_size)]
                )
            # Geiping MOE
            case ExecutionMode.geiping_moe:
                assert self.head_size > 0
                assert self.tail_size > 0
                self.head_layers = ModuleList(
                    [MoEUTLayer(config) for _ in range(self.head_size)]
                )
                self.body_layers = ModuleList(
                    [
                        MoEUTLayer(config, input_reinjection=True)
                        for _ in range(config.n_layers)
                    ]
                )
                self.tail_layers = ModuleList(
                    [MoEUTLayer(config) for _ in range(self.tail_size)]
                )

            case _:
                raise ValueError(
                    f"{config.execution_mode} needs to be one of {ExecutionMode}"
                )

    @torch.no_grad
    def reset_parameters(self) -> None:
        """Initialize all model parameters."""
        if self.enable_early_exit:
            scale = math.sqrt(2 / (self.n_repeats * len(self.total_depth_for_init)))
        else:
            scale = math.sqrt(2 / len(self.total_depth_for_init))

        # Initialize tracking variables
        self._seq_len = []
        self._latent_vectors = []
        self._residual_magnitudes = []
        self._exit_logits = []
        self._expert_sel = []

        # Initialize layer parameters
        for layer in self.modules():
            if isinstance(layer, DynaModule):
                layer.reset_parameters(scale)
            elif hasattr(layer, "reset_parameters"):
                assert isinstance(layer.reset_parameters, Callable)
                layer.reset_parameters()
            elif isinstance(layer, torch.nn.LayerNorm):
                torch.nn.init.ones_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def _collect_regularization_loss(self) -> torch.Tensor:
        if not self.use_reg_loss:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            return torch.tensor(0.0, device=device, dtype=dtype)
        reg_loss = torch.zeros(
            1,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
        )
        for layer in self.modules():
            if isinstance(layer, AttentionModule) and hasattr(layer, "get_reg_loss"):
                reg_loss = reg_loss + self.reg_entropy_attn * layer.get_reg_loss()
            elif isinstance(layer, DynaModule) and hasattr(layer, "get_reg_loss"):
                reg_loss = reg_loss + self.reg_entropy * layer.get_reg_loss()
        return reg_loss

    def mask_to_scatter_index(self, mask: torch.Tensor):
        idexes = torch.nonzero(mask.view(-1), as_tuple=True)[0]
        return idexes

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        attention_mask: Bool[Tensor, "batch 1 seq seq"],
        sequence_length: Int[Tensor, "batch seq"],
        e: Float[Tensor, "batch seq d_model"] | None = None,
        input_ids: Int[Tensor, "batch seq"] | None = None,
    ) -> tuple[Float[Tensor, "batch seq d_model"], Float[Tensor, "batch seq 1"] | None]:
        if input_ids is not None:
            _labels = torch.roll(input_ids, shifts=-1)
            _labels[:, -1] = CROSS_ENTROPY_IGNORE_INDEX

        # logging data containers
        self._expert_sel.append([])
        self._exit_logits.append([])
        self._latent_vectors.append([])
        self._seq_len.append([])
        self._residual_magnitudes.append([])

        x, reinjection_embeddings, layer_index = self.head(
            x, attention_mask, sequence_length, e
        )
        x, energy_per_sample, layer_index = self.body(
            x, attention_mask, sequence_length, e, reinjection_embeddings, layer_index
        )
        x = self.tail(x, attention_mask, sequence_length, e, layer_index)

        return x, energy_per_sample

    def head(
        self,
        x: Float[Tensor, "batch seq d_model"],
        attention_mask: Bool[Tensor, "batch 1 seq seq"],
        sequence_length: Int[Tensor, "batch seq"],
        e: Float[Tensor, "batch seq d_model"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch seq d_model"],
        None | Float[Tensor, "batch seq d_model"],
        int,
    ]:
        if self.head_layers is None:
            return x, None, 0
        reinjection_embeddings = None
        layer_index = 0
        for layer in self.head_layers:
            x, expert_sel, _ = layer(
                x=x,
                e=e,
                layer_index=layer_index,
                reinjection_embeddings=None,
                attention_mask=attention_mask,
                sequence_length=sequence_length,
                continue_mask=None,
            )
            layer_index += 1

            if self.gather_stats:
                assert isinstance(self._temp_lm_head, torch.nn.Module)
                self._latent_vectors[-1].append(
                    self._temp_lm_head(x[:, 10, :].detach()).detach().cpu()
                )
                if expert_sel[0] is not None:
                    self._expert_sel[-1].append(expert_sel)
                self._residual_magnitudes[-1].append(
                    torch.norm(x, dim=-1).detach().clone().cpu()
                )

        if self.execution_mode in GEIPING_METHODS:
            reinjection_embeddings = x.clone()
            x = torch.rand_like(x)

        return x, reinjection_embeddings, layer_index

    def body(
        self,
        x: Float[Tensor, "batch seq d_model"],
        attention_mask: Bool[Tensor, "batch 1 seq seq"],
        sequence_length: Int[Tensor, "batch seq"],
        e: Float[Tensor, "batch seq d_model"] | None = None,
        reinjection_embeddings: Float[Tensor, "batch seq d_model"] | None = None,
        layer_index: int | None = None,
    ) -> tuple[
        Float[Tensor, "batch seq d_model"], Float[Tensor, "batch seq 1"] | None, int
    ]:
        if layer_index is None:
            layer_index = 0
        residual_embeddings = None
        continue_mask = None
        energy_per_sample = None
        for _ in range(self.n_repeats):
            if residual_embeddings is not None:
                x = x + residual_embeddings
            for layer in self.body_layers:
                x_out, expert_sel, saturation_event = layer(
                    x=x,
                    e=e,
                    layer_index=layer_index,
                    reinjection_embeddings=reinjection_embeddings,
                    attention_mask=attention_mask,
                    sequence_length=sequence_length,
                    continue_mask=continue_mask,
                )
                layer_index += 1
                x, continue_mask, continue_processing, energy_per_sample = (
                    self._apply_early_exit(
                        x_out,
                        saturation_event,
                        old_continue_mask=continue_mask,
                        energy_per_sample=energy_per_sample,
                    )
                )

                if not continue_processing:
                    break
                if self.gather_stats:
                    assert isinstance(self._temp_lm_head, torch.nn.Module)
                    self._latent_vectors[-1].append(
                        self._temp_lm_head(x[:, 10, :].detach()).detach().cpu()
                    )
                    if expert_sel[0] is not None:
                        self._expert_sel[-1].append(expert_sel)
                    self._residual_magnitudes[-1].append(
                        torch.norm(x, dim=-1).detach().clone().cpu()
                    )

            if self.repeat_residual:
                residual_embeddings = x.clone()
            if not continue_processing:
                break
        return x, energy_per_sample, layer_index

    def _apply_early_exit(
        self,
        x_out: Float[Tensor, "batch seq d_model"],
        saturation_event: Float[Tensor, "batch seq"],
        old_continue_mask: Float[Tensor, "batch seq"] | None,
        energy_per_sample: Float[Tensor, "batch seq"] | None,
    ) -> tuple[
        Float[Tensor, "batch seq d_model"],
        Float[Tensor, "batch seq"] | None,
        bool,
        Float[Tensor, "batch seq"] | None,
    ]:
        if not self.enable_early_exit:
            return x_out, None, True, energy_per_sample
        continue_processing = True
        x = x_out
        if old_continue_mask is not None:
            if self.use_energy_per_sample:
                assert energy_per_sample is not None
                energy_per_sample = torch.scatter_add(
                    energy_per_sample.view(-1),
                    0,
                    old_continue_mask,
                    saturation_event.view(-1)[old_continue_mask],
                ).reshape(energy_per_sample.shape)
            saturation_event_tmp = torch.scatter(
                torch.zeros_like(saturation_event).view(-1),
                0,
                old_continue_mask,
                saturation_event.view(-1)[old_continue_mask],
            ).reshape(saturation_event.shape)
            continue_mask = self.mask_to_scatter_index(saturation_event_tmp)
            x = torch.scatter_reduce(
                x.view(-1),
                0,
                continue_mask,
                saturation_event.view(-1)[continue_mask],
                reduce="prod",
            ).reshape(x.shape)
        else:
            if self.use_energy_per_sample:
                energy_per_sample = saturation_event
            continue_mask = self.mask_to_scatter_index(saturation_event)

            x = torch.scatter_reduce(
                x.view(-1),
                0,
                continue_mask,
                saturation_event.view(-1)[continue_mask],
                reduce="prod",
            ).reshape(x.shape)

        if continue_mask is not None and continue_mask.numel() == 0:
            continue_processing = False

        return x, continue_mask, continue_processing, energy_per_sample

    def tail(
        self,
        x: Float[Tensor, "batch seq d_model"],
        attention_mask: Bool[Tensor, "batch 1 seq seq"],
        sequence_length: Int[Tensor, "batch seq"],
        e: Float[Tensor, "batch seq d_model"] | None = None,
        layer_index: int | None = None,
    ) -> Float[Tensor, "batch seq d_model"]:
        if layer_index is None:
            layer_index = 0
        if self.tail_layers is None:
            return x
        for layer in self.tail_layers:
            x, expert_sel, _ = layer(
                x=x,
                e=e,
                layer_index=layer_index,
                reinjection_embeddings=None,
                attention_mask=attention_mask,
                sequence_length=sequence_length,
                continue_mask=None,
            )
            layer_index += 1
            if self.gather_stats:
                assert isinstance(self._temp_lm_head, torch.nn.Module)
                self._latent_vectors[-1].append(
                    self._temp_lm_head(x[:, 10, :].detach()).detach().cpu()
                )
                if expert_sel[0] is not None:
                    self._expert_sel[-1].append(expert_sel)
                self._residual_magnitudes[-1].append(
                    torch.norm(x, dim=-1).detach().clone().cpu()
                )

        return x
