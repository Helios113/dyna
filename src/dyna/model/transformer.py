import math
import random
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


def calc_entropy(
    # chunks: tuple[Float[Tensor, "batch chunk vocab"], ...], temperature: float
    chunks: Float[Tensor, "batch seq_len vocab"],
    temperature: float,
) -> Float[Tensor, "batch seq_len"]:
    """Calculate entropy of logits.

    Args:
        chunks (Float[Tensor, "batch seq vocab"]): Logits tensor.
        temperature (float): Temperature for softmax.

    Returns:
        Float[Tensor, "batch seq"]: Entropy tensor.
    """
    # entropy = 0
    # for logits in chunks:
    logits = chunks
    probs = torch.softmax(logits / temperature, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    entropy = -torch.sum(probs * log_probs, dim=-1).detach()
    return entropy


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
        self.sample_iterations = config.sample_iterations

        # Looping behaviour
        self.n_layers = config.n_layers
        self.n_repeats = config.n_repeats
        self.repeat_residual = config.repeat_residual
        self.min_loop_layers = self.n_repeats
        self.total_depth_for_init = config.total_depth_for_init
        self.loop_normalization = config.loop_normalization
        if self.loop_normalization:
            self.loop_norm = torch.nn.LayerNorm(config.d_model)
        # Execution behaviour
        self.enable_early_exit = config.enable_early_exit
        self.execution_mode = config.execution_mode
        self.gather_stats = False
        self.loop_rebase = config.loop_rebase
        self.rope_base = 10000
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
            scale = math.sqrt(2 / (self.n_repeats * self.total_depth_for_init))
        else:
            scale = math.sqrt(2 / self.total_depth_for_init)

        # Initialize tracking variables
        self._seq_len = []
        self._latent_vectors = []
        self._residual_magnitudes = []
        self._exit_logits = []
        self._expert_sel = []

        # Initialize layer parameters
        for layer in self.modules():
            if isinstance(layer, DynaFormer):
                continue
            elif isinstance(layer, DynaModule):
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

    def _clear_selection_history(self):
        for layer in self.modules():
            if hasattr(layer, "clear_selection_history"):
                assert isinstance(layer, DynaModule)
                layer.clear_selection_history()

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
            x,
            attention_mask,
            sequence_length,
            self._get_repeat_number(),
            e,
            reinjection_embeddings,
            layer_index,
        )
        x = self.tail(x, attention_mask, sequence_length, e, layer_index)

        return x, energy_per_sample

    def _get_repeat_number(self) -> int:
        if not self.sample_iterations:
            return self.n_repeats
        # uniform sampling
        a = random.randint(1, self.n_repeats)
        print(f"We have sampled {a} iterations")
        return a
        # Gaussian sampling
        # TODO: Implement lambda sampling
        # Lambda sampling
        # TODO: Implement lambda sampling
        return 0

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
        layer_index = 1
        for layer in self.head_layers:
            x, expert_sel, _, layer_index = layer(
                x=x,
                e=e,
                layer_index=layer_index,
                reinjection_embeddings=None,
                attention_mask=attention_mask,
                sequence_length=sequence_length,
                continue_mask=None,
            )

            if self.gather_stats:
                self.gather_stats_func(x, expert_sel)

        if self.execution_mode in GEIPING_METHODS:
            reinjection_embeddings = x.clone()
            x = torch.rand_like(x)

        return x, reinjection_embeddings, layer_index

    def body(
        self,
        x: Float[Tensor, "batch seq d_model"],
        attention_mask: Bool[Tensor, "batch 1 seq seq"],
        sequence_length: Int[Tensor, "batch seq"],
        repeats: int,
        e: Float[Tensor, "batch seq d_model"] | None = None,
        reinjection_embeddings: Float[Tensor, "batch seq d_model"] | None = None,
        layer_index: int | None = None,
    ) -> tuple[
        Float[Tensor, "batch seq d_model"], Float[Tensor, "batch seq 1"] | None, int
    ]:
        if layer_index is None:
            layer_index = 1
        layer_index = 1

        residual_embeddings = None
        continue_mask = None
        energy_per_sample = None
        for i in range(repeats):
            if residual_embeddings is not None:
                x = x + residual_embeddings
            for layer in self.body_layers:
                print("Layer in loop:", layer_index, flush=True)
                x_out, expert_sel, saturation_event, layer_index = layer(
                    x=x,
                    e=e,
                    layer_index=layer_index,
                    reinjection_embeddings=reinjection_embeddings,
                    attention_mask=attention_mask,
                    sequence_length=sequence_length,
                    continue_mask=continue_mask,
                )
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
                    self.gather_stats_func(x, expert_sel)
            if self.loop_normalization:
                x = self.loop_norm(x)
            if self.loop_rebase:
                print("Updating rope base to:", self.rope_base * (i + 1), flush=True)
                self.update_inv_freq(self.rope_base * (i + 1))
            if self.repeat_residual:
                residual_embeddings = x.clone()
            if not continue_processing:
                break
        return x, energy_per_sample, layer_index

    def update_inv_freq(self, base: int):
        for layer in self.body_layers:
            layer.update_inv_freq(base)

    def _apply_early_exit(
        self,
        x_out: Float[Tensor, "batch seq d_model"],
        saturation_event: Float[Tensor, "batch seq"] | None,
        old_continue_mask: Int[Tensor, " batchseq"] | None,
        energy_per_sample: Float[Tensor, "batch seq"] | None,
    ) -> tuple[
        Float[Tensor, "batch seq d_model"],
        Int[Tensor, " batchseq"] | None,
        bool,
        Int[Tensor, "batch seq"] | None,
    ]:
        if not self.enable_early_exit or saturation_event is None:
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
                # TODO PRINT ENERGY PER SAMPLE to check if it is correct
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
            x, expert_sel, _, layer_index = layer(
                x=x,
                e=e,
                layer_index=layer_index,
                reinjection_embeddings=None,
                attention_mask=attention_mask,
                sequence_length=sequence_length,
                continue_mask=None,
            )
            if self.gather_stats:
                self.gather_stats_func(x, expert_sel)

        return x

    def gather_stats_func(
        self,
        x: Float[Tensor, "batch seq d_model"],
        expert_sel: tuple[
            tuple[
                Int[Tensor, "batch seq expert_heads attn_experts"] | None,
                Int[Tensor, "batch seq expert_heads attn_experts"] | None,
            ],
            Int[Tensor, "batch seq ffn_experts"] | None,
        ],
    ) -> None:
        assert isinstance(self._temp_lm_head, Callable)
        self._latent_vectors[-1].append(
            calc_entropy(
                # torch.chunk(self._temp_lm_head(x.detach()), chunks=4, dim=1), 1.0
                self._temp_lm_head(x.detach()),
                1.0,
                #
            )
        )
        if expert_sel[0] is not None:
            self._expert_sel[-1].append(expert_sel)
        self._residual_magnitudes[-1].append(torch.norm(x.detach(), dim=-1).cpu())
