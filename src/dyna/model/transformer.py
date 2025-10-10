import math

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import ModuleList, Parameter
from torch.nn.modules.normalization import RMSNorm

from dyna.config import (
    CROSS_ENTROPY_IGNORE_INDEX,
    LATENT_RECURSION_METHODS,
    GEIPING_METHODS,
)
from dyna.config.enums import ExecutionMode

print("Importing 10", flush=True)
from dyna.layers import MoEUTLayer, SimpleLayer
print("Importing 11", flush=True)

from dyna.model.base import DynaPretrainedModel, DynaConfig
from dyna.modules import DynaModule, AttentionModule, LayerModule
from collections.abc import Iterator
from typing import Generic, TypeVar, overload

from torch import nn

T = TypeVar("T", bound=nn.Module)


class TypedModuleList(Generic[T], nn.ModuleList):
    def __iter__(self) -> Iterator[T]:
        return super().__iter__()  # type: ignore[no-any-return]

    def append(self, module: T) -> "TypedModuleList[T]":  # type: ignore[override]
        return super().append(module)  # type: ignore[return-value]

    @overload
    def __getitem__(self, idx: slice) -> "TypedModuleList[T]": ...

    @overload
    def __getitem__(self, idx: int) -> T: ...

    def __getitem__(self, idx):  # type: ignore[no-untyped-def]
        return super().__getitem__(idx)

    def __setitem__(self, idx: int, module: T) -> None:  # type: ignore[override]
        super().__setitem__(idx, module)

class DynaFormer(DynaPretrainedModel):  # equivalne to MPTModel
    """MoEUT transformer model with configurable behavior."""

    def __init__(self, config: DynaConfig):
        super().__init__(config)

        self.reg_entropy = config.reg_entropy
        self.perfiery_size = config.perfiery_size
        self.reg_entropy_attn = config.reg_entropy_attn
        self.n_layers = config.n_layers
        self.n_repeats = config.n_repeats
        self.d_model = config.d_model
        self.enable_early_exit = config.enable_early_exit
        self.collect_reg_loss = config.collect_reg_loss
        self.execution_mode = config.execution_mode
        self.router = Parameter(torch.zeros(self.d_model), requires_grad=False)
        self.tau = Parameter(torch.ones(1), requires_grad=self.enable_early_exit)
        self.gather_stats = False
        self.layers: ModuleList[LayerModule]
        match config.execution_mode.value:
            case ExecutionMode.moe.value:
                self.layers = ModuleList(
                    [MoEUTLayer(config) for _ in range(config.n_layers)]
                )
            case ExecutionMode.transformer.value:
                self.layers = ModuleList(
                    [SimpleLayer(config) for _ in range(config.n_layers)]
                )
            case ExecutionMode.geiping_std.value:  # Geiping et al getting rid of the head and tail lists
                self.head = ModuleList(
                    [SimpleLayer(config) for _ in range(self.perfiery_size)]
                )
                self.layers = ModuleList(
                    [SimpleLayer(config, True) for _ in range(config.n_layers)]
                )
                self.tail = ModuleList(
                    [SimpleLayer(config) for _ in range(self.perfiery_size)]
                )
            case ExecutionMode.arbit.value:
                self.head = ModuleList(
                    [SimpleLayer(config) for _ in range(self.perfiery_size)]
                )
                self.layers = ModuleList(
                    [SimpleLayer(config, True) for _ in range(config.n_layers)]
                )
                self.tail = ModuleList(
                    [SimpleLayer(config) for _ in range(self.perfiery_size)]
                )

            case ExecutionMode.geiping_moe.value:
                self.head = ModuleList([MoEUTLayer(config) for _ in range(2)])
                self.layers = ModuleList(
                    [MoEUTLayer(config, True) for _ in range(config.n_layers)]
                )
                self.tail = ModuleList([MoEUTLayer(config) for _ in range(2)])

            case _:
                raise ValueError(
                    f"{config.execution_mode} needs to be one of {ExecutionMode}"
                )
        self.repeat_residual = config.repeat_residual
        self.min_loop_layers = self.n_repeats
        self.repeats = config.n_repeats
        # # Initialize parameters
        # self.reset_parameters()

    @torch.no_grad
    def reset_parameters(self) -> None:
        """Initialize all model parameters."""
        if self.enable_early_exit:
            scale = math.sqrt(2 / (self.n_repeats * len(self.layers)))
            torch.nn.init.normal_(self.router, 0, scale / math.sqrt(self.d_model))
        else:
            scale = math.sqrt(2 / len(self.layers))

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
            elif isinstance(layer, (RMSNorm, torch.nn.LayerNorm)):
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
                else:
                    # For standard LayerNorm
                    torch.nn.init.ones_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)

    def _collect_regularization_loss(self) -> torch.Tensor:
        if not self.collect_reg_loss:
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
        e: Float[Tensor, "batch seq d_model"] | None,
        mask: tuple[Bool[Tensor, "batch seq seq"], Int[Tensor, "batch seq"]],
        input_ids: Int[Tensor, "batch seq"] | None = None,
    ) -> tuple[Float[Tensor, "batch seq d_model"], torch.Tensor]:
        if input_ids is not None:
            _labels = torch.roll(input_ids, shifts=-1)
            _labels[:, -1] = CROSS_ENTROPY_IGNORE_INDEX

        self._expert_sel.append([])
        self._exit_logits.append([])
        self._latent_vectors.append([])
        self._seq_len.append([])
        self._residual_magnitudes.append([])
        cum_sum = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        continue_mask = None
        continue_processing = True
        reinjection_embeddings = None
        residual_embeddings = None
        if self.execution_mode in LATENT_RECURSION_METHODS:
            for idx, layer in enumerate(self.head):
                x, expert_sel, saturation_event = layer(
                    x=x,
                    layer_index=idx + 2,
                    e=e,
                    reinjection_embeddings=None,
                    router=self.router,
                    cum_sum=cum_sum,
                    tau=self.tau,
                    mask=mask,
                    total_layers=self.n_layers * self.repeats,
                    continue_mask=continue_mask,
                )
                if self.gather_stats:
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
        energy_per_sample = torch.zeros(
            x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype
        )
        
        for li in range(self.n_repeats):
            if self.repeat_residual and li > 0:
                x = x + residual_embeddings
            for idx, layer in enumerate(self.layers):
                
                # Calculate correct layer index based on execution mode
                if self.execution_mode.value in LATENT_RECURSION_METHODS:
                    layer_index = 2 + (li * self.n_layers) + idx + 2
                else:
                    # Standard: current_position_in_repeated_layers + 2
                    layer_index = (li * self.n_layers) + idx + 2
                assert isinstance(layer, LayerModule)

                x_out, expert_sel, saturation_event = layer(
                    x=x,
                    layer_index=layer_index,
                    e=e,
                    reinjection_embeddings=reinjection_embeddings,
                    mask=mask,
                    continue_mask=continue_mask,
                )

                if self.enable_early_exit and idx == 1:
                    x = x_out
                    if continue_mask is not None:
                        energy_per_sample = torch.scatter_add(
                            energy_per_sample.view(-1),
                            0,
                            continue_mask,
                            saturation_event.view(-1)[continue_mask],
                        ).reshape(energy_per_sample.shape)
                        # saturation_event changes
                        # But I am only intested in the union of the current and the previous
                        saturation_event_tmp = torch.scatter(
                            torch.zeros_like(saturation_event).view(-1),
                            0,
                            continue_mask,
                            saturation_event.view(-1)[continue_mask],
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
                        energy_per_sample = saturation_event
                        continue_mask = self.mask_to_scatter_index(saturation_event)

                        x = torch.scatter_reduce(
                            x.view(-1),
                            0,
                            continue_mask,
                            saturation_event.view(-1)[continue_mask],
                            reduce="prod",
                        ).reshape(x.shape)
                    print(
                        f"continued tokens after layer {layer_index-2}",
                        continue_mask.numel(),
                        "/",
                        x.shape[0] * x.shape[1],
                        flush=True,
                    )

                    if continue_mask is not None and continue_mask.numel() == 0:
                        continue_processing = False
                else:
                    x = x_out
                # # make continue_processing just be conditioned on the last token
                # if self.repeats == li * (self.n_layers) + idx + 1:
                #     continue_processing = False
                if not continue_processing:
                    break
                if self.gather_stats:
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

        if self.execution_mode.value in LATENT_RECURSION_METHODS:
            for idx, layer in enumerate(self.tail):
                # Calculate tail layer index: head(2) + all_repeated_layers + current_tail_position + 1
                layer_index = 2 + (self.repeats * self.n_layers) + idx + 2
                x, expert_sel, saturation_event = layer(
                    x=x,
                    layer_index=layer_index,
                    e=e,
                    reinjection_embeddings=None,
                    router=self.router,
                    cum_sum=cum_sum,
                    tau=self.tau,
                    mask=mask,
                    total_layers=self.n_layers * self.repeats,
                    continue_mask=continue_mask,
                )
                if self.gather_stats:
                    self._latent_vectors[-1].append(
                        self._temp_lm_head(x[:, 10, :].detach()).detach().cpu()
                    )
                    if expert_sel[0] is not None:
                        self._expert_sel[-1].append(expert_sel)
                    self._residual_magnitudes[-1].append(
                        torch.norm(x, dim=-1).detach().clone().cpu()
                    )

        return x, energy_per_sample
