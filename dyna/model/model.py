from abc import ABC, abstractmethod
import math
from collections.abc import Callable
from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from composer.models import HuggingFaceModel
from torch.nn.modules.normalization import RMSNorm
from llmfoundry.utils.builders import build_metric
from transformers import PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from torch.nn import Module, ModuleList, Parameter
from dyna.model.cvmm import CVMMSel, cvmm, cvmm_prepare_sel2
from beartype import beartype
from .model_config import NormStructure, RescaleMethod, ExecutionMode

# from composer.callbacks
# Add jaxtyping imports
from jaxtyping import Float, Int, Bool
from torch import Tensor

# Constants
CROSS_ENTROPY_IGNORE_INDEX = -100
DEFAULT_CAUSAL_LM_TRAIN_METRICS = [
    "language_cross_entropy",
    "language_perplexity",
    "token_accuracy",
]
PROT_EMB_RESCALING_METHODS = [
    RescaleMethod.cum_avg_prot_emb,
    RescaleMethod.sqrt_prot_emb,
]


@beartype
def get_targets(labels: Int[Tensor, "batch seq"]) -> Int[Tensor, "batch seq"]:
    """Shift labels for causal language modeling."""
    targets = torch.roll(labels, shifts=-1)
    targets[:, -1] = CROSS_ENTROPY_IGNORE_INDEX
    return targets


@beartype
def compute_loss_from_logits(
    outputs: CausalLMOutputWithPast,
    shift_labels: bool,
    labels: Int[Tensor, "batch seq"],
    loss_fn: Module,
) -> Float[Tensor, ""]:
    """Compute cross-entropy loss from logits and labels."""
    targets = get_targets(labels) if shift_labels else labels

    losses = loss_fn(
        outputs.logits.view(-1, outputs.logits.size(-1)),
        targets.view(-1),
    )

    if torch.all(targets == loss_fn.ignore_index):
        loss = losses.sum()
    else:
        loss = losses.sum() / (targets != loss_fn.ignore_index).sum()

    return loss


@beartype
def round_up_to_multiple_of_256(n: torch.Tensor) -> torch.Tensor:
    """Return the smallest number divisible by 256 that is >= n."""
    if n <= 0:
        return torch.tensor(
            256, device=n.device
        )  # Ensure tensor is on the same device as input
    return ((n - 1) // 256 + 1) * 256


@beartype
def log_mean(x: Float[Tensor, "*batch dim"], dim: int = 0) -> Float[Tensor, "*batch"]:
    """Compute log of mean along specified dimension."""
    return x.logsumexp(dim) - math.log(x.shape[dim])


@beartype
def entropy_l(l: Float[Tensor, "*batch dim"]) -> Float[Tensor, "*batch"]:
    """Compute entropy from log probabilities."""
    return -(l * l.exp()).sum(-1)


@beartype
def entropy_reg(sel: Float[Tensor, "*batch n_experts"], dim: int) -> Float[Tensor, ""]:
    """Compute entropy regularization term."""
    sel = F.log_softmax(sel, dim=-1)
    sel = log_mean(sel, dim)
    return -entropy_l(sel).mean()


class DynaConfig(PretrainedConfig):
    """Configuration class for Dyna model."""

    model_type = "dyna"

    def __init__(self, **kwargs):
        super().__init__(**{"model_type": self.model_type})

        # Import required for enum handling
        from .model_config import (
            ModelConfig,
            NormStructure,
            RescaleMethod,
            ExecutionMode,
        )

        # Required parameters with defaults from model_config
        self.vocab_size = kwargs.pop("vocab_size", 49152)
        self.d_model = kwargs.pop("d_model", 412)
        self.n_repeats = kwargs.pop("n_repeats", 18)
        self.n_heads = kwargs.pop("n_heads", 4)
        self.n_experts_ffn = kwargs.pop("n_experts_ffn", 155)
        self.n_experts_attn = kwargs.pop("n_experts_attn", 8)
        self.d_head = kwargs.pop("d_head", 82)
        self.d_ffn = kwargs.pop(
            "d_ffn", 4096
        )  # Default based on typical transformer sizing

        # Handle enums properly
        norm_structure_val = kwargs.pop("norm_structure", "moeut")
        if isinstance(norm_structure_val, str):
            self.norm_structure = NormStructure[norm_structure_val]
        else:
            self.norm_structure = norm_structure_val

        rescaling_method_val = kwargs.pop("rescaling_method", "none")
        if isinstance(rescaling_method_val, str):
            self.rescaling_method = RescaleMethod[rescaling_method_val]
        else:
            self.rescaling_method = rescaling_method_val

        # Parameters with defaults
        self.n_layers = kwargs.pop("n_layers", 2)
        self.k_ffn = kwargs.pop("k_ffn", 12)
        self.k_attn = kwargs.pop("k_attn", 2)
        self.dropout_expert_ffn = kwargs.pop("dropout_expert_ffn", 0.0)
        self.dropout_expert_attn = kwargs.pop("dropout_expert_attn", 0.0)
        self.d_expert_ffn = kwargs.pop("d_expert_ffn", 128)
        self.dropout = kwargs.pop("dropout", 0.0)
        self.reg_entropy = kwargs.pop("reg_entropy", 0.01)
        self.reg_entropy_attn = kwargs.pop("reg_entropy_attn", 0.001)
        self.shift_labels = kwargs.pop("shift_labels", True)
        self.n_expert_shared_ffn = kwargs.pop("n_expert_shared_ffn", 0)
        self.n_expert_shared_attn = kwargs.pop("n_expert_shared_attn", 0)
        self.enable_early_exit = kwargs.pop("enable_early_exit", False)
        self.use_rms_norm = kwargs.pop("use_rms_norm", True)
        self.collect_reg_loss = kwargs.pop("collect_reg_loss", False)
        self.device = kwargs.pop("device", "cuda")
        # Handle execution_mode enum
        execution_mode_val = kwargs.pop("execution_mode", "moe")
        if isinstance(execution_mode_val, str):
            self.execution_mode = ExecutionMode[execution_mode_val]
        else:
            self.execution_mode = execution_mode_val


@beartype
class DynaModule(Module, ABC):
    @abstractmethod
    def get_reg_loss(self) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, token_stream: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def reset_parameters(self, std_scale: float) -> None:
        pass


@beartype
class AttentionModule(DynaModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def attend(
        self,
        v: Float[Tensor, "batch n_heads seq d_head"],
        k: Float[Tensor, "batch n_heads seq d_head"],
        q: Float[Tensor, "batch n_heads seq d_head"],
        attention_mask: Bool[Tensor, "batch n_heads max_len src_len"],
        position_mask_trimed: Int[Tensor, "batch max_len"],
        position_mask_full: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch n_heads seq d_head"]:
        """Compute attention with RoPE."""
        # Apply rotary position encoding
        # Remove debug print that could cause issues
        if self.n_rotate > 0:
            q, k = self._apply_rope(q, k, position_mask_trimed, position_mask_full)

        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask
        )

    def _trim_attention_mask(
        self,
        seq_len: int,
        mask: tuple[Bool[Tensor, "batch seq seq"], Int[Tensor, "batch seq"]],
        skip_mask: None | Bool[Tensor, "batch seq"],
    ) -> tuple[
        Bool[Tensor, "batch n_heads max_len seq"],
        Int[Tensor, "batch max_len"],
    ]:
        if skip_mask is None:
            return mask[0].unsqueeze(1).expand(-1, self.n_heads, -1, -1), mask[1]
        lengths = skip_mask.sum(dim=1)
        max_len = round_up_to_multiple_of_256(lengths.max())

        attention_mask: Bool[Tensor, "batch n_heads max_len seq"] = torch.zeros(
            skip_mask.shape[0],
            self.n_heads,
            max_len,
            seq_len,
            dtype=torch.bool,
            device=skip_mask.device,
        )
        position_ids: Int[Tensor, "batch max_len"] = torch.zeros(
            skip_mask.shape[0],
            max_len,
            dtype=torch.int,
            device=skip_mask.device,
        )

        for i in range(skip_mask.shape[0]):
            idx = skip_mask[i].nonzero(as_tuple=False).squeeze(-1)
            n = idx.numel()
            if n > 0:
                attention_mask[i, :, :n] = mask[0][i, idx]
                position_ids[i, :n] = mask[1][i, idx]

        return (attention_mask, position_ids)

    def project_to_torch_order(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor to PyTorch attention format."""
        return x.view(*x.shape[:-1], self.n_heads, self.d_head).transpose(-2, -3)

    def _apply_rope(
        self,
        q: Float[Tensor, "batch n_heads seq d_head"],
        k: Float[Tensor, "batch n_heads seq d_head"],
        position_mask_trimed: Int[Tensor, "batch max_len"],
        position_mask_full: Int[Tensor, "batch seq"],
    ) -> tuple[
        Float[Tensor, "batch n_heads seq d_head"],
        Float[Tensor, "batch n_heads seq d_head"],
    ]:
        """Apply rotary position encoding to queries and keys."""
        if self.n_rotate < self.d_head:
            # Split rotated and non-rotated parts
            r_k, nr_k = k[..., : self.n_rotate], k[..., self.n_rotate :]
            r_q, nr_q = q[..., : self.n_rotate], q[..., self.n_rotate :]

            # Apply RoPE to rotated parts
            r_q, r_k = self.pe(r_q, r_k, position_mask_trimed, position_mask_full)

            # Concatenate back
            return (torch.cat([r_q, nr_q], dim=-1), torch.cat([r_k, nr_k], dim=-1))
        else:
            # Apply RoPE to entire tensors
            return self.pe(q, k, position_mask_trimed, position_mask_full)


class DummyAttention(AttentionModule):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        rotate_fraction: float = 1.0,
        rope_base: float = 10000,
    ):
        super().__init__()
        # Model configuration
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else lambda x: x

        # Query and Key projections (shared)
        self.q = torch.nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.k = torch.nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.v = torch.nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.o = torch.nn.Linear(self.n_heads * self.d_head, self.d_model, bias=False)

        # RoPE configuration
        self.n_rotate = int(rotate_fraction * self.d_head)
        if self.n_rotate > 0:
            self.pe = RotaryPosEncoding(self.n_rotate, seq_dim=-2, base=rope_base)

        # This might be it?
        # The scaled dot prod has a scale of 1 because k and v were rescaled
        # # Attention scale
        # self.register_buffer(
        #     "scale",
        #     torch.full([1], 1.0 / math.sqrt(self.d_head)),
        #     persistent=False,
        # )
    @torch.no_grad
    def reset_parameters(self, std_scale: float) -> None:
        # Initialize projection parameters
        torch.nn.init.normal_(self.k.weight, 0, std_scale / math.sqrt(self.d_model))
        torch.nn.init.normal_(self.q.weight, 0, std_scale / math.sqrt(self.d_model))
        torch.nn.init.normal_(self.v.weight, 0, std_scale / math.sqrt(self.d_model))
        # FIX: Use proper scaling for output projection
        torch.nn.init.normal_(self.o.weight, 0, std_scale / math.sqrt(self.n_heads * self.d_head))

    def forward(
        self,
        q_src: Float[Tensor, "batch seq d_model"],
        k_src: Float[Tensor, "batch seq d_model"],
        v_src: Float[Tensor, "batch seq d_model"],
        mask: tuple[Bool[Tensor, "batch seq seq"], Int[Tensor, "batch seq"]],
        skip_mask: None | Bool[Tensor, "batch seq"],
    ) -> tuple[
        Float[Tensor, "batch seq d_model"],
        tuple[None, None],
    ]:

        return q_src, (None, None)

    def get_reg_loss(self) -> torch.Tensor:
        """Return zero for regularization loss since DummyAttention doesn't use expert routing."""
        return torch.tensor(0.0, device=self.q.weight.device, dtype=self.q.weight.dtype)


@beartype
class LayerModule(Module, ABC):
    def __init__(
        self,
        config: DynaConfig,
        attention_module: AttentionModule,
        ffn_module: DynaModule,
    ):
        super().__init__()
        self.attention = attention_module
        self.ffn = ffn_module
        # Layer normalization - configurable type
        norm_class = RMSNorm if config.use_rms_norm else torch.nn.LayerNorm
        self.attn_pre = norm_class(config.d_model)
        self.attn_post = norm_class(config.d_model)
        self.attn_post.requires_grad_(
            config.norm_structure in [NormStructure.peri, NormStructure.post]
        )
        self.ffn_pre = norm_class(config.d_model)
        self.ffn_post = norm_class(config.d_model)
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
    def _check_early_exit(
        self,
        x: Float[Tensor, "batch seq d_model"],
        router: Parameter,
        cum_sum: Float[Tensor, "batch seq"],
        tau: Float[Tensor, "1"],
    ) -> tuple[
        None | Bool[Tensor, "batch seq"], bool, None | Float[Tensor, "batch seq"]
    ]:
        """Check if tokens should exit early and create skip mask."""
        if not self.enable_early_exit:
            # Return mask that keeps all tokens
            # skip_mask = torch.ones_like(cum_sum, dtype=torch.bool)
            return None, True, None

        # Compute exit scores
        s_exit: Float[Tensor, "batch seq"] = F.sigmoid(F.linear(x, router))
        cum_sum += s_exit

        # Create skip mask (True = continue processing)
        skip_mask: Bool[Tensor, "batch seq"] = cum_sum < tau

        # Handle sequence-level early exit
        last_token_idx = x.shape[1] - 1
        last_tokens_exit = ~skip_mask[:, last_token_idx]

        # Mark entire sequences as done if last token exits
        for batch_idx in last_tokens_exit.nonzero(as_tuple=True)[0]:
            skip_mask[batch_idx, :] = False

        # Check if all sequences are done
        continue_processing = not torch.all(skip_mask == False)

        return skip_mask, continue_processing, s_exit

    # Done
    def _trim_sequence(
        self,
        x: Float[Tensor, "batch seq d_model"],
        skip_mask: None | Bool[Tensor, "batch seq"],
    ) -> tuple[Float[Tensor, "batch max_len d_model"], int]:
        """Pack sequences for efficient processing."""

        if skip_mask is None:
            return x, x.shape[-2]

        lengths = skip_mask.sum(dim=1)
        max_len = round_up_to_multiple_of_256(lengths.max())

        # Initialize output tensors
        x_trimed: Float[Tensor, "batch max_len d_model"] = torch.zeros(
            x.shape[0], max_len, x.shape[-1], device=x.device
        )

        # Get valid positions
        batch_idx, seq_idx = skip_mask.nonzero(as_tuple=True)

        if batch_idx.numel() > 0:
            # Compute packing indices efficiently
            batch_counts = torch.bincount(batch_idx, minlength=x.shape[0])
            cumsum_counts = torch.cumsum(
                torch.cat([torch.tensor([0], device=x.device), batch_counts[:-1]]),
                dim=0,
            )

            local_indices = (
                torch.arange(batch_idx.numel(), device=x.device)
                - cumsum_counts[batch_idx]
            )
            valid_mask = local_indices < max_len

            if valid_mask.any():
                # Pack data efficiently
                x_trimed[batch_idx[valid_mask], local_indices[valid_mask]] = x[
                    batch_idx[valid_mask], seq_idx[valid_mask]
                ]

        return x_trimed, max_len

    # Done
    def _apply_pre_norm_attn(
        self,
        residual_stream: Float[Tensor, "batch seq d_model"],
        skip_mask: None | Bool[Tensor, "batch seq"],
    ) -> tuple[
        Float[Tensor, "batch max_len d_model"],
        Float[Tensor, "batch seq d_model"],
        Float[Tensor, "batch seq d_model"],
        int,
    ]:
        if (
            self.norm_structure.value == NormStructure.peri.value
            or self.norm_structure == NormStructure.pre.value
        ):
            # Peri, Pre
            residual_stream_normed = self.attn_pre(residual_stream)
            residual_stream_trimmed, max_len = self._trim_sequence(
                residual_stream_normed, skip_mask
            )
            q_val = residual_stream_trimmed
            k_val = residual_stream_normed
            v_val = residual_stream_normed

        elif self.norm_structure.value == NormStructure.post.value:
            residual_stream_trimmed, max_len = self._trim_sequence(
                residual_stream, skip_mask
            )
            q_val = residual_stream_trimmed
            k_val = residual_stream
            v_val = residual_stream

        elif self.norm_structure.value == NormStructure.moeut.value:
            residual_stream_normed = self.attn_pre(residual_stream)
            residual_stream_trimmed, max_len = self._trim_sequence(
                residual_stream_normed, skip_mask
            )
            q_val = residual_stream_trimmed
            k_val = residual_stream_normed
            v_val = residual_stream
        else:
            raise ValueError(f"{self.norm_structure} must be one of {NormStructure}")

        return q_val, k_val, v_val, max_len

    def _apply_update_to_residual(
        self,
        residual_stream: Float[Tensor, "batch seq d_model"],
        update_on_stream: Float[Tensor, "batch seq d_model"],
        skip_mask: None | Bool[Tensor, "batch seq"],
        cum_sum: Float[Tensor, "batch seq"],
        tau: Float[Tensor, "1"],
        layer_index: int,
        norm_to_use: Module,
        e: Optional[Float[Tensor, "batch seq d_model"]] = None,
    ) -> Float[Tensor, "batch seq d_model"]:
        update = update_on_stream
        if self.norm_structure.value == NormStructure.peri.value:
            update = norm_to_use(update_on_stream)
        update = self.drop(update)

        match self.rescaling_method.value:
            case RescaleMethod.none.value:
                if self.enable_early_exit:
                    residual_stream[skip_mask] = (
                        residual_stream[skip_mask]
                        + update.view(-1, update.shape[-1])[: skip_mask.sum()]
                    )
                else:
                    residual_stream = residual_stream + update
            case (
                RescaleMethod.cum_avg_prot_emb.value
                | RescaleMethod.cum_avg_no_prot_emb.value
            ):
                if e is not None:
                    residual_stream = residual_stream - e
                if self.enable_early_exit:
                    scale_factor = (layer_index - 1) / layer_index
                    update_factor = cum_sum[skip_mask].unsqueeze(1) * tau / layer_index

                    residual_stream[skip_mask] = (
                        scale_factor * residual_stream[skip_mask]
                        + update.view(-1, update.shape[-1])[: skip_mask.sum()]
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
            case (
                RescaleMethod.sqrt_prot_emb.value | RescaleMethod.sqrt_no_prot_emb.value
            ):
                if e is not None:
                    residual_stream = residual_stream - e
                if self.enable_early_exit:
                    scale_factor = torch.sqrt(layer_index) - 1 / torch.sqrt(layer_index)
                    update_factor = (
                        cum_sum[skip_mask].unsqueeze(1) * tau / torch.sqrt(layer_index)
                    )

                    residual_stream[skip_mask] = (
                        scale_factor * residual_stream[skip_mask]
                        + update.view(-1, update.shape[-1])[: skip_mask.sum()]
                        * update_factor
                    )
                else:
                    # Apply to all tokens when early exit is disabled
                    scale_factor = torch.sqrt(layer_index) - 1 / torch.sqrt(layer_index)
                    residual_stream = scale_factor * residual_stream + update / (
                        torch.sqrt(layer_index)
                    )
                if e is not None:
                    residual_stream = residual_stream + e

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
        router: Parameter,
        cum_sum: Float[Tensor, "batch seq"],
        tau: Float[Tensor, "1"],
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


@beartype
class SigmaMoE(DynaModule):
    """Sigma Mixture of Experts layer for feed-forward networks."""

    def __init__(
        self,
        d_model: int,
        n_experts_ffn: int,
        d_expert_ffn: int,
        n_expert_shared_ffn: int,
        k_ffn: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        dropout_expert: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts_ffn = n_experts_ffn
        self.d_expert_ffn = d_expert_ffn
        self.n_expert_shared_ffn = min(n_expert_shared_ffn, n_experts_ffn)
        self.n_expert_routed_ffn = n_experts_ffn - self.n_expert_shared_ffn
        self.k_ffn = k_ffn
        self.activation = activation
        self.dropout_expert = dropout_expert

        # Bias tracking for load balancing
        self.register_buffer("bias_ffn", torch.zeros(n_experts_ffn))
        self.bias_update_lr = 0.001

        # Expert parameters
        self.keys = Parameter(
            torch.empty(self.n_experts_ffn, self.d_model, self.d_expert_ffn)
        )
        self.values = Parameter(
            torch.empty(self.n_experts_ffn, self.d_expert_ffn, self.d_model)
        )
        self.expert_sel = Parameter(torch.empty(self.n_experts_ffn, self.d_model))

        # Register shared expert indices
        self.register_buffer(
            "expert_shared",
            torch.arange(
                n_experts_ffn - self.n_expert_shared_ffn,
                n_experts_ffn,
                dtype=torch.long,
            ),
        )

        self.selection_history_s_moe = []
    @torch.no_grad
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
            weight.div_(weight.norm(dim=dim, keepdim=True))
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
        affinity: Float[Tensor, "batch seq n_experts"] = F.sigmoid(
            F.linear(
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
            shape_expert_shared = selection_index.shape[:-1] + (
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
        if self.training and self.n_expert_routed_ffn > 0:
            with torch.no_grad():  # Prevent gradient accumulation
                c_i = torch.bincount(
                    selection_index.flatten(), minlength=self.n_experts_ffn
                )
                c_i_avg = torch.mean(c_i, dtype=torch.float32)
                self.bias_ffn[
                    : self.n_expert_routed_ffn
                ] += self.bias_update_lr * torch.sign(
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
        # self.selection_history_s_moe.append(affinity.clone().detach())  # Detach to avoid storing gradients

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
        """Get regularization loss and reset selection history."""
        if not self.selection_history_s_moe:
            return torch.tensor(
                0.0, device=self.keys.device
            )  # Ensure tensor is on the correct device

        # Average over time and layers
        loss = entropy_reg(
            torch.stack(self.selection_history_s_moe, dim=-2).flatten(-3, -2), -2
        )
        # Clear the history to prevent memory accumulation
        self.selection_history_s_moe.clear()
        return loss


class BasicFFN(DynaModule):
    def __init__(
        self,
        d_model: int,
        d_expert_ffn: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
    ):
        super().__init__()  # Need to call the parent class constructor
        self.d_model = d_model
        self.d_expert_ffn = d_expert_ffn
        self.activation = activation
        self.projection_up = torch.nn.Linear(self.d_model, self.d_expert_ffn)
        self.projection_down = torch.nn.Linear(self.d_expert_ffn, self.d_model)

    def forward(
        self, token_stream: torch.Tensor, selection_input: torch.Tensor
    ) -> tuple[torch.Tensor, None]:  # Match return type with SigmaMoE
        output = self.projection_down(self.activation(self.projection_up(token_stream)))
        return output, None  # Return None for the selection index to match SigmaMoE
    @torch.no_grad
    def reset_parameters(self, std_scale: float) -> None:
        """Initialize parameters with proper scaling."""
        torch.nn.init.normal_(
            self.projection_up.weight, 0, std_scale / math.sqrt(self.d_model)
        )
        torch.nn.init.normal_(
            self.projection_down.weight, 0, std_scale / math.sqrt(self.d_expert_ffn)
        )
        if self.projection_up.bias is not None:
            torch.nn.init.zeros_(self.projection_up.bias)
        if self.projection_down.bias is not None:
            torch.nn.init.zeros_(self.projection_down.bias)

    def get_reg_loss(self) -> torch.Tensor:
        """Return zero for regularization loss since BasicFFN doesn't use expert routing."""
        return torch.tensor(0.0, device=self.projection_up.weight.device)


class BasicAttn(AttentionModule):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        rotate_fraction: float = 1.0,
        rope_base: float = 10000,
    ):
        super().__init__()
        # Model configuration
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else lambda x: x

        # Query and Key projections (shared)
        self.q = torch.nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.k = torch.nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.v = torch.nn.Linear(self.d_model, self.n_heads * self.d_head, bias=False)
        self.o = torch.nn.Linear(self.n_heads * self.d_head, self.d_model, bias=False)

        # RoPE configuration
        self.n_rotate = int(rotate_fraction * self.d_head)
        if self.n_rotate > 0:
            self.pe = RotaryPosEncoding(self.n_rotate, seq_dim=-2, base=rope_base)

        # # Attention scale
        # self.register_buffer(
        #     "scale",
        #     torch.full([1], 1.0 / math.sqrt(self.d_head)),
        #     persistent=False,
        # )
    @torch.no_grad
    def reset_parameters(self, std_scale: float) -> None:
        # Initialize projection parameters
        torch.nn.init.normal_(self.k.weight, 0, std_scale / math.sqrt(self.d_model))
        torch.nn.init.normal_(self.q.weight, 0, std_scale / math.sqrt(self.d_model))
        torch.nn.init.normal_(self.v.weight, 0, std_scale / math.sqrt(self.d_model))
        # FIX: Use proper scaling for output projection
        torch.nn.init.normal_(self.o.weight, 0, std_scale / math.sqrt(self.n_heads * self.d_head))

    def get_reg_loss(self) -> torch.Tensor:
        """Return zero for regularization loss since BasicAttn doesn't use expert routing."""
        return torch.tensor(0.0, device=self.q.weight.device)

    def forward(
        self,
        q_src: Float[Tensor, "batch seq d_model"],
        k_src: Float[Tensor, "batch seq d_model"],
        v_src: Float[Tensor, "batch seq d_model"],
        mask: tuple[Bool[Tensor, "batch seq seq"], Int[Tensor, "batch seq"]],
        skip_mask: None | Bool[Tensor, "batch seq"],
    ) -> tuple[
        Float[Tensor, "batch seq d_model"],
        tuple[None, None],
    ]:
        """Forward pass through the attention layer."""

        q: Float[Tensor, "batch seq n_heads*d_head"] = self.q(q_src)
        k: Float[Tensor, "batch seq n_heads*d_head"] = self.k(k_src)
        v: Float[Tensor, "batch seq n_heads*d_head"] = self.v(v_src)

        # Project to attention format

        q = self.project_to_torch_order(q)
        k = self.project_to_torch_order(k)
        v = self.project_to_torch_order(v)

        # Apply dropout
        q = self.dropout(q)

        # Prepare attention mask
        attention_mask, position_ids = self._trim_attention_mask(
            v.shape[-2], mask, skip_mask
        )

        # Apply attention
        res = self.attend(v, k, q, attention_mask, position_ids, mask[1])
        # Reshape result for output projection
        res = res.transpose(-2, -3).contiguous().view(res.shape[0], res.shape[2], -1)

        # Apply output projection
        out = self.o(res)
        return out, (None, None)


@beartype
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
        rope_base: float = 10000,
    ):
        super().__init__()
        # Model configuration
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else lambda x: x

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
        self.sel_hist = []

        self.call_h = 0
        # RoPE configuration
        self.n_rotate = int(rotate_fraction * self.d_head)
        if self.n_rotate > 0:
            self.pe = RotaryPosEncoding(self.n_rotate, seq_dim=-2, base=rope_base)

    def _init_expert_parameters(self) -> None:
        """Initialize expert-specific parameters."""
        # Value and output projections for multiple experts
        self.v = Parameter(
            torch.empty(self.n_heads * self.n_experts_attn, self.d_model, self.d_head)
        )
        self.o = Parameter(
            torch.empty(self.n_heads * self.n_experts_attn, self.d_head, self.d_model)
        )

        # Expert selection parameters
        self.sel_v = Parameter(
            torch.empty(self.n_heads * self.n_experts_attn, self.d_model)
        )
        self.sel_o = Parameter(
            torch.empty(self.n_heads * self.n_experts_attn, self.d_model)
        )

        # Bias parameters for load balancing
        self.bias_v = Parameter(torch.zeros(self.n_experts_attn), requires_grad=False)
        self.bias_o = Parameter(torch.zeros(self.n_experts_attn), requires_grad=False)

    @torch.no_grad
    def reset_parameters(self, std_scale: float) -> None:
        """Initialize all parameters with proper scaling."""
        # Initialize selection parameters
        if self.n_experts_attn > 1:
            torch.nn.init.normal_(self.sel_v, 0, std_scale / math.sqrt(self.d_model))
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
            x.div_(x.norm(dim=-1, keepdim=True))
            x.mul_(std_t / x.std())

    def get_reg_loss(self) -> Float[Tensor, ""]:
        """Get regularization loss from selection history."""
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        if self.sel_hist:
            for i in range(len(self.sel_hist[0])):
                loss = loss + entropy_reg(
                    torch.stack([l[i] for l in self.sel_hist], dim=-3).flatten(-4, -3),
                    -3,
                )
        # Clear the history to prevent memory accumulation
        self.sel_hist.clear()
        return loss

    def _get_expert_selection(
        self,
        input_tensor: Float[Tensor, "batch seq d_model"],
        weight: Float[Tensor, "n_heads_x_experts d_model"],
        bias: Optional[Float[Tensor, "n_experts"]] = None,
    ) -> tuple[
        CVMMSel,
        Float[Tensor, "batch seq n_heads n_experts"],
        Int[Tensor, "batch seq n_heads k_experts"],
    ]:
        """Get expert selection indices and weights."""
        # Compute selection scores - remove explicit float() cast
        affinity: Float[Tensor, "batch seq n_heads_x_experts"] = F.linear(
            input_tensor, weight
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
                *([1] * (sel_index.dim() - 1)), -1
            ).expand(shared_shape)
            sel_index = torch.cat([sel_index, expert_shared_expanded], dim=-1)

        # Update bias for load balancing
        if self.training and bias is not None:
            with torch.no_grad():
                c_i = torch.bincount(sel_index.flatten(), minlength=self.n_experts_attn)
                c_i_avg = torch.mean(c_i, dtype=torch.float32)
                bias[: self.n_expert_routed_attn] += self.bias_update_lr * torch.sign(
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
        skip_mask: None | Bool[Tensor, "batch seq"],
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
            o_sel, o_sel_r, o_sel_inedx = self._get_expert_selection(
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
        else:
            o_gate: Float[Tensor, "batch seq d_model"] = F.sigmoid(
                F.linear(q_src, self.sel_o)
            )

            v = torch.einsum("bsd,ndh->bsnh", v_src, self.v)
            v = self.project_to_torch_order(v.reshape(v.shape[0], v.shape[1], -1))

        # Project to attention format
        q: Float[Tensor, "batch n_heads seq d_head"] = self.project_to_torch_order(q)
        k: Float[Tensor, "batch n_heads seq d_head"] = self.project_to_torch_order(k)

        # Apply dropout and attention
        q = self.dropout(q)
        attention_mask = self._trim_attention_mask(v.shape[-2], mask, skip_mask)

        res: Float[Tensor, "batch n_heads seq d_head"] = self.attend(
            v, k, q, attention_mask[0], attention_mask[1], mask[1]
        )
        res = res.transpose(-2, -3)

        # Apply output projection
        if self.n_experts_attn > 1:
            o_sel.sel_index = o_sel.out_index // o_sel.reduction_weight.shape[-1]
            o_sel.reduction_weight = o_sel.reduction_weight.flatten(-2)
            out: Float[Tensor, "batch seq d_model"] = cvmm(res, o_sel, self.o)

            # Clean up intermediate tensors
            del o_sel, res

            # Return None instead of storing tensors

        else:
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


@beartype
class RotaryPosEncoding(Module):
    """Rotary Position Encoding (RoPE) implementation."""

    def __init__(self, d_model: int, base: int = 10000, seq_dim: int = 1):
        super().__init__()

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for efficiency
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        self.seq_dim = torch.tensor(seq_dim)

    # In-place version for maximum memory efficiency
    def rotate_half_optimized(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized rotation of the second half of the last dimension."""
        # Avoid redundant shape calculation and use tensor.chunk for cleaner split
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=x1.ndim - 1)

    def apply_rot_optimized(
        self,
        x: torch.Tensor,  # [batch, n_heads, seq, d_head]
        positions: torch.Tensor,  # [batch, seq]
    ) -> torch.Tensor:
        """Optimized rotary position encoding application."""

        sin, cos = self.get_sincos_positions(positions, x)

        # Get sequence length once
        seq_len = x.shape[self.seq_dim]

        # Use slice instead of narrow (more readable, same performance)
        sin = sin[..., :seq_len, :]
        cos = cos[..., :seq_len, :]

        # Apply rotation in one line using the optimized rotate_half
        return x * cos + self.rotate_half_optimized(x) * sin

    def get_sincos_positions(
        self,
        positions: Int[Tensor, "batch seq"],
        q: Float[Tensor, "batch n_heads seq d_head"],
    ) -> tuple[
        Float[Tensor, "batch 1 seq d_head"], Float[Tensor, "batch 1 seq d_head"]
    ]:
        """Get sin/cos values for specific positions."""
        freqs = torch.einsum("ki,j->kij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(positions.device)

        tgt_shape = [1] * q.ndim
        tgt_shape[0] = q.shape[0]
        tgt_shape[1] = 1
        tgt_shape[self.seq_dim] = q.shape[self.seq_dim]
        tgt_shape[-1] = q.shape[-1]

        return emb.sin().view(*tgt_shape), emb.cos().view(*tgt_shape)

    def forward(
        self,
        q: Float[Tensor, "batch n_heads seq d_head"],
        k: Float[Tensor, "batch n_heads seq d_head"],
        position_mask_trimed: Int[Tensor, "batch max_len"],
        position_mask_full: Int[Tensor, "batch seq"],
    ) -> tuple[
        Float[Tensor, "batch n_heads seq d_head"],
        Float[Tensor, "batch n_heads seq d_head"],
    ]:
        """Apply RoPE to query and key tensors."""
        return (
            self.apply_rot_optimized(q, position_mask_trimed),
            self.apply_rot_optimized(k, position_mask_full),
        )


@beartype
class MoEUTLayer(LayerModule):
    """Single layer of the MoEUT model with configurable behavior."""

    def __init__(self, config: DynaConfig):
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
            ),
            ffn_module=SigmaMoE(
                config.d_model,
                config.n_experts_ffn,
                config.d_expert_ffn,
                k_ffn=config.k_ffn,
                dropout_expert=config.dropout_expert_ffn,
                n_expert_shared_ffn=config.n_expert_shared_ffn,
            ),
        )

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        layer_index: int,
        e: None | Float[Tensor, "batch seq d_model"],
        router: Parameter,
        cum_sum: Float[Tensor, "batch seq"],
        tau: Float[Tensor, "1"],
        mask: tuple[Bool[Tensor, "batch seq seq"], Int[Tensor, "batch seq"]],
    ) -> tuple[
        Float[Tensor, "batch seq d_model"],
        bool,
        int,
        None | Float[Tensor, "batch seq"],
        tuple,
    ]:
        """Forward pass through the layer with configurable behavior."""
        skip_mask, continue_processing, s_exit = self._check_early_exit(
            x, router, cum_sum, tau
        )

        if not continue_processing:
            return x, False, 0, s_exit, ((None, None), None)

        # === ATTENTION BLOCK ===
        q_val, k_val, v_val, max_len = self._apply_pre_norm_attn(x, skip_mask)

        att_out, expert_sel_attn = self.attention(
            q_val,
            k_val,
            v_val,
            mask,
            skip_mask=skip_mask,
        )

        # Clean up attention intermediate tensors immediately
        del q_val, k_val, v_val

        x = self._apply_update_to_residual(
            x, att_out, skip_mask, cum_sum, tau, layer_index, self.attn_post, e
        )

        # Clean up attention output
        del att_out

        # === FFN BLOCK ===

        # FFN computation
        ffn_out, expert_sel_ffn = self.ffn(*self._apply_pre_norm_ffn(x))

        x = self._apply_update_to_residual(
            x, ffn_out, skip_mask, cum_sum, tau, layer_index, self.ffn_post, e
        )

        # Clean up FFN output and other intermediate tensors
        del ffn_out, skip_mask

        return (
            x,
            continue_processing,
            max_len,
            s_exit,
            (expert_sel_attn, expert_sel_ffn),
        )


@beartype
class SimpleLayer(LayerModule):
    def __init__(self, config: DynaConfig):

        super().__init__(
            config,
            BasicAttn(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_head=config.d_head,
                dropout=config.dropout,
            ),
            BasicFFN(
                config.d_model,
                config.d_ffn,
            ),
        )

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        layer_index: int,
        e: None | Float[Tensor, "batch seq d_model"],
        router: Parameter,
        cum_sum: Float[Tensor, "batch seq"],
        tau: Float[Tensor, "1"],
        mask: tuple[Bool[Tensor, "batch seq seq"], Int[Tensor, "batch seq"]],
    ) -> tuple[
        Float[Tensor, "batch seq d_model"],
        bool,
        int,
        None | Float[Tensor, "batch seq"],
        tuple,
    ]:
        """Forward pass through the layer with configurable behavior."""
        skip_mask, continue_processing, s_exit = self._check_early_exit(
            x, router, cum_sum, tau
        )
        if not continue_processing:
            return x, False, 0, s_exit, ((None, None), None)

        # === ATTENTION BLOCK ===
        q_val, k_val, v_val, max_len = self._apply_pre_norm_attn(x, skip_mask)
        att_out, expert_sel_attn = self.attention(
            q_val,
            k_val,
            v_val,
            mask,
            skip_mask=skip_mask,
        )

        x = self._apply_update_to_residual(
            x, att_out, skip_mask, cum_sum, tau, layer_index, self.attn_post, e
        )

        # === FFN BLOCK ===

        # FFN computation
        ffn_out, expert_sel_ffn = self.ffn(*self._apply_pre_norm_ffn(x))

        x = self._apply_update_to_residual(
            x, ffn_out, skip_mask, cum_sum, tau, layer_index, self.ffn_post, e
        )

        return (
            x,
            continue_processing,
            max_len,
            s_exit,
            (expert_sel_attn, expert_sel_ffn),
        )


@beartype
class DynaPretrainedModel(PreTrainedModel):
    """Base class for Dyna pretrained models."""

    config_class = DynaConfig
    base_model_prefix: str = "Dyna"
    is_parallelizable: bool = False
    main_input_name: str = "input_ids"
    load_tf_weights = None


@beartype
class DynaFormer(DynaPretrainedModel):
    """MoEUT transformer model with configurable behavior."""

    def __init__(self, config: DynaConfig):
        super().__init__(config)
        self.reg_entropy = config.reg_entropy
        self.reg_entropy_attn = config.reg_entropy_attn
        self.n_layers = config.n_layers
        self.n_repeats = config.n_repeats
        self.d_model = config.d_model
        self.enable_early_exit = config.enable_early_exit
        self.collect_reg_loss = config.collect_reg_loss
        self.router = Parameter(
            torch.zeros(self.d_model, device=config.device), requires_grad=False
        )
        self.tau = Parameter(
            torch.ones(1, device=config.device), requires_grad=self.enable_early_exit
        )
        self.gather_stats = False
        match config.execution_mode.value:
            case ExecutionMode.moe.value:
                self.layers = ModuleList(
                    [MoEUTLayer(config) for _ in range(config.n_layers)]
                )
            case ExecutionMode.transformer.value:
                self.layers = ModuleList(
                    [SimpleLayer(config) for _ in range(config.n_layers)]
                )
            case _:
                raise ValueError(
                    f"{config.execution_mode} needs to be one of {ExecutionMode}"
                )

        # Initialize parameters
        self.reset_parameters()

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

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        e: Float[Tensor, "batch seq d_model"] | None,
        mask: tuple[Bool[Tensor, "batch seq seq"], Int[Tensor, "batch seq"]],
    ) -> Float[Tensor, "batch seq d_model"]:
        """Forward pass through the model."""
        self._expert_sel.append([])
        self._exit_logits.append([])
        self._latent_vectors.append([])
        self._seq_len.append([])
        cum_sum = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)

        continue_processing = True
        for li in range(self.n_repeats):
            for idx, layer in enumerate(self.layers):
                # Forward through layer
                x, continue_processing, seq_lengths, s_exit, expert_sel = layer(
                    x, li + idx + 2, e, self.router, cum_sum, self.tau, mask
                )
                # Clean up intermediate variables immediately
                # del seq_lengths, s_exit, expert_sel

                # make continue_processing just be conditioned on the last token
                if not continue_processing:
                    break
                if self.gather_stats:
                    # Track sequence lengths and entropy for analysis
                    # self._seq_len[-1].append(copy.deepcopy(seq_lengths))
                    self._latent_vectors[-1].append(x[:, -1, :])
                    # self._exit_logits[-1].append(s_exit)
                    self._expert_sel[-1].append(expert_sel)

            if not continue_processing:
                break

        # Collect regularization loss if enabled
        reg_loss = None
        if self.collect_reg_loss:
            reg_loss = self._collect_regularization_loss()

        return x


# Done
@beartype
class DynaLM(DynaPretrainedModel):
    """MoEUT Language Model with embedding and output layers."""

    def __init__(self, config: DynaConfig, eos_token_id: int):
        super().__init__(config)

        # Core transformer
        self.transformer = DynaFormer(config)

        # Model configuration
        self.n_repeats = config.n_repeats
        self.use_rms_norm = config.use_rms_norm
        self.d_model = config.d_model
        # Input/output layers
        
        self.embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Output normalization - configurable type
        norm_class = RMSNorm if config.use_rms_norm else torch.nn.LayerNorm
        self.out_norm = norm_class(config.d_model)
        self.eos_token_id = eos_token_id

        self.rescaling_method = config.rescaling_method
        # Initialize parameters
        self.reset_parameters()

        # Provide LM head to transformer for entropy computation
        self.transformer._temp_lm_head = lambda x: self.lm_head(self.out_norm(x))

    @torch.no_grad
    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.embedding.weight, mode="fan_in", nonlinearity="linear")
        self.transformer.reset_parameters()

    def _generate_causal_mask(
        self, input_ids: Tensor  # [batch, seq]
    ) -> Tensor:  # [batch, seq, seq]

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create base causal mask - use float16 to save memory if acceptable
        base_causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )

        # Find EOS positions
        eos_mask = input_ids == self.eos_token_id  # [batch, seq]

        # Quick path: if no EOS tokens in entire batch
        if not eos_mask.any():
            return base_causal.unsqueeze(0).expand(batch_size, -1, -1)

        eos_positions = eos_mask.long()  # Convert to int: [batch, seq]

        sequence_ids = torch.cumsum(
            torch.cat(
                [
                    torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                    eos_positions[:, :-1],
                ],
                dim=1,
            ),
            dim=1,
        )  # [batch, seq]

        # Vectorized same-sequence mask computation
        # Use broadcasting instead of unsqueeze for better memory efficiency
        same_seq_mask = (
            sequence_ids[:, :, None] == sequence_ids[:, None, :]
        )  # [batch, seq, seq]

        # Apply masks efficiently
        final_mask = base_causal[None, :, :] & same_seq_mask

        return final_mask

    def _generate_source_len_mask(
        self, attention_mask: Bool[Tensor, "batch seq seq"]
    ) -> Int[Tensor, "batch seq"]:
        """Generate source length mask with position indices for each sequence."""
        batch_size, seq_len, _ = attention_mask.shape
        device = attention_mask.device

        # Create a range tensor for positions [0, 1, 2, ..., seq_len-1]
        pos_range = torch.arange(seq_len, device=device, dtype=torch.long)

        # Create upper triangular mask including diagonal: [[True], [True, True], [True, True, True], ...]
        # This represents which positions each token can attend to
        causal_indices = pos_range.unsqueeze(0) <= pos_range.unsqueeze(1)  # [seq, seq]

        # Expand to batch dimension
        causal_indices = causal_indices.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, seq, seq]

        # Apply the attention mask to get valid positions
        valid_positions = attention_mask & causal_indices  # [batch, seq, seq]

        # Sum along the last dimension to get count of valid positions each token can attend to
        # Subtract 1 to make it 0-indexed
        position_mask = valid_positions.sum(dim=-1) - 1  # [batch, seq]

        return position_mask

    def forward(
        self,
        input_ids: Optional[Int[Tensor, "batch seq"]] = None,
        labels: Optional[Int[Tensor, "batch seq"]] = None,
        inputs_embeds: Optional[Float[Tensor, "batch seq d_model"]] = None,
        attention_mask: Optional[Bool[Tensor, "batch seq seq"]] = None,
        src_len_mask: Optional[Int[Tensor, "batch seq"]] = None,
    ) -> CausalLMOutputWithPast:
        """Forward pass through the language model."""
        # Validate inputs
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")

        # Get embeddings
        if input_ids is not None:
            x = self.embedding(input_ids)
            # print("Expected: embedding, true:",x.grad_fn)
            if attention_mask is None:
                attention_mask = self._generate_causal_mask(input_ids)
            if src_len_mask is None:
                src_len_mask = self._generate_source_len_mask(attention_mask)
        elif isinstance(inputs_embeds, torch.Tensor):
            x = inputs_embeds

        assert attention_mask is not None
        assert src_len_mask is not None

        # Prepare protected embeddings if enabled
        e = x.clone() if self.rescaling_method in PROT_EMB_RESCALING_METHODS else None
        x = self.transformer(x, e, (attention_mask, src_len_mask))
        # print("Expected: transformer, true:",x)
        # Apply output projection
        logits = self.lm_head(self.out_norm(x))
        # print("Expected: lm_head, true:",logits)
        loss = None
        if labels is not None:
            _labels = torch.roll(labels, shifts=-1)
            _labels[:, -1] = CROSS_ENTROPY_IGNORE_INDEX
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                _labels.to(logits.device).view(-1),
            )
            
            
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


# Done
@beartype
class ComposerDynaModel(HuggingFaceModel):
    """Composer-compatible MoEUT model wrapper."""

    def __init__(
        self,
        config: DynaConfig,
        tokenizer: PreTrainedTokenizerBase,
    ):

        model = DynaLM(config, tokenizer.eos_token_id)
        # Configuration
        self.vocab_size = config.vocab_size
        self.shift_labels = config.shift_labels

        # Metrics
        train_metrics = [
            build_metric(metric, {}) for metric in DEFAULT_CAUSAL_LM_TRAIN_METRICS
        ]

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics={},
            shift_labels=config.shift_labels,
            allow_embedding_resizing=True,
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean", ignore_index=CROSS_ENTROPY_IGNORE_INDEX
        )

    def forward(self, batch) -> CausalLMOutputWithPast:
        
        return self.model(
            input_ids=batch.get("input_ids", None),
            labels=batch.get("labels", None),
            inputs_embeds=batch.get("inputs_embeds", None),
            attention_mask=batch.get("attention_mask", None),
        )
    def loss(self, outputs: CausalLMOutputWithPast, batch) -> torch.Tensor:
        labels = batch["labels"]
        logits = outputs.logits
        _labels = torch.roll(labels, shifts=-1)
        _labels[:, -1] = CROSS_ENTROPY_IGNORE_INDEX
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            _labels.to(logits.device).view(-1),
        )
        # loss = compute_loss_from_logits(
        #     outputs,
        #     True,
        #     batch["labels"],
        #     self.loss_fn,
        # )
        return loss
