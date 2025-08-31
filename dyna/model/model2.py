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


class RotaryPositionalEmbeddings(Module):
    """Rotary Positional Embeddings (RoPE) implementation."""
    
    def __init__(self, dim: int, base: int = 10000, max_position_embeddings: int = 2048):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        
        # Create frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def forward(self, x: Tensor, position_ids: Tensor) -> Tensor:
        """
        Apply rotary positional embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch, seq_len, n_heads, head_dim] or [batch, n_heads, seq_len, head_dim]
            position_ids: Position indices of shape [batch, seq_len]
            
        Returns:
            Rotated tensor with same shape as input
        """
        # Handle both possible input shapes
        if x.dim() == 4:
            if x.size(1) == position_ids.size(1):  # [batch, seq_len, n_heads, head_dim]
                batch_size, seq_len, n_heads, head_dim = x.shape
                needs_transpose = False
            else:  # [batch, n_heads, seq_len, head_dim]
                batch_size, n_heads, seq_len, head_dim = x.shape
                needs_transpose = True
                x = x.transpose(1, 2)  # Convert to [batch, seq_len, n_heads, head_dim]
        else:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D")
            
        # Create position embeddings
        seq_len = position_ids.size(-1)
        position_ids = position_ids.view(-1, seq_len)
        
        # Compute frequencies for each position
        freqs = torch.outer(position_ids.float().flatten(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Reshape to match input
        cos = emb.cos().view(batch_size, seq_len, 1, head_dim)
        sin = emb.sin().view(batch_size, seq_len, 1, head_dim)
        
        # Apply rotation
        x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
        rotated = torch.cat([
            x1 * cos[..., :head_dim//2] - x2 * sin[..., :head_dim//2],
            x1 * sin[..., head_dim//2:] + x2 * cos[..., head_dim//2:]
        ], dim=-1)
        
        # Convert back to original shape if needed
        if needs_transpose:
            rotated = rotated.transpose(1, 2)
            
        return rotated


class AttentionModule(Module):
    def __init__(self, config: DynaConfig):
        super().__init__()
        self.pe = RotaryPositionalEmbeddings(
            dim=config.d_head,
            base=10000,
            max_position_embeddings=getattr(config, "max_seq_len", 1024),
        )
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.n_heads = config.n_heads

        self.q_proj = torch.nn.Linear(
            self.d_model, self.n_heads * self.d_head, bias=False
        )
        self.k_proj = torch.nn.Linear(
            self.d_model, self.n_heads * self.d_head, bias=False
        )
        self.v_proj = torch.nn.Linear(
            self.d_model, self.n_heads * self.d_head, bias=False
        )
        self.out_proj = torch.nn.Linear(
            self.n_heads * self.d_head, self.d_model, bias=False
        )
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x, mask, pos_ids):
        batch_size, seq_len = x.size(0), x.size(1)
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape before applying rotary embeddings
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Apply RoPE to q and k
        q = self.pe(q, pos_ids)
        k = self.pe(k, pos_ids)
        
        print(f"q shape after RoPE: {q.shape}")
        print(f"k shape after RoPE: {k.shape}")
        
        # Transpose for attention computation: [batch, n_heads, seq_len, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0
        )
        
        # Reshape back to [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output


class FFNModule(Module):
    def __init__(self, config: DynaConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_ffn = config.d_ffn
        self.fc1 = torch.nn.Linear(self.d_model, self.d_ffn)
        self.fc2 = torch.nn.Linear(self.d_ffn, self.d_model)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Fix: Don't apply dropout twice


@beartype
class SimpleLayer(Module):
    def __init__(self, config: DynaConfig):
        super().__init__()
        self.attention = AttentionModule(config)
        self.ffn = FFNModule(config)
        self.attn_norm = RMSNorm(config.d_model)  # Fix: separate norms
        self.ffn_norm = RMSNorm(config.d_model)   # Fix: separate norms

    def forward(self, x: Float[Tensor, "batch seq d_model"], mask, seq_id):
        # Fix: Proper residual connections with separate norms
        x = x + self.attention(self.attn_norm(x), mask, seq_id)
        x = x + self.ffn(self.ffn_norm(x))
        return x


@beartype
class DynaPretrainedModel(PreTrainedModel):
    """Base class for Dyna pretrained models."""

    config_class = DynaConfig
    base_model_prefix: str = "Dyna"
    is_parallelizable: bool = False
    main_input_name: str = "input_ids"
    load_tf_weights = None


@beartype
class SimpleTransformer(DynaPretrainedModel):
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

        self.layers = ModuleList([SimpleLayer(config) for _ in range(config.n_layers)])

    def forward(self, x, mask, seq_id) -> CausalLMOutputWithPast:

        for idx, layer in enumerate(self.layers):
            x = layer(x, mask, seq_id)

        return CausalLMOutputWithPast(loss=None, logits=x, past_key_values=None)


# Done except reset params
@beartype
class DynaLM(DynaPretrainedModel):
    """MoEUT Language Model with embedding and output layers."""

    def __init__(self, config: DynaConfig, eos_token_id: int):
        super().__init__(config)

        # Core transformer
        self.transformer = SimpleTransformer(config)

        # Model configuration

        self.d_model = config.d_model
        # Input/output layers
        self.embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.out_norm = RMSNorm(config.d_model)
        self.eos_token_id = eos_token_id

    def _generate_causal_mask(
        self, input_ids: Tensor  # [batch, seq]
    ) -> Tensor:  # [batch, seq, seq]
        """Ultra-optimized version using advanced vectorization."""

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

        # Vectorized sequence ID computation for all batches at once
        # Create cumulative EOS count to identify sequence boundaries
        eos_positions = eos_mask.long()  # Convert to int: [batch, seq]

        # Create sequence IDs by cumulative sum of EOS indicators
        # Use a more efficient approach with broadcasting
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
        inputs_embeds: Optional[Float[Tensor, "batch seq d_model"]] = None,
        attention_mask: Optional[Bool[Tensor, "batch seq seq"]] = None,
        src_len_mask: Optional[Int[Tensor, "batch seq"]] = None,
    ) -> CausalLMOutputWithPast:
        """Forward pass through the language model."""

        x = self.embedding(input_ids)

        outputs = self.transformer(x, attention_mask, src_len_mask)
        # Apply output projection
        outputs.logits = self.lm_head(self.out_norm(outputs.logits))

        return outputs


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
        input_ids = batch.get("input_ids")
        
        attention_mask = self.model._generate_causal_mask(input_ids)
        src_len_mask = self.model._generate_source_len_mask(attention_mask)

        return self.model(
            input_ids=input_ids,
            inputs_embeds=batch.get("inputs_embeds", None),
            attention_mask=attention_mask,
            src_len_mask=src_len_mask,
        )

    def loss(self, outputs: CausalLMOutputWithPast, batch) -> torch.Tensor:
        loss = compute_loss_from_logits(
            outputs,
            self.shift_labels,
            batch["labels"],
            self.loss_fn,
        )

        return loss
        return loss
