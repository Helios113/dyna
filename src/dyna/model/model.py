from typing import cast

import torch
from composer.models.huggingface import HuggingFaceModel

# from composer.callbacks
# Add jaxtyping imports
from jaxtyping import Bool, Float, Int
from llmfoundry.models.layers.layer_builders import build_norm
from llmfoundry.utils.builders import build_metric
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)

from dyna.config import (
    CROSS_ENTROPY_IGNORE_INDEX,
    DEFAULT_CAUSAL_LM_TRAIN_METRICS,
    PROT_EMB_RESCALING_METHODS,
    DynaConfig,
)
from dyna.model.base import DynaPretrainedModel
from dyna.model.pass_through import PassThroughTransformer

# Import directly from specific modules to avoid circular imports
from dyna.model.transformer import DynaFormer
from dyna.modules import LayerModule


def _generate_attention_mask(
    input_ids: Int[Tensor, "batch seq"], eos_token_id: int
) -> Bool[Tensor, "batch 1 seq seq"]:
    """Generate attention mask for each sample in batch."""
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Create base causal mask
    base_causal = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )

    # Find EOS positions
    eos_mask = input_ids == eos_token_id  # [batch, seq]

    # Quick path -- if no EOS tokens in entire batch
    if not eos_mask.any():
        return base_causal.unsqueeze(0).expand(batch_size, -1, -1)

    # Find EOS positions
    eos_positions: Int[Tensor, "batch seq"] = (
        eos_mask.long()
    )  # Convert to int: [batch, seq]

    # Calculate sequence IDs
    sequence_ids: Int[Tensor, "batch seq"] = torch.cumsum(
        torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                eos_positions[:, :-1],
            ],
            dim=1,
        ),
        dim=1,
    )

    # Vectorized same-sequence mask computation
    # Use broadcasting instead of unsqueeze for better memory efficiency
    same_seq_mask: Bool[Tensor, "batch seq seq"] = (
        sequence_ids[:, :, None] == sequence_ids[:, None, :]
    )

    # Apply masks
    final_mask = base_causal[None, :, :] & same_seq_mask
    return final_mask.unsqueeze(1)


def _generate_source_len_mask(
    attention_mask: Bool[Tensor, "batch 1 seq seq"],
) -> Int[Tensor, "batch seq"]:
    """Generate source length mask with position indices for each sequence."""
    batch_size, _, seq_len, _ = attention_mask.shape
    device = attention_mask.device

    pos_range = torch.arange(seq_len, device=device, dtype=torch.long)

    causal_indices = pos_range.unsqueeze(0) <= pos_range.unsqueeze(1)

    # Expand to batch dimension
    causal_indices: Bool[Tensor, "batch seq seq"] = causal_indices.unsqueeze(0).expand(
        batch_size, -1, -1
    )

    # Apply the attention mask to get valid positions
    valid_positions: Bool[Tensor, "batch seq seq"] = (
        attention_mask.squeeze(1) & causal_indices
    )

    position_mask: Int[Tensor, "batch seq"] = valid_positions.sum(dim=-1) - 1
    return position_mask


TRANSFORMER_CLASSES = {
    "dyna": DynaFormer,
    "pass_through": PassThroughTransformer,
}


class DynaLM(DynaPretrainedModel):
    """Base Language model class.

    DynaLM contains a transformer core and input/output layers.
    """

    def __init__(self, config: DynaConfig, eos_token_id: int):
        """Initialize DynaLM model.

        Args:
            config (DynaConfig): Configuration object.
            eos_token_id (int): End-of-sentence token ID. This is passed so that we can
                                construct the base causal mask.
        """
        super().__init__(config)

        # Core transformer
        self.transformer = TRANSFORMER_CLASSES[config.transformer_type](config)

        # Model configuration
        self.n_repeats = config.n_repeats
        self.n_layers = config.n_repeats

        self.d_model = config.d_model
        self.eos_token_id = eos_token_id
        self.rescaling_method = config.rescaling_method
        self.use_energy_per_sample = config.use_energy_per_sample
        self.use_reg_loss = config.use_reg_loss

        # Input/output layers
        self.embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head._fsdp_wrap = True  # pyright: ignore[reportArgumentType]
        if config.use_embedding_norm:
            self.embedding_norm = build_norm(
                name=config.norms.norm_type,
                eps=config.norms.ffn_eps,
                normalized_shape=config.d_model,
            )
        else:
            self.embedding_norm = None
        self.out_norm = build_norm(
            name=config.norms.norm_type,
            eps=config.norms.ffn_eps,
            normalized_shape=config.d_model,
        )
        self.lm_head_scale = (config.current_width / config.base_width) ** (-1)
        # Provide LM head to transformer for entropy computation
        # The LM head cannot be used with no grad as
        # all gradients for that step are discarded
        self.transformer._temp_lm_head = lambda x: self.lm_head_scale * self.lm_head(
            self.out_norm(x)
        )
        self.head_size = config.head_size
        self.tail_size = config.tail_size
        self.n_layers = config.n_layers
        self.init_sigma = config.init_sigma

    def reset_parameters(self):
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=self.init_sigma)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.init_sigma)

        self.transformer.reset_parameters()

    def forward(
        self,
        input_ids: Int[Tensor, "batch seq"] | None = None,
        labels: Int[Tensor, "batch seq"] | None = None,
        inputs_embeds: Float[Tensor, "batch seq d_model"] | None = None,
        attention_mask: Bool[Tensor, "batch seq seq"] | None = None,
        src_len_mask: Int[Tensor, "batch seq"] | None = None,
    ) -> CausalLMOutputWithPast:
        """Forward pass through the language model."""
        # Validate inputs
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")

        # Get embeddings and subsequenent masks
        x, attention_mask, src_len_mask = self.embedding_stage(
            input_ids, inputs_embeds, attention_mask, src_len_mask
        )

        assert attention_mask is not None
        assert src_len_mask is not None

        # Prepare protected embeddings if enabled
        if self.rescaling_method in PROT_EMB_RESCALING_METHODS:
            e = x.clone()
            x = torch.zeros_like(x)
        else:
            e = None

        # Run the transformer model
        x, energy_per_sample = self.transformer(
            x, attention_mask=attention_mask, sequence_length=src_len_mask, e=e
        )

        # Apply output projection
        logits = self.lm_head_scale * self.lm_head(self.out_norm(x))

        # Calculate the loss
        loss = None
        if labels is not None:
            _labels = torch.roll(labels, shifts=-1)
            _labels[:, -1] = CROSS_ENTROPY_IGNORE_INDEX
            losses = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                _labels.to(logits.device).view(-1),
                reduction="none",
            )
            loss = losses.flatten()

            # energy per sample for early exit
            if self.use_energy_per_sample:
                loss = loss * energy_per_sample.flatten()

            # Reduce the loss according to the correct tokens
            if torch.all(_labels == CROSS_ENTROPY_IGNORE_INDEX):  # type: ignore
                loss = loss.sum()
            else:
                loss = loss.sum() / (_labels != CROSS_ENTROPY_IGNORE_INDEX).sum()  # type: ignore

            # Reg loss for moeut
            if self.use_reg_loss:
                loss = loss + self.transformer._collect_regularization_loss()
            else:
                self.transformer._clear_selection_history()

        return CausalLMOutputWithPast(
            loss=loss,  # pyright: ignore[reportArgumentType]
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def embedding_stage(self, input_ids, inputs_embeds, attention_mask, src_len_mask):
        if input_ids is not None:
            x = self.embedding(input_ids)
            if attention_mask is None:
                attention_mask = _generate_attention_mask(input_ids, self.eos_token_id)
            if src_len_mask is None:
                src_len_mask = _generate_source_len_mask(attention_mask)
        if self.embedding_norm is not None:
            x = self.embedding_norm(x)
        elif isinstance(inputs_embeds, torch.Tensor):
            x = inputs_embeds
        return x, attention_mask, src_len_mask

    @staticmethod
    def fsdp_wrap_fn(module: Module) -> bool:
        """Determines whether a module should be wrapped with FSDP."""
        if hasattr(module, "_fsdp_kwargs_dict"):
            return bool(module._fsdp_kwargs_dict)
        print(
            "fsdp_wrap_fn called to %s",
            module,
            isinstance(module, LayerModule),
            flush=True,
        )
        return isinstance(module, LayerModule)


class ComposerDynaModel(HuggingFaceModel):
    """Composer-compatible language model wrapper."""

    model: DynaLM

    def __init__(
        self,
        config: DynaConfig,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ):
        """Initialize the ComposerDynaModel.

        Args:
            config (DynaConfig): Configuration for the model.
            tokenizer (PreTrainedTokenizerBase): Tokenizer for the model.
        """
        # Configuration
        self.vocab_size = config.vocab_size
        self.shift_labels = config.shift_labels

        # Metrics
        train_metrics = [
            build_metric(metric, {}) for metric in DEFAULT_CAUSAL_LM_TRAIN_METRICS
        ]

        super().__init__(
            model=DynaLM(config, cast(int, tokenizer.eos_token_id)),
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=None,
            shift_labels=config.shift_labels,
            allow_embedding_resizing=True,
        )

        self.model.reset_parameters()

    def forward(self, batch) -> CausalLMOutputWithPast:
        return self.model(
            input_ids=batch.get("input_ids", None),
            labels=batch.get("labels", None),
            inputs_embeds=batch.get("inputs_embeds", None),
            attention_mask=batch.get("attention_mask", None),
        )

    def loss(self, outputs: CausalLMOutputWithPast, batch) -> torch.Tensor:
        labels = batch["labels"]
        logits: torch.Tensor = cast(torch.Tensor, outputs.logits)
        _labels = torch.roll(labels, shifts=-1)
        _labels[:, -1] = CROSS_ENTROPY_IGNORE_INDEX
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            _labels.to(logits.device).view(-1),
        )
        # I want to see the gradient pathway to the q weights
        # loss.backward(retain_graph=True)
        # print(self.model.transformer.attention_modules[0].q_linear.weight.grad)

        return loss
