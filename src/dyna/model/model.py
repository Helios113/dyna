import torch
from beartype import beartype
from composer.models.huggingface import HuggingFaceModel

# from composer.callbacks
# Add jaxtyping imports
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.normalization import RMSNorm
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel

from dyna.config import (
    CROSS_ENTROPY_IGNORE_INDEX,
    DEFAULT_CAUSAL_LM_TRAIN_METRICS,
    PROT_EMB_RESCALING_METHODS,
    ModelConfig,
)
from dyna.model.transformer import DynaFormer
from dyna.modules import LayerModule


class DynaPretrainedModel(PreTrainedModel):
    """Base class for Dyna pretrained models."""

    config_class = ModelConfig  # type: ignore[reportGeneralTypeIssues]
    base_model_prefix: str = "Dyna"
    is_parallelizable: bool = False
    main_input_name: str = "input_ids"
    load_tf_weights = None
    _no_split_modules = [LayerModule]  # type: ignore[reportGeneralTypeIssues]


class DynaLM(DynaPretrainedModel):
    """MoEUT Language Model with embedding and output layers."""

    def __init__(self, config: ModelConfig, eos_token_id: int):
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

        self.lm_head._fsdp_wrap = True

        # Output normalization - configurable type
        norm_class = RMSNorm if config.use_rms_norm else torch.nn.LayerNorm
        self.out_norm = norm_class(config.d_model)
        self.eos_token_id = eos_token_id

        self.rescaling_method = config.rescaling_method
        # Initialize parameters
        self.sample_iterations = config.sample_iterations
        # Provide LM head to transformer for entropy computation
        self.transformer._temp_lm_head = lambda x: self.lm_head(self.out_norm(x))

    @torch.no_grad
    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(
            self.embedding.weight, mode="fan_in", nonlinearity="linear"
        )
        self.transformer.reset_parameters()

    def _generate_causal_mask(
        self,
        input_ids: Tensor,  # [batch, seq]
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

        x, energy_per_sample = self.transformer(x, e, (attention_mask, src_len_mask))

        # Apply output projection
        logits = self.lm_head(self.out_norm(x))
        # print("Expected: lm_head, true:",logits)
        loss = None
        if labels is not None:
            _labels = torch.roll(labels, shifts=-1)
            _labels[:, -1] = CROSS_ENTROPY_IGNORE_INDEX
            losses = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                _labels.to(logits.device).view(-1),
                reduction="none",
            )
            loss = losses.flatten() * energy_per_sample.flatten()
            # Reduce the loss according to the correct tokens
            if torch.all(_labels == CROSS_ENTROPY_IGNORE_INDEX):  # type: ignore
                loss = loss.sum()
            else:
                loss = loss.sum() / (_labels != CROSS_ENTROPY_IGNORE_INDEX).sum()  # type: ignore

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    # FSDP Wrap function

    @staticmethod
    def fsdp_wrap_fn(module: Module) -> bool:
        if hasattr(module, "_fsdp_kwargs_dict"):
            return bool(module._fsdp_kwargs_dict)
        print(
            "fsdp_wrap_fn called to %s",
            module,
            isinstance(module, LayerModule),
            flush=True,
        )
        return isinstance(module, LayerModule)


# Done
@beartype
class ComposerDynaModel(HuggingFaceModel):
    """Composer-compatible MoEUT model wrapper."""

    def __init__(
        self,
        config: ModelConfig,
        tokenizer: PreTrainedTokenizerBase,
    ):
        # Setup distributed cleanup
        setup_distributed_cleanup()

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
