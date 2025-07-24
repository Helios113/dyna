class ComposerMPTCausalLM(HuggingFaceModel):

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        use_train_metrics: Optional[bool] = True,
        additional_train_metrics: Optional[list] = None,
        loss_fn: Optional[Union[str, dict]] = 'fused_crossentropy',
        **kwargs: dict[str, Any],
    ):
        from llmfoundry.metrics import (
            DEFAULT_CAUSAL_LM_EVAL_METRICS,
            DEFAULT_CAUSAL_LM_TRAIN_METRICS,
        )
        from llmfoundry.utils.builders import build_metric

        additional_train_metrics = additional_train_metrics or []

        model = self.model_class(self.config_class(**kwargs))

        use_train_metrics = use_train_metrics
        train_metric_names = DEFAULT_CAUSAL_LM_TRAIN_METRICS + additional_train_metrics
        train_metrics = [
            build_metric(metric, {}) for metric in train_metric_names
        ] if use_train_metrics else []
        eval_metric_names = DEFAULT_CAUSAL_LM_EVAL_METRICS + additional_train_metrics
        eval_metrics = [
            build_metric(metric, {}) for metric in eval_metric_names
        ]

        super().__init__(
            model=model,
            tokenizer=tokenizer,  # type: ignore
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=model.transformer.shift_labels,
            allow_embedding_resizing=True,
        )

        loss_fn_config = loss_fn
        if loss_fn_config == 'fused_crossentropy':
            try:
                from flash_attn.losses.cross_entropy import \
                    CrossEntropyLoss as FusedCrossEntropyLoss

                self.loss_fn = FusedCrossEntropyLoss(
                    ignore_index=CROSS_ENTROPY_IGNORE_INDEX,
                    reduction='none',
                )
            except:
                raise ValueError(
                    'Fused Cross Entropy is not installed. Either (1) have a CUDA-compatible GPU '
                    +
                    'and `pip install .[gpu]` if installing from source or `pip install xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v1.0.3#subdirectory=csrc/xentropy` '
                    +
                    'if installing from pypi, or (2) set your config model.loss_fn=torch_crossentropy.',
                )
        elif loss_fn_config == 'torch_crossentropy':
            self.loss_fn = nn.CrossEntropyLoss(
                ignore_index=CROSS_ENTROPY_IGNORE_INDEX,
                reduction='none',
            )
        else:
            raise ValueError(
                f'Specified loss_fn={self.loss_fn} not recognized. `loss_fn` must be one of [`fused_crossentropy`, `torch_crossentropy`].',
            )

    @property
    def model_class(self) -> type[MPTForCausalLM]:
        return MPTForCausalLM

    @property
    def config_class(self) -> type[MPTConfig]:
        return MPTConfig

    def get_targets(self, batch: Mapping) -> torch.Tensor:
        return get_targets(batch['labels'])

    def forward(self, batch: MutableMapping) -> CausalLMOutputWithPast:
        if self.config.ffn_config['ffn_type'] in ffns_with_megablocks:
            # Clear MegaBlocks MoE load balancing loss cache
            try:  # Add try/catch to avoid transformers complaining and raising errors
                from megablocks.layers.moe import clear_load_balancing_loss
            except:
                raise RuntimeError(
                    'Requirements for MegaBlocks not installed; see install instructions in `README.md`.',
                )
            clear_load_balancing_loss()
        return self.model(
            input_ids=batch.get('input_ids', None),
            attention_mask=batch.get('attention_mask', None),
            sequence_id=batch.get('sequence_id', None),
            inputs_embeds=batch.get('inputs_embeds', None),
            position_ids=batch.get('position_ids', None),
        )

    def loss(self, outputs: CausalLMOutputWithPast,
             batch: Mapping) -> Union[dict, torch.Tensor]:
        loss = compute_loss_from_logits(
            outputs,
            self.shift_labels,
            batch['labels'],
            self.loss_fn,
        )

        if self.config.ffn_config['ffn_type'] in ffns_with_megablocks:
            # MegaBlocks MoE load balancing loss
            try:  # Add try/catch to avoid transformers complaining and raising errors
                from megablocks.layers.moe import batched_load_balancing_loss
            except:
                raise RuntimeError(
                    'Requirements for MegaBlocks not installed; see install instructions in `README.md`.',
                )
            lbl = batched_load_balancing_loss(
                self.model.transformer.mb_args,  # type: ignore
            )  # type: ignore
            return {
                'total': loss + lbl,
                'loss': loss,
                'lbl': lbl,
            }
        return loss

    @cached_property
    def n_total_params(self):
        """Gets the total number of parameters in the model."""
        return mpt_get_total_params(self)

    @cached_property
    def n_active_params(self):
        """Gets the total number of active parameters in the model."""
        return mpt_get_active_params(self)

    def flops_per_batch(self, batch: Mapping):
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass

        if self.model.config.block_overrides is not None:
            warnings.warn(
                'Warning, flop computation is not supported when using block overrides. Returning 0 flops per batch.',
            )
            return 0

        bs, msl = batch['input_ids'].shape[0:2]
        params = self.n_active_params
        params_flops_per_token = 2 * params
        params_flops_per_seq = params_flops_per_token * msl
        attn_flops_per_seq = self.get_attention_flops(msl)
        return (params_flops_per_seq + attn_flops_per_seq) * 3 * bs

    def get_attention_flops(self, msl: int) -> int:
        """Computes the attention flops for the batch.

        Args:
            msl (int): The batch sequence length.

        Returns:
            attn_flops (int): The attention flops.
        """
        return (
            self.model.config.n_layers * 2 * 2 *
            (self.model.config.d_model * (msl**2))
        )