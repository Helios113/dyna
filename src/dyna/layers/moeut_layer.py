
class MoEUTLayer(LayerModule):
    """Single layer of the MoEUT model with configurable behavior."""

    def __init__(self, config: DynaConfig, input_reinjection: bool = False):
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
            input_projection=(
                torch.nn.Linear(2 * config.d_model, config.d_model, bias=False)
                if input_reinjection
                else None
            ),
        )
        self.input_reinjection = input_reinjection
        if config.enable_early_exit:
            self.saturation_detector = SaturationGate(config.d_model)
        else:
            self.saturation_detector = None

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        layer_index: int,
        e: None | Float[Tensor, "batch seq d_model"],
        reinjection_embeddings: None | Float[Tensor, "batch seq d_model"],
        router: Float[Tensor, "d_model"],
        cum_sum: Float[Tensor, "batch seq"],
        tau: Float[Tensor, "1"],
        mask: tuple[Bool[Tensor, "batch seq seq"], Int[Tensor, "batch seq"]],
        total_layers: int,
        continue_mask: None | Int[Tensor, "size"] = None,
    ) -> tuple[
        Float[Tensor, "batch seq d_model"],
        tuple,
        Float[Tensor, "batch seq"] | None,
    ]:
        """Forward pass through the layer with configurable behavior."""

        if self.input_reinjection and reinjection_embeddings is not None:
            x = torch.cat((x, reinjection_embeddings), dim=-1)
            x = self.input_projection(x)

        q_val, k_val, v_val = self._apply_pre_norm_attn(x)

        att_out, expert_sel_attn = self.attention(
            q_val,
            k_val,
            v_val,
            mask,
        )

        x = self._apply_update_to_residual(
            x,
            att_out,
            continue_mask,
            layer_index,
            self.attn_post,
            e,
        )

        ffn_out, expert_sel_ffn = self.ffn(*self._apply_pre_norm_ffn(x))

        saturation_event = None
        if self.saturation_detector is not None:
            saturation_event = self.saturation_detector(ffn_out)

        x = self._apply_update_to_residual(
            x,
            ffn_out,
            continue_mask,
            layer_index,
            self.ffn_post,
            e,
        )

        return (x, (expert_sel_attn, expert_sel_ffn), saturation_event)