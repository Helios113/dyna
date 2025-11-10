from dataclasses import asdict

from transformers import PretrainedConfig

from dyna.config.norm_config import NormConfig

from .enums import ExecutionMode, LayerType, NormStructure, RescaleMethod


class DynaConfig(PretrainedConfig):
    """Configuration class for Dyna model."""

    model_type = "dyna"

    def __init__(self, **kwargs):
        """Initialize DynaConfig with default values and optional overrides.

        Args:
            **kwargs: Keyword arguments for configuration parameters.
        """
        super().__init__(**{"model_type": self.model_type})

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
        self.head_size = kwargs.pop("head_size", 0)
        self.tail_size = kwargs.pop("tail_size", 0)
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
        layer_type = kwargs.pop("layer_type", "simple")
        if isinstance(layer_type, str):
            self.layer_type = LayerType[layer_type]
        else:
            self.layer_type = rescaling_method_val
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
        self.sample_iterations = kwargs.pop("sample_iterations", False)
        # Handle execution_mode enum
        execution_mode_val = kwargs.pop("execution_mode", "moe")
        if isinstance(execution_mode_val, str):
            self.execution_mode = ExecutionMode[execution_mode_val]
        else:
            self.execution_mode = execution_mode_val
        self.repeat_residual = kwargs.pop("repeat_residual", False)
        self.nope_pos = kwargs.pop("nope_pos", False)
        self.use_energy_per_sample = kwargs.pop("use_energy_per_sample", False)
        self.use_reg_loss = kwargs.pop("use_reg_loss", False)
        self.use_moe_bias = kwargs.pop("use_moe_bias", True)
        self.use_embedding_norm = kwargs.pop("use_embedding_norm", False)
        self.manual_scale = kwargs.pop("manual_scale", False)
        self.loop_normalization = kwargs.pop("loop_normalization", False)
        self.norms: NormConfig = NormConfig(**kwargs.pop("norms", {}))
        self.loop_rope_theta_rebase = kwargs.pop("loop_rope_theta_rebase", False)
        self.transformer_type = kwargs.pop("transformer_type", "dyna")
        self.loop_hyper_params: bool = kwargs.pop("loop_hyper_params", False)
        self.base_depth: int = kwargs.pop("base_depth", 12)
        self.current_depth: int = kwargs.pop("current_depth", 12)
        self.base_width: int = kwargs.pop("base_width", 12)
        self.current_width: int = kwargs.pop("current_width", 12)
        self.cp_alpha: float = kwargs.pop("cp_alpha", 1.0)

        def to_dict(self):
            output = super().to_dict()
            # Convert NormConfig to dict
            if isinstance(output.get("norm_config"), NormConfig):
                output["norm_config"] = asdict(output["norm_config"])
            return output
