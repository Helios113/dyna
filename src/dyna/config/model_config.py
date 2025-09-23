
class DynaConfig(PretrainedConfig):
    """Configuration class for Dyna model."""

    model_type = "dyna"

    def __init__(self, **kwargs):
        super().__init__(**{"model_type": self.model_type})

        # Import required for enum handling
        from .modules.model_config import (
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
        self.perfiery_size = kwargs.pop("perfiery_size", 2)
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
        self.sample_iterations = kwargs.pop("sample_iterations", False)
        # Handle execution_mode enum
        execution_mode_val = kwargs.pop("execution_mode", "moe")
        if isinstance(execution_mode_val, str):
            self.execution_mode = ExecutionMode[execution_mode_val]
        else:
            self.execution_mode = execution_mode_val
        self.repeat_residual = kwargs.pop("repeat_residual", False)



