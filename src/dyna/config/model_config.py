from dataclasses import dataclass, field

from dyna.config.enums import (
    ExecutionMode,
    LayerType,
    NormStructure,
    RescaleMethod,
    TransformerType,
)
from dyna.config.norm_config import NormConfig


@dataclass
class ModelConfig:
    tokenizer_name: str = "HuggingFaceTB/SmolLM2-1.7B"
    execution_mode: ExecutionMode = ExecutionMode.moe
    transformer_type: TransformerType = TransformerType.dyna
    vocab_size: int = 49152
    init_sigma: float = 0.02
    max_seq_len: int = 2048
    d_model: int = 1024
    d_ffn: int = 4096
    n_repeats: int = 12
    n_heads: int = 12
    n_experts_ffn: int = 10
    n_experts_attn: int = 2
    d_head: int | None = None
    n_layers: int = 2
    k_ffn: int = 8
    k_attn: int = 2
    dropout_expert_ffn: float = 0.0
    dropout_expert_attn: float = 0.0
    d_expert_ffn: int = 128
    dropout: float = 0.0
    reg_entropy: float = 0.01
    reg_entropy_attn: float = 0.001
    shift_labels: bool = True
    n_expert_shared_attn: int = 1
    n_expert_shared_ffn: int = 2
    collect_reg_loss: bool = False
    enable_early_exit: bool = True
    rescaling_method: RescaleMethod = RescaleMethod.none
    norm_structure: NormStructure = NormStructure.peri
    norms: NormConfig = field(default_factory=NormConfig)
    layer_type: LayerType = LayerType.simple
    run_id: str | None = None
    sample_iterations: bool = False
    repeat_residual: bool = False
    nope_pos: bool = False
    use_reg_loss: bool = False
    use_embedding_norm: bool = False
    sqrt_attention_scale: bool = False
    scale_qk: bool = False
    tail_size: int = 2
    head_size: int = 2
    loop_normalization: bool = False
    loop_rope_theta_rebase: bool = False

    # old name
    # total_depth_for_init: int = 12

    # new names
    # names for
    base_depth: int = 12
    current_depth: int = 12
    base_width: int = 12
    current_width: int = 12
    loop_hyper_params: bool = False
    cp_alpha: float = 1.0
