from dataclasses import dataclass

from dyna.config.enums import ExecutionMode, NormStructure, NormType, RescaleMethod


@dataclass
class ModelConfig:
    tokenizer_name: str = "HuggingFaceTB/SmolLM2-1.7B"
    execution_mode: ExecutionMode = ExecutionMode.moe
    vocab_size: int = 49152
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
    rescaling_method: RescaleMethod = RescaleMethod.cum_avg_prot_emb
    norm_structure: NormStructure = NormStructure.peri
    norm_type: NormType = NormType.rmsnorm
    run_id: str | None = None
    sample_iterations: bool = False
    repeat_residual: bool = False
    perfiery_size: int = 2
