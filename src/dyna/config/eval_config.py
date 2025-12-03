"""Configuration classes for model evaluation."""

from dataclasses import dataclass, field


@dataclass
class ICLTaskConfig:
    """Configuration for an ICL evaluation task."""

    label: str
    dataset_uri: str
    num_fewshot: int = field(default_factory=lambda: 0)
    icl_task_type: str = "language_modeling"
    metric_names: list[str] = field(
        default_factory=lambda: ["InContextLearningLMAccuracy"]
    )
    prompt_string: str = ""
    example_delimiter: str = "\n"
    continuation_delimiter: str = " "
    max_seq_len: int = 1024
    batch_size: int = 8


@dataclass
class EvalDatasetConfig:
    """Configuration for evaluation dataset."""

    local: str
    remote: str = ""
    split: str = "train"


@dataclass
class EvalLoaderConfig:
    """Configuration for evaluation data loader."""

    name: str = "text"
    dataset: EvalDatasetConfig = field(
        default_factory=lambda: EvalDatasetConfig(local="")
    )
    drop_last: bool = False
    num_workers: int = 8


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer."""

    name: str = "HuggingFaceTB/SmolLM2-1.7B"
    kwargs: dict = field(default_factory=lambda: {"padding_side": "left"})


@dataclass
class EvalConfig:
    """Main evaluation configuration."""

    run_name: str
    # Model checkpoint
    load_weights_only: bool = True

    # Device and precision
    device: str = "gpu"
    precision: str = "amp_bf16"
    seed: int = 42

    # Batch sizes
    device_eval_batch_size: int = 8  # For ICL evaluation
    eval_batch_size: int = 1024  # Total batch size for streaming perplexity
    eval_microbatch_size: int = 32  # Per-device microbatch size for perplexity

    # Sequence length
    icl_seq_len: int = 1024
    max_seq_len: int = 1024

    # ICL tasks
    icl_tasks: list[ICLTaskConfig] = field(default_factory=list)

    # Eval loader (for single-batch evaluation)
    eval_loader: EvalLoaderConfig | None = None

    # Tokenizer
    tokenizer: str = ""

    # Optional settings
    max_duration: str | None = None
    dist_timeout: int = 600
    num_retries: int = 0
