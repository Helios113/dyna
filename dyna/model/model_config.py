from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    tokenizer_name: str = "HuggingFaceTB/SmolLM2-1.7B"
    vocab_size: int = 49152
    max_seq_len: int = 2048
    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    n_experts_ffn: int = 10
    n_experts_attn: int = 2
    d_head: int | None = None
    n_group: int = 2
    k_ffn: int = 8
    k_attn: int = 2
    dropout_expert_ffn: float = 0.0
    dropout_expert_attn: float = 0.0
    d_expert_ffn: int = 128
    dropout: float = 0.0
    reg_entropy: float = 0.01
    reg_entropy_attn: float = 0.001
    attention: str = "SwitchHeadRope"
    shift_labels: bool = True
    scale_add: bool = True
    prot_emb: bool = False
    n_expert_shared_attn: int = 1
    n_expert_shared_ffn: int = 2
    collect_reg_loss: bool = False
    enable_early_exit: bool = True
    use_simple_residual: bool = False
    run_id: str | None = None


@dataclass
class TrainerConfig:
    # Train Dataloader
    train_dataloader_label: str = "train"
    train_subset_num_batches: int = -1
    spin_dataloaders: bool = True

    # Stopping Condition
    max_duration: str | None = None  # Use str for durations

    # Algorithms

    # Engine Pass Registration
    algorithm_passes: str | None = None  # Use str if not OmegaConf-compatible

    # Optimizers and Scheduling

    scale_schedule_ratio: float = 1.0
    step_schedulers_every_batch: bool | None = None

    # Evaluators
    eval_interval: str | None = "100ba"  # Use str for intervals
    eval_subset_num_batches: int = -1
    run_name: str | None = None
    progress_bar: bool = False
    log_to_console: bool = True
    console_stream: str | None = "stderr"
    console_log_interval: str | None = "10ba"
    log_traces: bool = False
    auto_log_hparams: bool = False

    # Load Checkpoint
    load_path: str | None = None
    load_object_store: str | None = None  # Use str if not OmegaConf-compatible
    load_weights_only: bool = False
    load_strict_model_weights: bool = True
    load_progress_bar: bool = True
    load_ignore_keys: list | None = None  # Use list for OmegaConf compatibility
    load_exclude_algorithms: list | None = None

    # Save Checkpoint
    save_folder: str | None = None
    save_filename: str = "ep{epoch}-ba{batch}-rank{rank}.pt"
    save_latest_filename: str | None = "latest-rank{rank}.pt"
    save_overwrite: bool = False
    save_interval: str | None = "1ep"
    save_weights_only: bool = False
    save_ignore_keys: list | None = None
    save_num_checkpoints_to_keep: int = -1
    save_metrics: bool = False

    # Graceful Resumption
    autoresume: bool = False

    # Parallelism
    parallelism_config: dict | None = None

    # System/Numerics
    device: str | None = "meta"
    precision: str | None = None
    precision_config: dict | None = None
    device_train_microbatch_size: str | int | None = "auto"
    accumulate_train_batch_on_tokens: bool = False

    # Reproducibility
    seed: int | None = None
    deterministic_mode: bool = False

    # Distributed Training
    dist_timeout: float = 300.0
    ddp_sync_strategy: str | None = None

    # Profiling
    profiler: str | None = None  # Use str if not OmegaConf-compatible

    # Python logging
    python_log_level: str | None = None

    # compile config for PyTorch 2.0 or higher
    compile_config: dict | None = None
    # Python logging
    python_log_level: str | None = None

    # compile config for PyTorch 2.0 or higher
    compile_config: dict | None = None


@dataclass
class DatasetConfig:
    streams: dict | None = None
    token_encoding_type: str = "int64"
    remote: str | None = None
    local: str | None = None
    split: str | None = None
    download_retry: int = 2
    download_timeout: float = 60
    validate_hash: str | None = None
    keep_zip: bool = False
    epoch_size: int | str | None = None
    predownload: int | None = None
    cache_limit: int | str | None = None
    partition_algo: str = "relaxed"
    num_canonical_nodes: int | None = None
    shuffle: bool = False
    shuffle_algo: str = "py1e"
    shuffle_seed: int = 9176
    shuffle_block_size: int | None = None
    sampling_method: str = "balanced"
    sampling_granularity: int = 1
    batching_method: str = "random"
    allow_unsafe_types: bool = False
    replication: int | None = None
    stream_name: str = "stream"
    stream_config: dict | None = None
    max_seq_len: int | None = 2048


@dataclass
class DataConfig:
    path: str
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    name: str = "text"
    drop_last: bool = True
    num_workers: int = 4
    device_batch_size: int = 512


@dataclass
class SchedulerConfig:
    name: str = "wsld"
    t_warmup: str | None = "1ba"
    t_max: str | None = "1dur"
    t_cooldown: str | None = None
    alpha_f: float = 0.0
    scale_warmup: bool = False
    scale_cooldown: bool = False
