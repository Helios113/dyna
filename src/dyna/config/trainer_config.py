from dataclasses import dataclass


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
    console_log_interval: str | None = "50ba"
    log_traces: bool = False
    auto_log_hparams: bool = False

    # Load Checkpoint
    load_path: str | None = None
    load_object_store: str | None = None  # Use str if not OmegaConf-compatible
    load_weights_only: bool = False
    load_strict_model_weights: bool = False
    load_progress_bar: bool = True
    load_ignore_keys: list | None = None  # Use list for OmegaConf compatibility
    load_exclude_algorithms: list | None = None

    # Save Checkpoint
    save_folder: str | None = None
    save_filename: str = "new_ep{epoch}-ba{batch}-rank{rank}.pt"
    save_latest_filename: str | None = "latest-rank{rank}.pt"
    save_overwrite: bool = True
    save_interval: str | None = "1ep"
    save_weights_only: bool = False
    save_ignore_keys: list | None = None
    save_num_checkpoints_to_keep: int = 0
    save_metrics: bool = False

    # Graceful Resumption
    autoresume: bool = False

    # System/Numerics
    device: str | None = "gpu"
    precision: str | None = None
    precision_config: dict | None = None
    device_train_microbatch_size: str | int | None = "auto"
    accumulate_train_batch_on_tokens: bool = False

    # Reproducibility
    seed: int | None = 42
    deterministic_mode: bool = False

    # Distributed Training
    dist_timeout: float = 300.0
    ddp_sync_strategy: str | None = None

    # Profiling
    # profiler: str | None = None  # Use str if not OmegaConf-compatible

    # Python logging
    python_log_level: str | None = None

    # compile config for PyTorch 2.0 or higher
    compile_config: dict | None = None
