import csv
import glob
import os
import secrets
import string
import subprocess
import time
from pathlib import Path
from typing import cast

import yaml
from composer import DataSpec
from composer.core import Callback
from composer.optim.scheduler import ComposerScheduler
from llmfoundry.utils.builders import build_callback, build_dataloader, build_scheduler
from omegaconf import DictConfig, OmegaConf
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from dyna.config import (
    DataConfig,
    FSDPConfig,
    ModelConfig,
    SchedulerConfig,
    TrainerConfig,
)


def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits."""
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def make_wandb_run_name(
    model_config: DictConfig,
    trainer_config: DictConfig,
    unique: str,
    metadata_dir: str = "wandb_metadata",
) -> str:
    """Generate a unique and informative wandb run name from config.

    Format: <index>__<run_name>__<timestamp>__<unique_id>

    Uses double underscores (__) as delimiters to separate major sections,
    allowing single underscores in run names.

    All config parameters are saved to a CSV file for reference instead of
    being embedded in the run name, keeping names short and readable.

    Args:
        model_config: Model configuration
        trainer_config: Trainer configuration
        unique: Unique identifier for this run
        metadata_dir: Directory to save parameter CSV files

    Returns:
        Formatted run name string
    """
    load_path = trainer_config.get("load_path")

    if load_path is not None:
        # Resume from checkpoint: parse existing name and update unique ID
        name = os.path.basename(load_path)
        parts = name.split("__")

        # Replace the unique ID (4th part after index, run_name, timestamp)
        if len(parts) >= 4:
            parts[3] = unique
            name = "__".join(parts)
        else:
            raise ValueError(f"Invalid checkpoint name format: {name}")
    else:
        # Create new run name
        run_name = trainer_config.get("run_name")
        if not run_name:
            run_name = generate_id(8)

        # Timestamp
        timestamp = time.strftime("%d%b%y").lower()

        # Compose short name: run_name__timestamp__unique_id
        name = f"{run_name}__{timestamp}__{unique}"

    # Add or increment index prefix
    name = _add_index_prefix(name)

    # Save all parameters to CSV for reference
    _save_parameters_to_csv(name, model_config, trainer_config, metadata_dir)

    return name


def _save_parameters_to_csv(
    run_name: str,
    model_config: DictConfig,
    trainer_config: DictConfig,
    metadata_dir: str,
) -> None:
    """Save all config parameters to a CSV file.

    Creates a CSV with columns: run_name, parameter, value, source
    This allows easy tracking and comparison of runs without cluttering the run name.

    Args:
        run_name: The generated run name
        model_config: Model configuration
        trainer_config: Trainer configuration
        metadata_dir: Directory to save the CSV file
    """
    # Create metadata directory if it doesn't exist
    metadata_path = Path(metadata_dir)
    metadata_path.mkdir(parents=True, exist_ok=True)

    # CSV file path
    csv_file = metadata_path / "run_parameters.csv"

    # Check if file exists to determine if we need to write headers
    file_exists = csv_file.exists()

    # Collect all parameters
    rows = []

    # Add model config parameters
    for key, value in model_config.items():
        rows.append(
            {
                "run_name": run_name,
                "parameter": key,
                "value": str(value),
                "source": "model_config",
            }
        )

    # Add trainer config parameters
    for key, value in trainer_config.items():
        rows.append(
            {
                "run_name": run_name,
                "parameter": key,
                "value": str(value),
                "source": "trainer_config",
            }
        )

    # Append to CSV
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run_name", "parameter", "value", "source"]
        )

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        # Write all rows
        writer.writerows(rows)

    print(f"Parameters saved to: {csv_file}")


def _add_index_prefix(name: str) -> str:
    """Add or increment the run index prefix.

    If name starts with a number followed by double underscore (e.g., "1__..."), increment it.
    Otherwise, add "1__" prefix.

    Args:
        name: The run name (may or may not have index prefix)

    Returns:
        Name with index prefix
    """
    parts = name.split("__", 1)

    if len(parts) > 1 and parts[0].isdigit():
        # Has existing index, increment it
        index = int(parts[0]) + 1
        return f"{index}__{parts[1]}"
    else:
        # No index, add "1__" prefix
        return f"1__{name}"


def get_callbacks(cfg: DictConfig) -> list[Callback]:
    return [
        build_callback(
            name=str(name),
            kwargs=callback_cfg,
            train_config=cfg,
        )
        for name, callback_cfg in cfg.items()
    ]


def load_and_concat_yamls(directory):
    """Reads all YAML files in a directory.

    loads them, and merges them into a single
    dict.Returns an OmegaConf DictConfig.
    """
    merged = {}
    for file in sorted(
        glob.glob(os.path.join(directory, "*.yaml"))
        + glob.glob(os.path.join(directory, "*.yml"))
    ):
        with open(file) as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                merged.update(data)
            else:
                raise ValueError(f"YAML file {file} does not contain a dict.")
    return OmegaConf.create(merged)


def get_data_loader(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    device_train_batch_size: int,
) -> DataSpec:
    os.environ["S3_ENDPOINT_URL"] = "http://128.232.115.19:9000"
    train_loader = build_dataloader(
        cast(dict[str, object], OmegaConf.to_container(cfg)),
        tokenizer,
        device_train_batch_size,
    )
    return train_loader


def get_scheduler(cfg: DictConfig) -> ComposerScheduler:
    scheduler_name = cfg.name
    del cfg.name
    scheduler = build_scheduler(
        name=scheduler_name,
        scheduler_config=cast(dict[str, object], OmegaConf.to_container(cfg)),
    )
    return scheduler


def check_duplicate_keys(cfg, value_map=None, exceptions=None, path=""):
    """Traverse every key in the config.

    (recursively, regardless of path) and add its
    value to a flat dictionary.

    If a key is seen again, check if the value matches all previous values; if not,
    raise ValueError.
    Allows duplicate keys for those listed in `exceptions` (full path or key).
    Prints the full path of the key and the previous path when a conflict is found.
    """
    if value_map is None:
        value_map = {}
    if exceptions is None:
        exceptions = ["remote", "local", "dataset.split", "proportion"]
    if isinstance(cfg, dict) or hasattr(cfg, "keys"):
        for k in cfg:
            v = cfg[k]
            full_path = f"{path}.{k}" if path else k
            excepted = (k in exceptions) or (full_path in exceptions)
            if not excepted:
                if k in value_map:
                    prev_val, prev_path = value_map[k]
                    if prev_val != v:
                        print(f"Conflict at: {full_path} and {prev_path}")
                        raise ValueError(
                            f"""Duplicate key '{k}' with different values: {prev_val}
                            (at {prev_path}) vs {v} (at {full_path})"""
                        )
                else:
                    value_map[k] = (v, full_path)
            # Recurse into containers
            if isinstance(v, dict) or hasattr(v, "keys"):
                check_duplicate_keys(v, value_map, exceptions, full_path)
            elif isinstance(v, list):
                for idx, item in enumerate(v):
                    check_duplicate_keys(
                        item, value_map, exceptions, f"{full_path}[{idx}]"
                    )
    elif isinstance(cfg, list):
        for idx, item in enumerate(cfg):
            check_duplicate_keys(item, value_map, exceptions, f"{path}[{idx}]")
    return value_map


def build_full_concrete_config(cfg: DictConfig):
    """Constructs and merges all configs.

    (model, trainer, data) and returns a single config dict.
    """
    OmegaConf.resolve(cfg)
    # Model Config
    model_schema = OmegaConf.structured(ModelConfig)
    model_config = OmegaConf.merge(model_schema, cfg.model_config)

    # Trainer Config
    trainer_schema = OmegaConf.structured(TrainerConfig)
    trainer_config = OmegaConf.merge(trainer_schema, cfg.trainer_config)

    # Data Config (including streams)
    data_schema = OmegaConf.structured(DataConfig)
    data_config = OmegaConf.merge(data_schema, cfg.data_config)
    streams_configs = load_and_concat_yamls(data_config.path)

    del data_config.path  # pyright: ignore[reportAttributeAccessIssue]

    data_config.dataset.streams = streams_configs

    scheduler_schema = OmegaConf.structured(SchedulerConfig)
    scheduler_config = OmegaConf.merge(scheduler_schema, cfg.scheduler_config)

    fsdp_schema = OmegaConf.structured(FSDPConfig)
    fsdp_config = cfg.get("fsdp_config", {})
    if fsdp_config:
        fsdp_config = OmegaConf.merge(fsdp_schema, fsdp_config)

    # Merge all configs into one dict for duplicate key checking
    merged_config: dict[str, object] = {}
    merged_config.update(
        cast(dict[str, object], OmegaConf.to_container(model_config, resolve=True))
    )
    merged_config.update(
        cast(dict[str, object], OmegaConf.to_container(trainer_config, resolve=True))
    )
    merged_config.update(
        cast(dict[str, object], OmegaConf.to_container(data_config, resolve=True))
    )
    merged_config.update(
        cast(dict[str, object], OmegaConf.to_container(scheduler_config, resolve=True))
    )
    if fsdp_config:
        merged_config.update(
            cast(dict[str, object], OmegaConf.to_container(fsdp_config, resolve=True))
        )
        # cfg.fsdp_config.load_planner = fsdp_config.get("load_planner", "default")

    check_duplicate_keys(merged_config)

    # Convert merged_config back into an OmegaConf DictConfig
    cfg.model_config = model_config
    cfg.trainer_config = trainer_config
    cfg.data_config = data_config
    cfg.scheduler_config = scheduler_config

    if fsdp_config:
        cfg.fsdp_config = fsdp_config

    return cfg


def create_param_groups(
    model,
    eps,
    base_depth,
    current_depth,
    base_width,
    current_width,
    cp_alpha,
    default_wd=1e-5,
    frozen_param_names=None,
):
    if frozen_param_names is None:
        frozen_param_names = []
    depth_lr_scaling = (current_depth / base_depth) ** (cp_alpha - 1)
    width_lr_scaling = (current_width / base_width) ** (-1)
    emb_params = []
    hidden_ln_params = []
    hidden_weight_params = []
    hidden_bias_params = []
    final_ln_params = []
    lm_head_params = []
    print("Optimzer parameter scaling factors:", depth_lr_scaling, width_lr_scaling)
    frozen_count = 0
    adam_eps = (
        eps
        * (current_width / base_width) ** (-1)
        * (current_depth / base_depth) ** (-1 * cp_alpha)
    )

    total_params = sum(1 for _ in model.parameters())
    total_named_params = sum(1 for _ in model.named_parameters())
    assigned_params = 0
    for name, param in model.named_parameters():
        # Check if parameter should be frozen
        if any(frozen_name in name for frozen_name in frozen_param_names):
            param.requires_grad = False
            print(f"Frozen parameter: {name}")
            frozen_count += 1
            continue
        if name == "model.embedding.weight":
            emb_params.append(param)
            assigned_params += 1
        elif "transformer" in name:
            if "pre" in name or "post" in name:
                # print("norm name", name, flush=True)
                hidden_ln_params.append(param)
                assigned_params += 1
            elif "weight" in name:
                # print("weight name", name, flush=True)
                hidden_weight_params.append(param)
                assigned_params += 1
            elif "bias" in name:
                # print("bias name", name, flush=True)
                hidden_bias_params.append(param)
                assigned_params += 1
        elif name == "model.out_norm.weight":
            final_ln_params.append(param)
            assigned_params += 1
        elif "model.lm_head" in name:
            lm_head_params.append(param)
            assigned_params += 1
    print(f"Assigned {assigned_params} parameters")
    print(f"Total parameters: {total_params}")
    print(f"Total named parameters: {total_named_params}")

    print(f"Frozen {frozen_count} parameter groups")
    # Maintain order for lr updates consistency
    optim_groups = [
        {
            "params": emb_params,
            "weight_decay": default_wd,
            "lr_scale": 1.0,
            "eps": eps,
        },
        {
            "params": hidden_ln_params,
            "weight_decay": 0.0,
            "lr_scale": depth_lr_scaling,
            "eps": adam_eps,
        },
        {
            "params": hidden_weight_params,
            "weight_decay": default_wd / width_lr_scaling,
            "lr_scale": width_lr_scaling * depth_lr_scaling,
            "eps": adam_eps,
        },
        {
            "params": hidden_bias_params,
            "weight_decay": 0.0,
            "lr_scale": depth_lr_scaling,
            "eps": adam_eps,
        },
        {
            "params": final_ln_params,
            "weight_decay": 0.0,
            "lr_scale": 1.0,
            "eps": adam_eps,
        },
        {
            "params": lm_head_params,
            "weight_decay": default_wd,
            "lr_scale": 1.0,
            "eps": eps,
        },
    ]
    return optim_groups


# To get a shortened hash (e.g., 7 characters), use 'git rev-parse --short HEAD'
def get_current_git_short_hash(repo_path=".") -> str:
    """Retrieves the abbreviated Git commit hash of the current repository HEAD."""
    # The logic is similar, but with the '--short' flag
    short_hash = (
        subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_path,
            stderr=subprocess.STDOUT,
        )
        .decode("ascii")
        .strip()
    )

    return short_hash
