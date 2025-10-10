from __future__ import annotations

import glob
import os
import secrets
import string
import time

import yaml
from composer import DataSpec
from composer.core import Callback
from composer.optim.scheduler import ComposerScheduler
from llmfoundry.utils.builders import build_callback, build_dataloader, build_scheduler
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

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
    model_config: DictConfig, trainer_config: DictConfig, unique: str
) -> str:
    """Generate a unique and informative wandb run name from config.

    Includes a consistent part (run_id), a unique part (timestamp), and key info.
    """
    # Consistent part: use config['run_id'] if present, else generate random string
    run_name = trainer_config.get("run_name")
    load_path = trainer_config.get("load_path")
    if load_path is not None:
        name = os.path.basename(load_path)
        name_break_down = name.split("_")
        name_break_down[4] = unique
        name = "_".join(name_break_down)
    else:
        if not run_name:
            run_name = generate_id(8)
        else:
            # check that there are no underscores in run_name
            if "_" in run_name:
                raise ValueError("run_name should not contain underscores")
        # Unique part: timestamp
        timestamp = time.strftime("%d%b%y").lower()
        # Important info: select a few key hyperparameters (customize as needed)
        abbrev_moa = {
            "d_model": "dim",
            "n_layers": "n_l",
            "n_repeats": "n_r",
            "n_heads": "n_h",
            "d_head": "d_hd",
            "n_experts_ffn": "n_e_ffn",
            "n_experts_attn": "n_e_attn",
            "ff_expert_size": "f_e_size",
            "device_train_batch_size": "bs",
            "enable_early_exit": "ee",
            "execution_mode": "mode",
            "norm_structure": "norm",
            "rescaling_method": "rescale",
        }

        abbrev_trans = {
            "d_model": "dim",
            "d_ffn": "d_ffn",
            "n_layers": "n_l",
            "n_heads": "n_h",
            "d_head": "d_hd",
            "device_train_batch_size": "bs",
            "enable_early_exit": "ee",
            "execution_mode": "mode",
            "norm_structure": "norm",
            "rescaling_method": "rescale",
        }

        if model_config.get("execution_mode") == "moe":
            keys = abbrev_moa
        else:
            keys = abbrev_trans

        info = []
        for k, short in keys.items():
            val = None
            if k in model_config:
                val = model_config[k]
            elif k in trainer_config:
                val = trainer_config[k]
            if "." in str(val):
                val = str(val).split(".")[-1]  # Simplify floats
            if val is not None:
                info.append(f"{short}~{val}")

        info_str = "_".join(info)
        # Compose name
        name = f"{run_name}_{timestamp}_{unique}"
        if info_str:
            name += f"_{info_str}"
    has_index = name.split("_")[0].isdigit()
    if has_index:
        index = int(name.split("_")[0])
        name = "_".join(name.split("_")[1:])
        name = str(index + 1) + "_" + name
    else:
        name = "1_" + name
    return name


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
    """Reads all YAML files in a directory, loads them, and merges them into a single dict.

    Returns an OmegaConf DictConfig.
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
        OmegaConf.to_container(cfg),
        tokenizer,
        device_train_batch_size,
    )
    return train_loader


def get_scheduler(cfg: DictConfig) -> ComposerScheduler:
    scheduler_name = cfg.name
    del cfg.name
    scheduler = build_scheduler(name=scheduler_name, scheduler_config=cfg)
    return scheduler


def check_duplicate_keys(cfg, value_map=None, exceptions=None, path=""):
    """Traverse every key in the config (recursively, regardless of path) and add its value to a flat dictionary.

    If a key is seen again, check if the value matches all previous values; if not, raise ValueError.
    Allows duplicate keys for those listed in `exceptions` (full path or key).
    Prints the full path of the key and the previous path when a conflict is found.
    """
    if value_map is None:
        value_map = {}
    if exceptions is None:
        exceptions = ["remote", "local", "dataset.split", "proportion"]
    if isinstance(cfg, dict) or hasattr(cfg, "keys"):
        for k in cfg.keys():
            v = cfg[k]
            full_path = f"{path}.{k}" if path else k
            excepted = (k in exceptions) or (full_path in exceptions)
            if not excepted:
                if k in value_map:
                    prev_val, prev_path = value_map[k]
                    if prev_val != v:
                        print(f"Conflict at: {full_path} and {prev_path}")
                        raise ValueError(
                            f"Duplicate key '{k}' with different values: {prev_val} (at {prev_path}) vs {v} (at {full_path})"
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


def build_full_concrete_config(cfg):
    """Constructs and merges all configs (model, trainer, data) and returns a single config dict."""
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
    del data_config.path
    data_config.dataset.streams = streams_configs

    scheduler_schema = OmegaConf.structured(SchedulerConfig)
    scheduler_config = OmegaConf.merge(scheduler_schema, cfg.scheduler_config)

    fsdp_schema = OmegaConf.structured(FSDPConfig)
    fsdp_config = cfg.get("fsdp_config", {})
    if fsdp_config:
        fsdp_config = OmegaConf.merge(fsdp_schema, fsdp_config)

    # Merge all configs into one dict for duplicate key checking
    merged_config = {}
    merged_config.update(OmegaConf.to_container(model_config, resolve=True))
    merged_config.update(OmegaConf.to_container(trainer_config, resolve=True))
    merged_config.update(OmegaConf.to_container(data_config, resolve=True))
    merged_config.update(OmegaConf.to_container(scheduler_config, resolve=True))
    if fsdp_config:
        merged_config.update(OmegaConf.to_container(fsdp_config, resolve=True))
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


def visualize_attention_mask(mask, max_display=2048, index=0):
    """Helper function to visualize the attention mask"""
    import matplotlib.pyplot as plt

    # Only show first max_display tokens for readability
    display_len = min(max_display, mask.shape[-1])
    mask_subset = mask[0, 0].cpu()
    print(mask_subset)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mask_subset.float(), cmap="Blues", aspect="equal")

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.set_title("Causal Attention Mask for Packed Sequence")
    ax.set_xlabel("Key Positions")
    ax.set_ylabel("Query Positions")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"attn_mask_{index}.png")


def visualize_position_mask(mask, index=0):
    """Helper function to visualize the attention mask"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    print(mask[0])
    ax.plot(mask[0].cpu())

    plt.tight_layout()
    plt.show()
    plt.savefig(f"pos_mask_{index}.png")
