from __future__ import annotations

import csv
import functools
import glob
import logging
import os
import secrets
import string
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar, cast


import catalogue
import yaml
from composer import DataSpec
from composer.core import Callback
from composer.optim.scheduler import ComposerScheduler
import omegaconf as om
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from dyna.config import (
    DataConfig,
    EvalConfig,
    FSDPConfig,
    ICLTaskConfig,
    ModelConfig,
    SchedulerConfig,
    TrainerConfig,
)
from dyna.data.text_data import build_text_dataloader

T = TypeVar("T")
TypeBoundT = TypeVar("TypeBoundT", bound=type[Any])


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
    cfg_dict = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    dataset_cfg = cast(dict[str, Any], cfg_dict.get("dataset", {})).copy()
    return build_text_dataloader(
        tokenizer=tokenizer,
        device_batch_size=device_train_batch_size,
        dataset=dataset_cfg,
        drop_last=bool(cfg_dict.get("drop_last", True)),
        num_workers=int(cfg_dict.get("num_workers", 0)),
        pin_memory=bool(cfg_dict.get("pin_memory", False)),
        prefetch_factor=int(cfg_dict.get("prefetch_factor", 2)),
        persistent_workers=bool(cfg_dict.get("persistent_workers", False)),
        timeout=int(cfg_dict.get("timeout", 0)),
    )





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
        exceptions = [
            "remote",
            "local",
            "dataset.split",
            "proportion",
            "label",
            "dataset_uri",
            "icl_task_type",
            "continuation_delimiter",
            "example_delimiter",
            "prompt_string",
            "num_fewshot",
        ]
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
    Includes validation of all configuration schemas.
    """
    log = logging.getLogger(__name__)
    log.info("Validating and building full configuration...")

    OmegaConf.resolve(cfg)
    # Model Config
    model_config = None
    models = None
    if "model_config" in cfg and "models" not in cfg:
        model_schema = OmegaConf.structured(ModelConfig)
        model_config = OmegaConf.merge(model_schema, cfg.model_config)
    elif "models" in cfg:
        model_schema = OmegaConf.structured(ModelConfig)
        model_list = []
        for mod in cfg.models:
            model_list.append(OmegaConf.merge(model_schema, mod))
        models = model_list

    # Trainer Config
    if "trainer_config" in cfg:
        trainer_schema = OmegaConf.structured(TrainerConfig)
        trainer_config = OmegaConf.merge(trainer_schema, cfg.trainer_config)

    # Data Config (including streams)
    if "data_config" in cfg:
        data_schema = OmegaConf.structured(DataConfig)
        data_config = OmegaConf.merge(data_schema, cfg.data_config)
        streams_configs = load_and_concat_yamls(data_config.path)

        del data_config.path  # pyright: ignore[reportAttributeAccessIssue]

        data_config.dataset.streams = streams_configs

    if "scheduler_config" in cfg:
        scheduler_schema = OmegaConf.structured(SchedulerConfig)
        scheduler_config = OmegaConf.merge(scheduler_schema, cfg.scheduler_config)

    if "fsdp_config" in cfg:
        fsdp_schema = OmegaConf.structured(FSDPConfig)
        fsdp_config = cfg.get("fsdp_config", {})
        if fsdp_config:
            fsdp_config = OmegaConf.merge(fsdp_schema, fsdp_config)

    if "eval_config" in cfg:
        eval_schema = OmegaConf.structured(EvalConfig)

        # Validate ICL tasks if present in eval_config
        if "icl_tasks" in cfg.eval_config and cfg.eval_config.icl_tasks:
            validated_tasks = []
            for task in cfg.eval_config.icl_tasks:
                task_schema = OmegaConf.structured(ICLTaskConfig)
                validated_task = OmegaConf.merge(task_schema, task)
                validated_tasks.append(validated_task)
            cfg.eval_config.icl_tasks = validated_tasks

        eval_config = OmegaConf.merge(eval_schema, cfg.eval_config)

    # Merge all configs into one dict for duplicate key checking
    merged_config: dict[str, object] = {}
    if model_config in cfg:
        merged_config.update(
            cast(dict[str, object], OmegaConf.to_container(model_config, resolve=True))
        )
    if "trainer_config" in cfg:
        merged_config.update(
            cast(
                dict[str, object], OmegaConf.to_container(trainer_config, resolve=True)
            )
        )
    if "data_config" in cfg:
        merged_config.update(
            cast(dict[str, object], OmegaConf.to_container(data_config, resolve=True))
        )
    if "scheduler_config" in cfg:
        merged_config.update(
            cast(
                dict[str, object],
                OmegaConf.to_container(scheduler_config, resolve=True),
            )
        )
    if "fsdp_config" in cfg:
        merged_config.update(
            cast(dict[str, object], OmegaConf.to_container(fsdp_config, resolve=True))
        )
        # cfg.fsdp_config.load_planner = fsdp_config.get("load_planner", "default")
    if "eval_config" in cfg:
        merged_config.update(
            cast(dict[str, object], OmegaConf.to_container(eval_config, resolve=True))
        )

    check_duplicate_keys(merged_config)

    # Convert merged_config back into an OmegaConf DictConfig
    if models is None:
        cfg.model_config = model_config
    else:
        cfg.models = models
    if "trainer_config" in cfg:
        cfg.trainer_config = trainer_config

    if "data_config" in cfg:
        cfg.data_config = data_config

    if "scheduler_config" in cfg:
        cfg.scheduler_config = scheduler_config

    if "fsdp_config" in cfg:
        cfg.fsdp_config = fsdp_config

    if "eval_config" in cfg:
        cfg.eval_config = eval_config

    log.info("✓ Configuration validation and build successful")
    return cfg


def create_param_groups(
    model,
    lr,
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
    print(
        f"Depth LR scaling: {depth_lr_scaling}, Width LR scaling: {width_lr_scaling}",
        flush=True,
    )
    emb_params = []
    hidden_ln_params = []
    hidden_weight_params = []
    hidden_bias_params = []
    final_ln_params = []
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
        elif name == "model.out_norm.weight":
            final_ln_params.append(param)
            assigned_params += 1
        elif "model.lm_head" in name:
            emb_params.append(param)
            assigned_params += 1
        elif "transformer" in name:
            if "pre" in name or "post" in name:
                # print("norm name", name, flush=True)
                print("names in hidden ln", name, flush=True)
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
        else:
            # Default to emb_params weights as no scaling is applied
            emb_params.append(param)
            assigned_params += 1

    print(f"Assigned {assigned_params} parameters")
    print(f"Total parameters: {total_params}")
    print(f"Total named parameters: {total_named_params}")

    print(f"Frozen {frozen_count} parameter groups")
    # Maintain order for lr updates consistency
    names = ["embedding", "hidden_ln", "hidden_weight", "hidden_bias", "final_ln"]
    optim_groups = [
        {
            "params": emb_params,
            "weight_decay": default_wd,
            "lr": 1.0 * lr,
            "eps": eps,
        },
        {
            "params": hidden_ln_params,
            "weight_decay": 0.0,
            "lr": depth_lr_scaling * lr,
            "eps": adam_eps,
        },
        {
            "params": hidden_weight_params,
            "weight_decay": default_wd / width_lr_scaling,
            "lr": width_lr_scaling * depth_lr_scaling * lr,
            "eps": adam_eps,
        },
        {
            "params": hidden_bias_params,
            "weight_decay": 0.0,
            "lr": depth_lr_scaling * lr,
            "eps": adam_eps,
        },
        {
            "params": final_ln_params,
            "weight_decay": 0.0,
            "lr": 1.0 * lr,
            "eps": adam_eps,
        },
    ]
    for i, val in enumerate(optim_groups):
        print(
            f"{names[i]} with {len(val['params'])} params, wd: {val['weight_decay']}, lr: {val['lr']}, eps: {val['eps']}"
        )
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

class TypedRegistry(catalogue.Registry, Generic[T]):
    """A thin wrapper around catalogue.Registry to add static typing and.

    descriptions.
    """

    def __init__(
        self,
        namespace: Sequence[str],
        entry_points: bool = False,
        description: str = '',
    ) -> None:
        super().__init__(namespace, entry_points=entry_points)

        self.description = description

    def __call__(self, name: str, func: T | None = None) -> Callable[[T], T]:
        return super().__call__(name, func)

    def register(self, name: str, *, func: T | None = None) -> T:
        return super().register(name, func=func)

    def register_class(
        self,
        name: str,
        *,
        func: TypeBoundT | None = None,
    ) -> TypeBoundT:
        return super().register(name, func=func)

    def get(self, name: str) -> T:
        return super().get(name)

    def get_all(self) -> dict[str, T]:
        return super().get_all()

    def get_entry_point(self, name: str, default: T | None = None) -> T:
        return super().get_entry_point(name, default=default)

    def get_entry_points(self) -> dict[str, T]:
        return super().get_entry_points()


def construct_from_registry(
    name: str,
    registry: TypedRegistry,
    partial_function: bool = True,
    pre_validation_function: Callable[[Any], None] | type | None = None,
    post_validation_function: Callable[[Any], None] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> Any:
    """Helper function to build an item from the registry.

    Args:
        name (str): The name of the registered item
        registry (catalogue.Registry): The registry to fetch the item from
        partial_function (bool, optional): Whether to return a partial function for registered callables. Defaults to True.
        pre_validation_function (Callable[[Any], None] | type | None, optional): An optional validation function called
            before constructing the item to return. This should throw an exception if validation fails. Defaults to None.
        post_validation_function (Callable[[Any], None] | None, optional): An optional validation function called after
            constructing the item to return. This should throw an exception if validation fails. Defaults to None.
        kwargs (dict[str, Any] | None): Other relevant keyword arguments.

    Raises:
        ValueError: If the validation functions failed or the registered item is invalid

    Returns:
        Any: The constructed item from the registry
    """
    if kwargs is None:
        kwargs = {}

    registered_constructor = registry.get(name)

    if pre_validation_function is not None:
        if isinstance(pre_validation_function, type):
            if not issubclass(registered_constructor, pre_validation_function):
                raise ValueError(
                    f'Expected {name} to be of type {pre_validation_function}, but got {type(registered_constructor)}',
                )
        elif isinstance(pre_validation_function, Callable):
            pre_validation_function(registered_constructor)
        else:
            raise ValueError(
                f'Expected pre_validation_function to be a callable or a type, but got {type(pre_validation_function)}',
            )

    # If it is a class, or a builder function, construct the class with kwargs
    # If it is a function, create a partial with kwargs
    if isinstance(
        registered_constructor,
        type,
    ) or callable(registered_constructor) and not partial_function:
        constructed_item = registered_constructor(**kwargs)
    elif callable(registered_constructor):
        constructed_item = functools.partial(registered_constructor, **kwargs)
    else:
        raise ValueError(
            f'Expected {name} to be a class or function, but got {type(registered_constructor)}',
        )

    if post_validation_function is not None:
        post_validation_function(constructed_item)

    return constructed_item


def to_dict_container(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    maybe_dict = to_container(cfg)
    if isinstance(maybe_dict, dict):
        return maybe_dict
    else:
        raise ValueError(f'Expected a dict-like type, got {type(maybe_dict)}')


def to_list_container(
    cfg: ListConfig | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    maybe_list = to_container(cfg)
    if isinstance(maybe_list, list):
        return maybe_list
    else:
        raise ValueError(f'Expected a list-like type, got {type(maybe_list)}')


def to_container(
    cfg: DictConfig | ListConfig | dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Converts a DictConfig or ListConfig to a dict or list.

    `omegaconf.to_container` does not handle nested DictConfig or ListConfig
    objects, so this function is used to convert them to dicts or lists.
    """
    if isinstance(cfg, DictConfig):
        ret = om.to_container(cfg, resolve=True)
        assert isinstance(ret, dict)
        return ret  # type: ignore (return type is correct and converting all keys to str would be unnecessarily costly)
    elif isinstance(cfg, ListConfig):
        ret = om.to_container(cfg, resolve=True)
        assert isinstance(ret, list)
        return ret  # type: ignore (see above)
    else:
        return cfg  # type: ignore (dicts and lists are already in the correct format)
