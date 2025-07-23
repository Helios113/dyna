from pprint import pprint
import random
import string
import time
import secrets
import string
from composer.core import Callback
from llmfoundry.utils.builders import build_callback, build_dataloader
import os
from transformers import PreTrainedTokenizerBase
from composer import DataSpec
from omegaconf import OmegaConf
from model_config import ModelConfig, TrainerConfig, DataConfig
def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits."""
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def make_wandb_run_name(config: dict) -> str:
    """
    Generate a unique and informative wandb run name from config.
    Includes a consistent part (run_id), a unique part (timestamp), and key info.
    """
    # Consistent part: use config['run_id'] if present, else generate random string
    run_id = config.get('run_id')
    if not run_id:
        run_id = generate_id(8)
    # Unique part: timestamp
    timestamp = time.strftime("%d%b%y").lower()
    unique = generate_id(8)
    # Important info: select a few key hyperparameters (customize as needed)
    keys = [ 'type','model', 'd_model', 'n_layers', 'n_heads', 'n_ffn_experts', 'n_att_experts', 'ff_expert_size', 'dropout']
    info = []
    for k in keys:
        if k in config:
            info.append(f"{k[:4]}={config[k]}")
    info_str = "_".join(info)
    # Compose name
    name = f"{run_id}_{timestamp}_{unique}"
    if info_str:
        name += f"_{info_str}"
    return name


def get_callbacks(cfg: dict) -> list[Callback]:

    return [
        build_callback(
            name=str(name),
            kwargs=callback_cfg,
            train_config=cfg,
        )
        for name, callback_cfg in cfg.items()
    ]
    
import glob
import yaml
from omegaconf import OmegaConf

def load_and_concat_yamls(directory):
    """
    Reads all YAML files in a directory, loads them, and merges them into a single dict.
    Returns an OmegaConf DictConfig.
    """
    merged = {}
    for file in sorted(glob.glob(os.path.join(directory, "*.yaml")) + glob.glob(os.path.join(directory, "*.yml"))):
        with open(file, "r") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                merged.update(data)
            else:
                raise ValueError(f"YAML file {file} does not contain a dict.")
    return OmegaConf.create(merged)
    
def get_data_loader(
    cfg: dict,
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

def check_duplicate_keys(cfg, value_map=None, exceptions=None, path=""):
    """
    Traverse every key in the config (recursively, regardless of path) and add its value to a flat dictionary.
    If a key is seen again, check if the value matches all previous values; if not, raise ValueError.
    Allows duplicate keys for those listed in `exceptions` (full path or key).
    Prints the full path of the key and the previous path when a conflict is found.
    """
    if value_map is None:
        value_map = {}
    if exceptions is None:
        exceptions = ["remote", "local", "dataset.split"]
    if isinstance(cfg, dict) or hasattr(cfg, 'keys'):
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
            if isinstance(v, dict) or hasattr(v, 'keys'):
                check_duplicate_keys(v, value_map, exceptions, full_path)
            elif isinstance(v, list):
                for idx, item in enumerate(v):
                    check_duplicate_keys(item, value_map, exceptions, f"{full_path}[{idx}]")
    elif isinstance(cfg, list):
        for idx, item in enumerate(cfg):
            check_duplicate_keys(item, value_map, exceptions, f"{path}[{idx}]")
    return value_map

def build_full_concrete_config(cfg):
    """
    Constructs and merges all configs (model, trainer, data) and returns a single config dict.
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
    del data_config.path
    data_config.dataset.streams = streams_configs

    # Merge all configs into one dict for duplicate key checking
    merged_config = {}
    merged_config.update(OmegaConf.to_container(model_config, resolve=True))
    merged_config.update(OmegaConf.to_container(trainer_config, resolve=True))
    merged_config.update(OmegaConf.to_container(data_config, resolve=True))
    check_duplicate_keys(merged_config)
    
    # Convert merged_config back into an OmegaConf DictConfig
    cfg.model_config = model_config
    cfg.trainer_config = trainer_config
    cfg.data_config = data_config
    return cfg