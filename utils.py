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
from model_config import DatasetConfig, DataConfig

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
    keys = [ 'type','model', 'd_model', 'n_layers', 'n_heads', 'ff_n_experts', 'att_n_experts', 'ff_expert_size', 'dropout']
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
    

    data_schema = OmegaConf.structured(DataConfig)
    

    data_config = OmegaConf.merge(data_schema, cfg.data_config)
    streams_configs = load_and_concat_yamls(data_config.path)
    del data_config.path
    data_config.dataset.streams = streams_configs
    
    os.environ["S3_ENDPOINT_URL"] = "http://128.232.115.19:9000"

    train_loader = build_dataloader(
        OmegaConf.to_container(data_config),
        tokenizer,
        device_train_batch_size,
    )
    return train_loader