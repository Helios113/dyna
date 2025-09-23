from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Iterable
import torch
from torch.distributed.checkpoint.planner import LoadPlanner
from torch.distributed.checkpoint.planner import SavePlanner

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
    shuffle: bool = True
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
    max_seq_len: int | None = 1024

@dataclass
class DataConfig:
    path: str
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    name: str = "text"
    drop_last: bool = True
    num_workers: int = 1
    pin_memory: bool = True
    device_batch_size: int = 1024
