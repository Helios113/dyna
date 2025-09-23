from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Iterable
import torch
from torch.distributed.checkpoint.planner import LoadPlanner
from torch.distributed.checkpoint.planner import SavePlanner
from .enums import (
    NormStructure,
    RescaleMethod,
    ExecutionMode,
)


@dataclass
class FSDPConfig:
    activation_checkpointing: bool = False
    activation_checkpointing_reentrant: bool = True
    activation_cpu_offload: bool = False
    backward_prefetch: str = 'BACKWARD_POST'  # 'BACKWARD_PRE' | 'BACKWARD_POST' | 'NONE'
    cpu_offload: bool = False  # cpu_offload not supported yet
    data_parallel_shard_degree: int = -1
    data_parallel_replicate_degree: int = 1
    forward_prefetch: bool = False
    ignored_modules = None
    keep_low_precision_grads: bool = False
    limit_all_gathers: bool = False
    load_monolith_rank0_only: bool = False
    load_planner = None
    mixed_precision: str = 'DEFAULT'  # 'FULL' | 'DEFAULT' | 'PURE'
    save_planner = None
    sharded_ckpt_prefix_dir: str = 'ep{epoch}-ba{batch}'
    sharding_strategy: str = 'FULL_SHARD'  # 'FULL_SHARD' | 'SHARD_GRAD_OP' | 'NO_SHARD'
    state_dict_type: str = 'full'  # 'full' | 'local' | 'sharded'
    sync_module_states: bool = False
    use_orig_params: bool = True
    verbose: bool = False