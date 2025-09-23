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
class SchedulerConfig:
    name: str = "wsld"
    t_warmup: str | None = "1ba"
    t_max: str | None = "1dur"
    t_cooldown: str | None = None
    alpha_f: float = 0.0
    scale_warmup: bool = False
    scale_cooldown: bool = False

