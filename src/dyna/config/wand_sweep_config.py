from dataclasses import dataclass

from .enums import SweepType


@dataclass
class WandbSweep:
    name: str = "none"
    type: SweepType = SweepType.continious
    min_val: int | float | None = None
    max_val: int | float | None = None
    step_size: int | float | None = None
    category_list: list[str] | None = None
