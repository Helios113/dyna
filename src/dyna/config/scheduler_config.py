from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    name: str = "wsld"
    t_warmup: str | None = "1ba"
    t_max: str | None = "1dur"
    t_cooldown: str | None = None
    alpha_f: float = 0.0
    scale_warmup: bool = False
    scale_cooldown: bool = False
