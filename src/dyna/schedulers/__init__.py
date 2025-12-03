from dyna.registry import schedulers
from dyna.schedulers.scheduler import ConstantWithLinWarmupAndCosCooldown
from dyna.schedulers.wsld import WarmupStableLinearDecay

__all__ = [
    "ConstantWithLinWarmupAndCosCooldown",
    "WarmupStableLinearDecay",
]

schedulers.register("wsld", WarmupStableLinearDecay)
schedulers.register(
    "constant_with_lin_warmup_and_cos_cooldown",
    ConstantWithLinWarmupAndCosCooldown,
)
