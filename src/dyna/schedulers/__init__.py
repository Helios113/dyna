from llmfoundry.registry import schedulers

from dyna.schedulers.scheduler import ConstantWithLinWarmupAndCosCooldown
from dyna.schedulers.wsld import WarmupStableLinearDecay

__all__ = [
    "ConstantWithLinWarmupAndCosCooldown",
    "WarmupStableLinearDecay",
]

schedulers.register("wsld", func=WarmupStableLinearDecay)
