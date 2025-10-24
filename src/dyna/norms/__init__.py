from llmfoundry.layers_registry import norms

from .dynamic_tanh import DynamicTanh
from .unit_norm import UnitNorm

__all__ = ["DynamicTanh", "UnitNorm"]

norms.register("dynamic_tanh", func=DynamicTanh)
norms.register("unit_norm", func=UnitNorm)
