from llmfoundry.layers_registry import norms

from .dynamic_tanh import DynamicTanh
from .ln_norm import LNNorm
from .rms_norm import RMSNorm
from .unit_norm import UnitNorm

__all__ = ["DynamicTanh", "UnitNorm", "RMSNorm", "LNNorm"]

norms.register("dynamic_tanh", func=DynamicTanh)
norms.register("rms_norm", func=RMSNorm)
norms.register("unit_norm", func=UnitNorm)
norms.register("ln_norm", func=LNNorm)
