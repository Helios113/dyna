from torch.nn import LayerNorm, RMSNorm as TorchRMSNorm

from dyna.registry import norms

from .dynamic_tanh import DynamicTanh
from .ln_norm import LNNorm
from .rms_norm import RMSNorm
from .unit_norm import UnitNorm

__all__ = ["DynamicTanh", "UnitNorm", "RMSNorm", "LNNorm"]

# Standard Torch norms (equivalents of the llm-foundry low precision variants)
norms.register("layernorm", LayerNorm)
norms.register("low_precision_layernorm", LayerNorm)
norms.register("rmsnorm", TorchRMSNorm)
norms.register("low_precision_rmsnorm", TorchRMSNorm)

# Custom Dyna norms
norms.register("dynamic_tanh", DynamicTanh)
norms.register("rms_norm", RMSNorm)
norms.register("unit_norm", UnitNorm)
norms.register("ln_norm", LNNorm)
