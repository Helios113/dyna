from llmfoundry.layers_registry import norms

from .dynamic_tanh import DynamicTanh

__all__ = ["DynamicTanh"]

norms.register("dynamic_tanh", func=DynamicTanh)
