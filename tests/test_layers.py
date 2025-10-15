"""Basic tests for layers submodule."""

from dyna.config import ModelConfig
from dyna.layers import SimpleLayer


def test_layers_import():
    """Test that layers can be imported."""
    assert SimpleLayer is not None


def test_layers_instantiation():
    """Test basic layer instantiation."""
    config = ModelConfig()
    layer = SimpleLayer(config)
    assert layer is not None
