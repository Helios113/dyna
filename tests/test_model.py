"""Basic tests for model submodule."""

from dyna.model import ComposerDynaModel, DynaFormer


def test_model_import():
    """Test that model can be imported."""
    assert ComposerDynaModel is not None
    assert DynaFormer is not None
