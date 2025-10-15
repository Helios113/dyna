"""Basic tests for modules submodule."""

from dyna.modules import AttentionModule, DynaModule


def test_modules_import():
    """Test that modules can be imported."""
    assert DynaModule is not None
    assert AttentionModule is not None


def test_modules_instantiation():
    """Test basic module instantiation."""
    module = DynaModule()
    assert module is not None

    attention = AttentionModule(d_model=512, n_heads=8, d_head=64)
    assert attention is not None
