"""Basic tests for attention submodule."""

from dyna.attention import BasicAttn


def test_attention_import():
    """Test that attention can be imported."""
    assert BasicAttn is not None


def test_attention_instantiation():
    """Test basic attention instantiation."""
    attention = BasicAttn(d_model=512, n_heads=8, d_head=64)
    assert attention is not None
