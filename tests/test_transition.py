"""Basic tests for transition submodule."""

from dyna.transition import BasicFFN


def test_transition_import():
    """Test that transition can be imported."""
    assert BasicFFN is not None


def test_transition_instantiation():
    """Test basic transition instantiation."""
    ffn = BasicFFN(d_model=512, d_expert_ffn=2048)
    assert ffn is not None
