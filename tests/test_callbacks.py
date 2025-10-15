"""Basic tests for callbacks submodule."""

from dyna.callbacks import abbie_number, activation_monitor


def test_callbacks_import():
    """Test that callbacks can be imported."""
    assert abbie_number is not None
    assert activation_monitor is not None
