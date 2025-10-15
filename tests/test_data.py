"""Basic tests for data submodule."""

from dyna.data import constants, types


def test_data_import():
    """Test that data can be imported."""
    assert constants is not None
    assert types is not None
