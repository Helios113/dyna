"""Basic tests for utils submodule."""

from dyna.utils import config_utils, utils


def test_utils_import():
    """Test that utils can be imported."""
    assert config_utils is not None
    assert utils is not None
