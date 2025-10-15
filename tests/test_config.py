"""Basic tests for config submodule."""

from dyna.config import ExecutionMode, NormStructure, RescaleMethod


def test_config_import():
    """Test that config can be imported."""
    assert ExecutionMode is not None
    assert NormStructure is not None
    assert RescaleMethod is not None


def test_config_instantiation():
    """Test basic config instantiation."""
    # Test enums
    assert ExecutionMode.moe == "moe"
    assert NormStructure.peri == "peri"
    assert RescaleMethod.none == "none"
