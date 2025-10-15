"""Basic tests for schedulers submodule."""

from dyna.schedulers import scheduler, wsld


def test_schedulers_import():
    """Test that schedulers can be imported."""
    assert scheduler is not None
    assert wsld is not None
