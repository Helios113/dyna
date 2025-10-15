"""Basic tests for cvmm submodule."""

from dyna.cvmm import cvmm, cvmm_sel


def test_cvmm_import():
    """Test that cvmm can be imported."""
    assert cvmm is not None
    assert cvmm_sel is not None
