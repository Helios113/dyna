"""dyna: A modular package for dynamic modeling and attention-based architectures.

Type checking is automatically applied to all modules in the package using
jaxtyping and beartype.

Author: Preslav Aleksandorv
Version: 0.1.0

"""

__version__ = "0.1.0"
__author__ = "Preslav Aleksandorv"

from jaxtyping import install_import_hook

from dyna import (
    attention,
    callbacks,
    config,
    cvmm,
    data,
    layers,
    modules,
    schedulers,
    transition,
    utils,
)

install_import_hook("dyna", "beartype.beartype")


__all__ = [
    "attention",
    "callbacks",
    "config",
    "cvmm",
    "data",
    "layers",
    "modules",
    "schedulers",
    "transition",
    "utils",
]
