"""dyna: A modular package for dynamic modeling and attention-based architectures.

Type checking is automatically applied to all modules in the package using
jaxtyping and beartype.

Author: Preslav Aleksandorv
Version: 0.1.0

"""

__version__ = "0.1.0"
__author__ = "Preslav Aleksandorv"

# from jaxtyping import install_import_hook
from beartype.claw import beartype_this_package

beartype_this_package()

from dyna import (
    attention,
    callbacks,
    config,
    data,
    layers,
    model,
    modules,
    schedulers,
    transition,
    utils,
)

__all__ = [
    "attention",
    "callbacks",
    "config",
    "data",
    "kernel",
    "layers",
    "modules",
    "schedulers",
    "transition",
    "utils",
]
