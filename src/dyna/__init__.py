"""dyna: A modular package for dynamic modeling and attention-based architectures.

Type checking is automatically applied to all modules in the package using
jaxtyping and beartype.

Author: Preslav Aleksandorv
Version: 0.1.0

"""

__version__ = "0.1.0"
__author__ = "Preslav Aleksandorv"

# from jaxtyping import install_import_hook
from beartype.claw import beartype_packages

beartype_packages(
    (
        "attention",
        "callbacks",
        "config",
        "data",
        "layers",
        "modules",
        "schedulers",
        "transition",
        "utils",
    )
)

from dyna import (  # noqa: E402
    attention,
    callbacks,
    config,
    data,
    kernel,
    layers,
    model,
    modules,
    norms,
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
