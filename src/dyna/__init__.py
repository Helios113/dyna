"""dyna: A modular package for dynamic modeling and attention-based architectures.

Type checking is automatically applied to all modules in the package using
jaxtyping and beartype.

Author: Preslav Aleksandorv
Version: 0.1.0

"""

__version__ = "0.1.0"
__author__ = "Preslav Aleksandorv"

from jaxtyping import install_import_hook

with install_import_hook("dyna", "beartype.beartype"):
    from dyna import (
        attention,
        callbacks,
        config,
        kernel,
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
    "kernel",
    "data",
    "layers",
    "modules",
    "schedulers",
    "transition",
    "utils",
]
