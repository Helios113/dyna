"""
dyna: A modular package for dynamic modeling and attention-based architectures.

Type checking is automatically applied to all modules in the package using jaxtyping and beartype.

Author: Preslav Aleksandorv
Version: 0.1.0

"""

__version__ = "0.1.0"
__author__ = "Preslav Aleksandorv"


from jaxtyping import install_import_hook
install_import_hook("dyna", "beartype.beartype")

from . import attention
from . import data
from . import model
from . import schedulers
from . import callbacks
from . import utils


# Apply jaxtyping to all modules in this package


__all__ = ['attention', 'data', "model", "schedulers", "utils", "callbacks"]
