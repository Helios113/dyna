from __future__ import annotations

from importlib import import_module
from typing import Any

_UTILS_MODULE = "dyna.utils.utils"

__all__ = [
    "build_full_concrete_config",
    "check_duplicate_keys",
    "condition_model",
    "create_param_groups",
    "generate_id",
    "get_callbacks",
    "get_data_loader",
    "get_scheduler",
    "make_wandb_run_name",
    "get_current_git_short_hash",
]


def _load_utils():
    return import_module(_UTILS_MODULE)


def __getattr__(name: str) -> Any:
    if name == "condition_model":
        from .condition_model import condition_model as _condition_model

        globals()[name] = _condition_model
        return _condition_model

    if name in __all__:
        module = _load_utils()
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'dyna.utils' has no attribute '{name}'")
