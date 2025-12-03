"""Lightweight registries for Dyna components.

This module replaces the llm-foundry registries with a minimal implementation
that is sufficient for the components we register locally (e.g., norms,
callbacks, schedulers).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    """Simple name -> factory registry."""

    def __init__(self, name: str):
        self._name = name
        self._store: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any]) -> None:
        key = name.lower()
        if key in self._store:
            raise ValueError(
                f"{self._name} registry already contains an entry named '{name}'"
            )
        self._store[key] = func

    def get(self, name: str) -> Callable[..., Any]:
        key = name.lower()
        if key not in self._store:
            raise KeyError(
                f"{self._name} registry has no entry named '{name}'"
            )
        return self._store[key]

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        factory = self.get(name)
        return factory(*args, **kwargs)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._store


norms = Registry("norms")
callbacks = Registry("callbacks")
schedulers = Registry("schedulers")
dataset_replication_validators = Registry("dataset_replication_validators")
collators = Registry("collators")
data_specs = Registry("data_specs")
icl_datasets = Registry("icl_datasets")
metrics = Registry("metrics")


__all__ = [
    "Registry",
    "norms",
    "callbacks",
    "schedulers",
    "dataset_replication_validators",
    "collators",
    "data_specs",
    "icl_datasets",
]
