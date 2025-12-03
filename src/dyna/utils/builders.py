"""Local builders replacing llm-foundry helpers."""
from __future__ import annotations

from enum import Enum
from typing import Any

from composer.metrics.nlp import LanguageCrossEntropy, LanguagePerplexity

try:  # Composer >=0.20 removed TokenAccuracy
    from composer.metrics.nlp import TokenAccuracy
except ImportError:  # pragma: no cover - Composer version mismatch
    from composer.metrics.nlp import MaskedAccuracy as TokenAccuracy

from dyna.registry import callbacks, norms, schedulers


def _normalize_name(name: str | Enum) -> str:
    if isinstance(name, Enum):
        return str(name.value)
    return str(name)


def build_norm(name: str | Enum, **kwargs: Any):
    """Instantiate a registered normalization module."""
    try:
        factory = norms.get(_normalize_name(name))
    except KeyError as exc:  # pragma: no cover - developer error
        raise ValueError(f"Unknown norm type '{name}'") from exc
    return factory(**kwargs)


_METRICS = {
    "language_cross_entropy": LanguageCrossEntropy,
    "language_perplexity": LanguagePerplexity,
    "token_accuracy": TokenAccuracy,
}


def build_metric(name: str, kwargs: dict[str, Any] | None = None):
    """Instantiate one of the default Composer NLP metrics."""
    metric_name = name.lower()
    if metric_name not in _METRICS:  # pragma: no cover - developer error
        raise ValueError(f"Unknown metric '{name}'")
    metric_kwargs = kwargs or {}
    return _METRICS[metric_name](**metric_kwargs)


def build_callback(
    name: str,
    kwargs: dict[str, Any] | None = None,
) -> Any:
    """Instantiate a callback by name from the callback registry."""
    try:
        factory = callbacks.get(name)
    except KeyError as exc:  # pragma: no cover - developer error
        raise ValueError(f"Unknown callback '{name}'") from exc
    callback_kwargs = kwargs or {}
    return factory(**callback_kwargs)


def build_scheduler(
    name: str,
    scheduler_config: dict[str, Any] | None = None,
):
    """Instantiate a scheduler from the scheduler registry."""
    try:
        factory = schedulers.get(name)
    except KeyError as exc:  # pragma: no cover - developer error
        raise ValueError(f"Unknown scheduler '{name}'") from exc
    config = scheduler_config or {}
    return factory(**config)


__all__ = [
    "build_norm",
    "build_metric",
    "build_callback",
    "build_scheduler",
]
