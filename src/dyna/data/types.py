from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum


class ConcatMode(Enum):
    NO_CONCAT = "NO_CONCAT"
    CONCAT_TOKENS = "CONCAT_TOKENS"


@dataclass
class DataSplitConstants:
    hf_split: str
    folder_split: str
    raw_samples: int | None
    truncated_samples: int | None


@dataclass
class DatasetConstants:
    chars_per_sample: int
    chars_per_token: int
    splits: dict[str, DataSplitConstants] = field(default_factory=dict)

    def __iter__(self) -> Iterator[DataSplitConstants]:
        yield from self.splits.values()


class TrainSmallConstants(DataSplitConstants):
    def __init__(
        self,
        hf_split: str = "train",
        folder_split: str = "train_small",
        raw_samples: int = 100000,
        truncated_samples: int = 100000,
    ):
        """Initialize TrainSmallConstants with default values."""
        super().__init__(hf_split, folder_split, raw_samples, truncated_samples)


class ValSmallConstants(DataSplitConstants):
    def __init__(
        self,
        hf_split: str = "validation",
        folder_split: str = "val_small",
        raw_samples: int = 10000,
        truncated_samples: int = 10000,
    ):
        """Initialize ValSmallConstants with default values."""
        super().__init__(hf_split, folder_split, raw_samples, truncated_samples)


class ValXSmallConstants(DataSplitConstants):
    def __init__(
        self,
        hf_split: str = "validation",
        folder_split: str = "val_xsmall",
        raw_samples: int = 3000,
        truncated_samples: int = 3000,
    ):
        """Initialize ValXSmallConstants with default values."""
        super().__init__(hf_split, folder_split, raw_samples, truncated_samples)
