from __future__ import annotations

import os

SUPPORTED_MDS_ENCODING_TYPES = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]


def stream_remote_local_validate(
    remote: str | None,
    local: str | None,
    split: str | None,
) -> None:
    """Ensure local dataset directories exist when streaming without remote."""

    if remote is None or (local == remote):
        if local is not None and os.path.isdir(local):
            contents = set(os.listdir(local))
            if split is not None and split not in contents:
                raise ValueError(
                    f"Local directory {local} does not contain split {split}",
                )
