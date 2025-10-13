from collections.abc import Iterable

import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader


def generate_samples(
    loader: DataLoader,
    truncate_num_samples: int | None = None,
) -> Iterable[dict[str, bytes] | dict[str, NDArray]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {
                k: (v[idx].numpy() if isinstance(v[idx], torch.Tensor) else v[idx])
                for k, v in batch.items()
            }
