from __future__ import annotations

import platform

import datasets as hf_datasets
import psutil
from llmfoundry.data import ConcatTokensDataset, NoConcatDataset
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from dyna.data.types import ConcatMode


def build_hf_dataset(
    dataset_name: str,
    split: str,
    mode: ConcatMode,
    max_length: int | None = None,
    bos_text: str = "",
    eos_text: str = "",
    no_wrap: bool = False,
    tokenizer: PreTrainedTokenizerBase | None = None,
    data_subset: str | None = None,
) -> IterableDataset:
    hf_dataset = hf_datasets.load_dataset(
        path=dataset_name,
        name=data_subset,
        split=split,
        streaming=True,
    )
    if mode == ConcatMode.NO_CONCAT:
        dataset = NoConcatDataset(hf_dataset)
    else:
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(f"{tokenizer=} must be of type PreTrainedTokenizerBase")
        if max_length is None:
            raise ValueError("max_length must be set.")
        if bos_text + eos_text == "":
            test_tokens = tokenizer("test")
            if (
                test_tokens["input_ids"][0] != tokenizer.bos_token_id
                and test_tokens["input_ids"][-1] != tokenizer.eos_token_id
            ):
                tok_error_msg = "This tokenizer does not insert an EOS nor BOS token. "
                tok_error_msg += (
                    "Concatenating with this tokenizer will result in sequences being "
                )
                tok_error_msg += (
                    "attached without a separating token. Please use another tokenizer, "
                )
                tok_error_msg += (
                    "such as facebook/opt-125m, or specify EOS/BOS text with e.g. "
                )
                tok_error_msg += "--bos_text=<|endoftext|>."
                raise ValueError(tok_error_msg)
        dataset = ConcatTokensDataset(
            hf_dataset=hf_dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            bos_text=bos_text,
            eos_text=eos_text,
            no_wrap=no_wrap,
        )
    return dataset


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int | None,
) -> DataLoader:
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        if "linux" in platform.platform().lower() or "macos" in platform.platform().lower():
            num_workers = max(1, psutil.cpu_count())
        else:
            num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many
    # samples as it can, up to the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value
    # for prefetch_factor, which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size // num_workers) if num_workers > 0 else 2

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
