from __future__ import annotations

import json
import os
from typing import Any

from llmfoundry.utils.builders import build_tokenizer
from streaming import MDSWriter
from tqdm import tqdm

from dyna.data.constants import CONSTS
from dyna.data.hf_build import build_dataloader, build_hf_dataset
from dyna.data.io_mds import generate_samples
from dyna.data.types import ConcatMode, DatasetConstants


def _est_progress_denominator(
    total_samples: int,
    chars_per_sample: int,
    chars_per_token: int,
    mode: ConcatMode,
    max_length: int,
) -> int:
    est_tokens_per_sample = chars_per_sample // chars_per_token
    if mode == ConcatMode.NO_CONCAT:
        return total_samples
    elif mode == ConcatMode.CONCAT_TOKENS:
        return total_samples * est_tokens_per_sample // max_length
    else:
        return total_samples


def convert_dataset_hf(
    dataset: str,
    dataset_constants: DatasetConstants,
    data_subset: str | None,
    splits: list[str],
    out_root: str,
    compression: str | None,
    concat_tokens: int | None,
    tokenizer: str | None,
    tokenizer_kwargs: dict[str, Any],
    bos_text: str,
    eos_text: str,
    no_wrap: bool,
    num_workers: int | None,
) -> None:
    """Converts HuggingFace datasets to MDS format."""
    mode = ConcatMode.CONCAT_TOKENS
    built_tokenizer = build_tokenizer(tokenizer, tokenizer_kwargs)
    # we will enforce length, so suppress warnings about sequences too long
    # for the model
    built_tokenizer.model_max_length = int(1e30)

    columns = {"tokens": "ndarray:int32"}

    for split_name in splits:
        try:
            split = dataset_constants.splits[split_name]
        except KeyError as e:
            raise KeyError(f"Constants not defined for split {split_name}.") from e
        hf_split = split.hf_split
        folder_split = split.folder_split
        expected_num_samples = split.raw_samples
        truncate_num_samples = split.truncated_samples

        # Only generate the splits requested
        if folder_split not in splits:
            continue

        # Get samples
        hf_dataset = build_hf_dataset(
            dataset_name=dataset,
            data_subset=data_subset,
            split=hf_split,
            mode=mode,
            max_length=concat_tokens,
            bos_text=bos_text,
            eos_text=eos_text,
            no_wrap=no_wrap,
            tokenizer=built_tokenizer,
        )
        loader = build_dataloader(
            dataset=hf_dataset,
            batch_size=512,
            num_workers=num_workers,
        )
        samples = generate_samples(
            loader,
            truncate_num_samples=truncate_num_samples,
        )

        if expected_num_samples is not None and concat_tokens is not None:
            denominator = (
                truncate_num_samples
                if truncate_num_samples is not None
                else _est_progress_denominator(
                    total_samples=expected_num_samples,
                    chars_per_sample=dataset_constants.chars_per_sample,
                    chars_per_token=dataset_constants.chars_per_token,
                    mode=mode,
                    max_length=concat_tokens,
                )
            )
        else:
            denominator = None

        # Write samples
        print(f"Converting {folder_split} to MDS format...")
        print(
            "Note: the progress bar is based on the dataset length before "
            "tokenization, and may finish at a value before 100%.",
        )
        with MDSWriter(
            columns=columns,
            out=os.path.join(out_root, folder_split),
            compression=compression,
        ) as out:
            if denominator is not None:
                for sample in tqdm(
                    samples,
                    desc=folder_split,
                    total=denominator,
                ):
                    out.write(sample)
            else:
                for sample in tqdm(samples, desc=folder_split):
                    out.write(sample)


def convert_dataset_hf_from_args(
    dataset: str,
    data_subset: str | None,
    splits: list[str],
    out_root: str,
    compression: str | None,
    concat_tokens: int | None,
    tokenizer: str | None,
    tokenizer_kwargs: str | None,
    bos_text: str | None,
    eos_text: str | None,
    no_wrap: bool,
    num_workers: int | None,
) -> None:
    """A wrapper for convert_dataset_hf that parses arguments."""
    os.environ["WORLD_SIZE"] = "1"

    parsed_tokenizer_kwargs = (
        json.loads(tokenizer_kwargs) if tokenizer_kwargs else {}
    )

    if (
        os.path.isdir(out_root)
        and len(set(os.listdir(out_root)).intersection(set(splits))) > 0
    ):
        raise ValueError(
            f"--out_root={out_root} contains {os.listdir(out_root)} which "
            f"cannot overlap with the requested splits {splits}.",
        )

    # Make sure we have needed concat options
    if (
        concat_tokens is not None
        and isinstance(concat_tokens, int)
        and tokenizer is None
    ):
        raise ValueError("When setting --concat_tokens, you must specify a --tokenizer")

    # Lookup dataset constants by dataset name
    if dataset not in CONSTS:
        raise KeyError(
            f"No constants configured for dataset {dataset}. "
            f"Available: {list(CONSTS.keys())}"
        )
    dataset_constants = CONSTS[dataset]

    # now that we have validated them, change BOS/EOS to strings and convert
    convert_dataset_hf(
        dataset=dataset,
        dataset_constants=dataset_constants,
        data_subset=data_subset,
        splits=splits,
        out_root=out_root,
        compression=compression,
        concat_tokens=concat_tokens,
        tokenizer=tokenizer,
        tokenizer_kwargs=parsed_tokenizer_kwargs,
        bos_text=bos_text if bos_text else "",
        eos_text=eos_text if eos_text else "",
        no_wrap=no_wrap,
        num_workers=num_workers,
    )
