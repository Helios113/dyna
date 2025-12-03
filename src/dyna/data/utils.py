from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Mapping, Union

import torch
import transformers
from composer.core.data_spec import DataSpec
from composer.core.types import Batch
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import PreTrainedTokenizerBase

from dyna.data.text_data import ConcatenatedSequenceCollatorWrapper

log = logging.getLogger(__name__)

CROSS_ENTROPY_IGNORE_INDEX = -100


class LossGeneratingTokensCollatorWrapper:
    """Inject per-sample token counts so downstream microbatching stays exact."""

    def __init__(
        self,
        base_collator: Callable[[list[Any]], Mapping[str, torch.Tensor]],
        token_counting_func: Callable[[Batch], Union[int, dict[str, int]]],
    ) -> None:
        self.base_collator = base_collator
        self.token_counting_func = token_counting_func
        self._token_count_batch_keys = [
            "input_ids",
            "attention_mask",
            "labels",
            "decoder_attention_mask",
        ]

    def __call__(self, examples: list[Any]) -> dict[str, torch.Tensor]:
        batch = dict(self.base_collator(examples))
        total_tokens: list[int] = []
        loss_generating_tokens: list[int] = []
        num_rows = batch["input_ids"].shape[0]
        for row in range(num_rows):
            row_batch: dict[str, torch.Tensor] = {}
            for key in self._token_count_batch_keys:
                if key in batch:
                    row_batch[key] = batch[key][row : row + 1]
            num_tokens = self.token_counting_func(row_batch)
            if isinstance(num_tokens, dict):
                total_tokens.append(num_tokens["total"])
                loss_generating_tokens.append(num_tokens["loss_generating"])
            else:
                total_tokens.append(num_tokens)
                loss_generating_tokens.append(num_tokens)
        batch["total_tokens"] = total_tokens
        batch["loss_generating_tokens"] = loss_generating_tokens
        return batch


def _validate_cfg(
    dataset_cfg: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    eos_token_id = dataset_cfg.get("eos_token_id")
    bos_token_id = dataset_cfg.get("bos_token_id")

    tokenizer_eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and eos_token_id != tokenizer_eos_token_id:
        eos_msg = (
            f"Provided eos_token_id={eos_token_id} does not match tokenizer eos_token_id="
            f"{tokenizer_eos_token_id}."
        )
        if dataset_cfg.pop("override_eos_token_id_mismatch_error", False):
            log.warning(eos_msg)
        else:
            raise ValueError(
                eos_msg
                + " To override this error, set the override_eos_token_id_mismatch_error flag to True "
                + "in the dataset config section of the YAML."
            )

    tokenizer_bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None and bos_token_id != tokenizer_bos_token_id:
        bos_msg = (
            f"Provided bos_token_id={bos_token_id} does not match tokenizer bos_token_id="
            f"{tokenizer_bos_token_id}."
        )
        if dataset_cfg.pop("override_bos_token_id_mismatch_error", False):
            log.warning(bos_msg)
        else:
            raise ValueError(
                bos_msg
                + " To override this error, set the override_bos_token_id_mismatch_error flag to True "
                + "in the dataset config section of the YAML."
            )

    max_seq_len = dataset_cfg.get("max_seq_len")
    if max_seq_len is not None:
        if max_seq_len != int(max_seq_len):
            raise ValueError("max_seq_len must be an integer")
        dataset_cfg["max_seq_len"] = int(max_seq_len)


def validate_ds_replication(
    dataset_cfg: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: Union[int, float],
) -> tuple[int, int]:
    _validate_cfg(dataset_cfg, tokenizer)
    if (dataset_cfg.get("seq_parallel_replication", 1) or 1) > 1:
        raise NotImplementedError("Sequence parallelism is not supported.")
    if not isinstance(device_batch_size, int):
        raise ValueError("device_batch_size should be integer.")
    replication = dataset_cfg.get("replication", 1) or 1
    return replication, device_batch_size


def get_tokens_per_batch_func(
    decoder_only: bool = True,
) -> Callable[[Batch], Union[int, dict[str, int]]]:
    def get_num_tokens_in_batch(batch: Batch) -> Union[int, dict[str, int]]:
        if not isinstance(batch, Mapping) or (
            "attention_mask" not in batch and "input_ids" not in batch
        ):
            raise ValueError(
                "get_tokens_per_batch_func() requires a batch with an attention_mask key or an input_ids key",
            )
        if not decoder_only and "decoder_attention_mask" not in batch:
            raise ValueError(
                "get_tokens_per_batch_func() for encoder decoder requires a batch with a decoder_attention_mask key",
            )
        if "total_tokens" in batch and "loss_generating_tokens" in batch:
            return {
                "total": int(sum(batch["total_tokens"])),
                "loss_generating": int(sum(batch["loss_generating_tokens"])),
            }
        if "attention_mask" in batch:
            input_ids_tokens = int(torch.sum(batch["attention_mask"]).item())
        else:
            input_ids_tokens = batch["input_ids"].numel()
        loss_generating_tokens: int | None = None
        if "labels" in batch:
            seq_length = batch["labels"].shape[1]
            total_candidate = batch["labels"].shape[0] * (seq_length - 1)
            ignore = torch.count_nonzero(
                torch.eq(batch["labels"][..., 1:], CROSS_ENTROPY_IGNORE_INDEX)
            )
            loss_generating_tokens = int(total_candidate - ignore)
        decoder_tokens = 0
        if not decoder_only:
            decoder_tokens = int(torch.sum(batch["decoder_attention_mask"]).item())
        if loss_generating_tokens is not None:
            return {
                "total": input_ids_tokens + decoder_tokens,
                "loss_generating": loss_generating_tokens,
            }
        return input_ids_tokens + decoder_tokens

    return get_num_tokens_in_batch


def get_text_collator(
    dataloader_cfg: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    dataset_batch_size: int,
) -> tuple[
    Union[
        transformers.DataCollatorForLanguageModeling,
        ConcatenatedSequenceCollatorWrapper,
        LossGeneratingTokensCollatorWrapper,
    ],
    int,
]:
    dataset_cfg = dataloader_cfg.get("dataset")
    assert isinstance(dataset_cfg, dict)
    eos_token_id = dataset_cfg.get("eos_token_id")
    bos_token_id = dataset_cfg.get("bos_token_id")
    mlm_probability = dataset_cfg.pop("mlm_probability", None)
    collate_fn: Callable[[list[Any]], Mapping[str, torch.Tensor]] = (
        transformers.DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm_probability is not None,
            mlm_probability=mlm_probability if mlm_probability else 0,
        )
    )
    if (eos_token_id is not None) or (bos_token_id is not None):
        collate_fn = ConcatenatedSequenceCollatorWrapper(
            base_collator=collate_fn,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
        )
    collate_fn = LossGeneratingTokensCollatorWrapper(
        collate_fn,
        get_tokens_per_batch_func(),
    )
    return collate_fn, dataset_batch_size


def get_data_spec(
    dl: Union[Iterable[Any], TorchDataLoader],
    dataset_cfg: dict[str, Any],
) -> DataSpec:
    del dataset_cfg
    token_counting_func = get_tokens_per_batch_func()
    return DataSpec(
        dataloader=dl,
        get_num_tokens_in_batch=token_counting_func,
    )
