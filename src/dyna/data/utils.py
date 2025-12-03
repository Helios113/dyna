from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Mapping

import torch
import transformers
from composer.core.data_spec import DataSpec
from composer.core.types import Batch
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import PreTrainedTokenizerBase

from dyna.data.text_data import ConcatenatedSequenceCollatorWrapper
import random

log = logging.getLogger(__name__)
# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
"""Utility and helper functions for datasets."""


CROSS_ENTROPY_IGNORE_INDEX = -100


class LossGeneratingTokensCollatorWrapper:
    """Inject per-sample token counts so downstream microbatching stays exact."""

    def __init__(
        self,
        base_collator: Callable[[list[Any]], Mapping[str, torch.Tensor]],
        token_counting_func: Callable[[Batch], int | dict[str, int]],
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
    device_batch_size: int | float,
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
) -> Callable[[Batch], int | dict[str, int]]:
    def get_num_tokens_in_batch(batch: Batch) -> int | dict[str, int]:
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
    transformers.DataCollatorForLanguageModeling
    | ConcatenatedSequenceCollatorWrapper
    | LossGeneratingTokensCollatorWrapper,
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
    dl: Iterable[Any] | TorchDataLoader,
    dataset_cfg: dict[str, Any],
) -> DataSpec:
    del dataset_cfg
    token_counting_func = get_tokens_per_batch_func()
    return DataSpec(
        dataloader=dl,
        get_num_tokens_in_batch=token_counting_func,
    )



log = logging.getLogger(__name__)


def strip_data(example: dict) -> dict:
    """Remove white space from the begging and end of string values in a.

    dictionary.

    Args:
        example: Dictionary to be stripped

    Returns:
        dict: The same dictionary with .strip() applied to any value in the dict that is a string
    """
    return {
        k: v.strip() if isinstance(v, str) else v for k, v in example.items()
    }


def tokenizer_needs_prefix_space(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> bool:
    """Test for whether a prefix space is needed before the continuation.

    Sentencepiece tokenization should not have a prefix space, but gpt2 style
    BPE should.

    Args:
        tokenizer: Tokenizer to test

    Returns:
        bool: Whether or not the tokenizer needs a prefix space
    """
    test_tokens = tokenizer(' a', add_special_tokens=False)['input_ids']
    assert isinstance(test_tokens, list)
    return len(test_tokens) == 1


def trim_context(
    context_enc: list,
    continuation_enc: list,
    max_seq_len: int,
) -> list:
    """Trims a list of tokens down to `max_seq_len` if the length of the list.

    plus the continuation is more than `max_seq_len`. It will always trim tokens
    from the left, i.e. tokens at the beginning of the context will be removed.

    Args:
        context_enc (list): List of tokens in the context
        continuation_enc (list): List of tokens in the continuation
        max_seq_len (int): Maximum length the model can ingest

    Returns:
        list: The encoded context trimmed from the left
    """
    if len(continuation_enc) + len(context_enc) > max_seq_len:
        context_max_subseq_len = max_seq_len - len(continuation_enc)

        if context_max_subseq_len < 0:
            # can't support continuations which are longer than the max seq len
            raise Exception(
                f'Dataset included continuation longer than the max seq len',
            )

        # clip from the end
        context_enc = context_enc[-(context_max_subseq_len):]
    return context_enc


def get_continuation_span(
    context_enc: list,
    continuation_enc: list,
) -> torch.Tensor:
    """Gets the list of indices of the continuation tokens for language.

    modeling.

    or generation tasks.

    Args:
        context_enc (list): List of context tokens
        continuation_enc (list): List of continuation tokens

    Returns:
        torch.tensor: A tensor containing indices corresponding to continuation tokens
    """
    return torch.tensor(
        range(len(context_enc),
              len(context_enc) + len(continuation_enc)),
    )


def make_padded_input(
    context_enc: list,
    continuation_enc: list,
    max_seq_len: int,
    pad_tok_id: int,
    padding_side: str = 'right',
) -> torch.Tensor:
    """Takes an encoded context and continuation and clips the beginning of the.

    context if they're too long. Adds the padding token to the specified side.

    Args:
        context_enc (List): The encoded input to the model
        continuation_enc (List): The encoded desired output for the example
        max_seq_len (int): Maximum length sequences can be
        pad_tok_id (int): The token id we pad with
        padding_side (str): Which side to pad the context on. Can be 'right' or 'left

    Returns:
        input (torch.tensor): The padded and encoded context
        continuation_span (torch.tensor): The _inclusive_ range of indices corresponding to the continuation
    """
    inp = torch.tensor(
        (context_enc + continuation_enc),
        dtype=torch.long,
    )
    (inp_len,) = inp.shape

    # Sometimes tokenizers that have neither a pad_tok_id or eos_tok_id will pass None in as the padding
    # token and cause errors
    if not isinstance(pad_tok_id, int):
        raise ValueError(
            f'`pad_tok_id` must be an integer. Found {type(pad_tok_id)} instead',
        )
    # pad length from seq to padding_length
    if padding_side == 'right':
        inp = torch.cat(
            [
                inp,  # [seq]
                torch.LongTensor((max_seq_len - inp_len) * [pad_tok_id]),
            ],
            dim=0,
        )
    elif padding_side == 'left':
        inp = torch.cat(
            [
                torch.LongTensor((max_seq_len - inp_len) * [pad_tok_id]),
                inp,  # [seq]
            ],
            dim=0,
        )
    else:
        raise ValueError(
            f"Unknown padding_side {padding_side}. padding_side must be either 'left' or 'right'",
        )

    return inp


def convert_tokens_to_tensors(batch: dict,
                              tokenize_labels: bool) -> dict[str, Any]:
    """HF Datasets converts tensors into lists when we store them, and we don't.

    want to use `type='torch'` because some content in the dataset, like
    generation args or single ints, should not be converted.

    Here, we convert those lists of tokens back into tensors in order to feed them into the model.

    Args:
        batch (dict): A dictionary of batched inputs
        tokenize_labels (bool): Whether or not the labels are tokenized (and need to be stacked)

    Returns:
        dict: The batch with torch tensors in the corresponding keys instead of lists of lists
    """
    batch['input_ids'] = torch.stack(
        list(map(torch.tensor, batch['input_ids'])),
    )
    if tokenize_labels:
        batch['labels'] = torch.stack(list(map(torch.tensor, batch['labels'])))
        batch['continuation_indices'] = list(
            map(torch.tensor, batch['continuation_indices']),
        )
    return batch


def get_fewshot_sample_idxs(
    dataset_size: int,
    num_fewshot: int,
    example_idx: int,
    rng: random.Random,
) -> set[int]:
    """Samples indices without replacement. If num_fewshot exceeds the number.

    of unique examples in the dataset, then we will have fewer than num_fewshot examples in context.

    Args:
        dataset_size (int): Length of the dataset
        num_fewshot (int): Number of examples to prepend
        example_idx (int): Current example's index (excluded from fewshot choices)
        rng (random.Random): RNG for repeatable sample selection

    Returns:
        list: Indices of the examples chosen for fewshot selection
    """
    num_fewshot = min(dataset_size - 1, num_fewshot)
    fewshot_idxs = set(rng.sample(range(0, dataset_size), num_fewshot))

    if example_idx in fewshot_idxs:
        fewshot_idxs.remove(example_idx)
        if len(fewshot_idxs) >= dataset_size - 1:
            return fewshot_idxs

        replacement_sample = rng.choice(range(0, dataset_size))
        while replacement_sample in fewshot_idxs or replacement_sample == example_idx:
            replacement_sample = rng.choice(range(0, dataset_size))
        fewshot_idxs.add(replacement_sample)
    return fewshot_idxs


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence.

    Slightly modified from: https://github.com/EleutherAI/lm-evaluation-harness/blob/78545d42f2ca95c6fe0ed220d456eeb94f4485e9/lm_eval/utils.py#L614-L649
    """

    def __init__(
        self,
        stop_sequence: str,
        tokenizer: transformers.PreTrainedTokenizerBase,
        batch_size: int,
    ) -> None:
        self.done_tracker = [False] * batch_size
        self.stop_sequence = stop_sequence
        self.stop_sequence_ids = tokenizer.encode(
            stop_sequence,
            add_special_tokens=False,
        )

        # sentence piece tokenizers add a superfluous underline token before string-initial \n
        # that throws off our calculation of the stop sequence length
        # so we remove any token ids that produce empty strings
        self.stop_sequence_ids = [
            id for id in self.stop_sequence_ids if tokenizer.decode(id) != ''
        ]

        # we look back for 1 more token than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        self.stop_sequence_id_len = len(self.stop_sequence_ids) + 1
        self.tokenizer = tokenizer

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: Optional[torch.FloatTensor] = None,
        **kwargs: dict[str, Any],
    ) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, :][:, -self.stop_sequence_id_len:]
        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
        for i, done in enumerate(self.done_tracker):
            if i >= len(lookback_tokens_batch):
                # The last batch of a dataset may be smaller than `batch_size`
                # Automatically set those indices in the done_tracker to True
                # since those indices don't show up in the current batch
                self.done_tracker[i] = True
                break
            elif not done:
                self.done_tracker[
                    i] = self.stop_sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizerBase,
    stop_sequences: list[str],
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList([
        *[
            MultiTokenEOSCriteria(sequence, tokenizer, batch_size)
            for sequence in stop_sequences
        ],
    ])