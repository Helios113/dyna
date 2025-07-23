import inspect
from itertools import islice
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)
import os
import numpy as np
import torch
from composer.core.data_spec import DataSpec
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


SUPPORTED_MDS_ENCODING_TYPES = [
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
]

def stream_remote_local_validate(
    remote: Optional[str],
    local: Optional[str],
    split: Optional[str],
):
    """Check that, if needed, the local/split directory exists.

    Args:
        remote (Optional[str]): Remote path to the dataset.
        local (Optional[str]): Local path to the dataset.
        split (Optional[str]): Subdirectory specifying which dataset split to use, if any.
    """
    if remote is None or (local == remote):
        if local is not None and os.path.isdir(local):
            contents = set(os.listdir(local))
            if split is not None and split not in contents:
                raise ValueError(
                    f'Local directory {local} does not contain split {split}',
                )

class StreamingTextDataset(StreamingDataset):
    """Generic text dataset using MosaicML's StreamingDataset.

    Args:
        tokenizer (Tokenizer): HuggingFace tokenizer to
            tokenize samples.
        max_seq_len (int): The max sequence length of each sample.
        token_encoding_type (str): The encoding type of the tokenized samples. This is only used
            for legacy datasets that have been written directly as 'bytes' instead of numpy
            arrays. Types are auto-inferred for numpy arrays. Defaults to 'int64'.
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            `False``.
        epoch_size (Union[int, str], optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. If ``None``, its value is set to ``8 * batch_size``. Defaults to ``None``.
        cache_limit (Union[int, str], optional) - Maximum size in bytes of this StreamingDataset's
            shard cache. Before downloading a shard, the least recently used resident shard(s) may
            be evicted (deleted from the local cache) in order to stay under the limit. Set to None
            to disable shard eviction. Supports integer bytes as well as string human-readable
            bytes (e.g., 100b, 64kb, 77mb, and so on). Defaults to None.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. If ``None``, this is interpreted as 64 times the number of physical
            nodes of the initial run if ``shuffle_algo`` is ``py1s`` or ``py2s``, and simply the
            number of physical nodes of the initial run otherwise. Defaults to ``None``.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1e``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int, optional): Unit of shuffle. A canonical node's samples are split
            into blocks of this size, and samples within each block are shuffled. If ``None``, its
            value is calculated as ``max(4_000_000 // num_canonical_nodes), 1 << 18)``. Defaults to
            ``None``.
        sampling_method (str): Which sampling method to use, either ``balanced`` or ``fixed``.
            Defaults to ``balanced``.
        sampling_granularity (int): When picking samples for a stream's final partial repeat,
            how many samples to pick from the same shard at a time (``1`` for evenly balanced
            across shards, ``1000`` to pick 1000 samples from the same shard at a time, etc).
            Defaults to ``1``.
        batching_method (str): Which batching method to use, either ``random``, ``stratified``, or
            ``per_stream``. Defaults to ``random``.
        allow_unsafe_types (bool): If a shard contains Pickle, which allows arbitrary code
            execution during deserialization, whether to keep going if ``True`` or raise an error
            if ``False``. Defaults to ``False``.
        replication (int, optional): Determines how many consecutive devices will receive the same
            samples. Useful for training with tensor or sequence parallelism, where multiple
            devices need to see the same partition of the dataset. Defaults to ``None``.
        stream_name (str): The name of the Stream to use which is registered in
            streaming.base.stream.streams_registry. Defaults to ``stream``.
        stream_config (dict[str, Any]): Additional arguments to pass to the Stream constructor.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
        token_encoding_type: str = 'int64',
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        download_retry: int = 2,
        download_timeout: float = 60,
        validate_hash: Optional[str] = None,
        keep_zip: bool = False,
        epoch_size: Optional[Union[int, str]] = None,
        predownload: Optional[int] = None,
        cache_limit: Optional[Union[int, str]] = None,
        partition_algo: str = 'relaxed',
        num_canonical_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        shuffle_algo: str = 'py1e',
        shuffle_seed: int = 9176,
        shuffle_block_size: Optional[int] = None,
        sampling_method: str = 'balanced',
        sampling_granularity: int = 1,
        batching_method: str = 'random',
        allow_unsafe_types: bool = False,
        replication: Optional[int] = None,
        stream_name: str = 'stream',
        stream_config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):

        if token_encoding_type not in SUPPORTED_MDS_ENCODING_TYPES:
            raise ValueError(
                f'The token_encoding_type must be one of {SUPPORTED_MDS_ENCODING_TYPES}, but got {token_encoding_type}',
            )
        self.token_encoding_type = token_encoding_type

        if streams is None:
            stream_remote_local_validate(remote, local, split)
        else:
            for stream in streams:
                stream_remote_local_validate(
                    stream.remote,
                    stream.local,
                    split,
                )

        # TODO: discover where yamls are being converted incorrect, but temporary workaround
        if isinstance(shuffle_block_size, float):
            shuffle_block_size = int(shuffle_block_size)

        # Build Dataset
        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            epoch_size=epoch_size,
            predownload=predownload,
            cache_limit=cache_limit,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
            shuffle_block_size=shuffle_block_size,
            sampling_method=sampling_method,
            sampling_granularity=sampling_granularity,
            batching_method=batching_method,
            allow_unsafe_types=allow_unsafe_types,
            replication=replication,
            stream_name=stream_name,
            stream_config=stream_config,
            **kwargs,
        )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample: Mapping) -> dict[str, torch.Tensor]:
        if self.tokenizer.pad_token is None:
            raise RuntimeError(
                'If tokenizing on-the-fly, tokenizer must have a pad_token_id',
            )
        tokens = self.tokenizer(
            text_sample['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_len,
        )
        # Ensure 'input_ids' is a tensor
        input_ids = torch.tensor(tokens['input_ids'], dtype=torch.int64)
        return {'input_ids': input_ids}

    def _read_binary_tokenized_sample(
        self,
        sample: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if isinstance(sample['tokens'], np.ndarray):
            input_ids = torch.from_numpy(
                sample['tokens'][:self.max_seq_len].copy(),
            ).to(torch.int64)
        else:
            input_ids = torch.from_numpy(
                np.frombuffer(
                    sample['tokens'],
                    dtype=getattr(np, self.token_encoding_type),
                )[:self.max_seq_len].copy(),
            ).to(torch.int64)
        return {'input_ids': input_ids}

    # How to process a sample
    def __getitem__(self,
                    idx: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        if 'text' in sample:
            token_sample = self._tokenize(sample)
        elif 'tokens' in sample:
            token_sample = self._read_binary_tokenized_sample(sample)
        else:
            raise RuntimeError(
                'StreamingTextDataset needs samples to have a `text` or `tokens` column',
            )
        return token_sample
    
    
    
# class ConcatenatedSequenceCollatorWrapper:
#     """Collator wrapper to add sequence_id to batch."""

