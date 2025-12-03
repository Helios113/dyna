from dyna.data.text_data import (
    ConcatenatedSequenceCollatorWrapper,
    StreamingTextDataset,
    build_text_dataloader,
)
from dyna.data.utils import (
    LossGeneratingTokensCollatorWrapper,
    get_data_spec,
    get_text_collator,
    validate_ds_replication,
)
from dyna.registry import (
    collators,
    data_specs,
    dataset_replication_validators,
)

if "dataset_replication_validator" not in dataset_replication_validators:
    dataset_replication_validators.register(
        "dataset_replication_validator",
        validate_ds_replication,
    )

if "text_collator" not in collators:
    collators.register("text_collator", get_text_collator)

if "data_spec" not in data_specs:
    data_specs.register("data_spec", get_data_spec)

__all__ = [
    "ConcatenatedSequenceCollatorWrapper",
    "LossGeneratingTokensCollatorWrapper",
    "StreamingTextDataset",
    "build_text_dataloader",
    "get_data_spec",
    "get_text_collator",
    "validate_ds_replication",
]
