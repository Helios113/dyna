from dyna.data.types import DatasetConstants, DataSplitConstants

pileconstants = DatasetConstants(
    chars_per_sample=6212,  # Computed over validation set
    chars_per_token=4,  # OpenAI estimate
)
pileconstants.splits["train"] = DataSplitConstants(
    hf_split="train",
    folder_split="train",
    raw_samples=210607728,
    truncated_samples=None,
)
pileconstants.splits["train_small"] = DataSplitConstants(
    hf_split="train",
    folder_split="train_small",
    raw_samples=100000,
    truncated_samples=100000,
)
pileconstants.splits["val"] = DataSplitConstants(
    hf_split="validation",
    folder_split="val",
    raw_samples=214670,
    truncated_samples=None,
)
pileconstants.splits["val_small"] = DataSplitConstants(
    hf_split="validation",
    folder_split="val_small",
    raw_samples=10000,
    truncated_samples=10000,
)
pileconstants.splits["val_xsmall"] = DataSplitConstants(
    hf_split="validation",
    folder_split="val_xsmall",
    raw_samples=3000,
    truncated_samples=3000,
)

c4constants = DatasetConstants(
    chars_per_sample=2163,  # Computed over validation set
    chars_per_token=4,  # OpenAI estimate
)
c4constants.splits["train"] = DataSplitConstants(
    hf_split="train",
    folder_split="train",
    raw_samples=364868892,
    truncated_samples=None,
)
c4constants.splits["train_small"] = DataSplitConstants(
    hf_split="train",
    folder_split="train_small",
    raw_samples=100000,
    truncated_samples=100000,
)
c4constants.splits["val"] = DataSplitConstants(
    hf_split="validation",
    folder_split="val",
    raw_samples=364608,
    truncated_samples=None,
)
c4constants.splits["val_small"] = DataSplitConstants(
    hf_split="validation",
    folder_split="val_small",
    raw_samples=10000,
    truncated_samples=10000,
)
c4constants.splits["val_xsmall"] = DataSplitConstants(
    hf_split="validation",
    folder_split="val_xsmall",
    raw_samples=3000,
    truncated_samples=3000,
)
c4constants.splits["val_xxsmall"] = DataSplitConstants(
    hf_split="validation",
    folder_split="val_xxsmall",
    raw_samples=100,
    truncated_samples=100,
)

CONSTS = {
    "allenai/c4": c4constants,
    "the_pile": pileconstants,
}
