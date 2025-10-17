import os

import torch
from generate_standard_inputs import generate_standard_inputs

from dyna.model import _generate_attention_mask, _generate_source_len_mask


def test_source_len_mask():
    # --- 1. Basic Shape Validation ---
    # Generate a standard input tensor
    inputs = generate_standard_inputs()

    # Generate the attention mask
    mask = _generate_attention_mask(inputs, 0)
    mask = _generate_source_len_mask(mask)

    assert inputs.dim() == 2
    assert mask.dim() == 2

    assert inputs.shape[0] == mask.shape[0] and inputs.shape[1] == mask.shape[1]

    batch_size = inputs.shape[0]

    # --- 2. Iterate Through Batch ---
    for b_idx in range(batch_size):
        input_seq = inputs[b_idx]
        source_len_mask = mask[b_idx]

        # Find the indices of all padding tokens (zeros) in the sequence
        zero_indices = torch.where(input_seq == 0)[0]
        if zero_indices.numel() == 0:
            # No zeros in this sequence, so no rules to check.
            continue

        # --- 3. Check the Rule for Each Zero ---
        n = zero_indices.numel()
        for idx in range(n):
            if idx == 0:
                cur_len = zero_indices[idx]
            else:
                cur_len = zero_indices[idx] - zero_indices[idx - 1]
            assert cur_len - source_len_mask[zero_indices[idx]] <= 1


# enable test_attention_mask if we run this not from pytest
if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_source_len_mask()
