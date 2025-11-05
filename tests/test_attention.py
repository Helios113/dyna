import os

import torch

from dyna.model import _generate_attention_mask

from .generate_standard_inputs import generate_standard_inputs


def test_attention_mask():
    # --- 1. Basic Shape Validation ---
    # Generate a standard input tensor
    inputs = generate_standard_inputs()

    # Generate the attention mask
    mask = _generate_attention_mask(inputs, 0)

    assert inputs.dim() == 2
    assert mask.dim() == 4

    assert (
        inputs.shape[0] == mask.shape[0]
        and inputs.shape[1] == mask.shape[2]
        and inputs.shape[1] == mask.shape[3]
    )

    batch_size = inputs.shape[0]

    # --- 2. Iterate Through Batch ---
    for b_idx in range(batch_size):
        input_seq = inputs[b_idx]
        attention_mask = mask[b_idx]

        # Find the indices of all padding tokens (zeros) in the sequence
        zero_indices = torch.where(input_seq == 0)[0]
        if zero_indices.numel() == 0:
            # No zeros in this sequence, so no rules to check.
            continue

        # --- 3. Check the Rule for Each Zero ---
        for z_idx_tensor in zero_indices:
            z_idx = z_idx_tensor.item()

            rows_after_z = slice(z_idx + 1, None)
            rows_before_z = slice(None, z_idx)
            cols_before_z = slice(None, z_idx)
            cols_after_z = slice(z_idx + 1, None)

            # This slice can be empty if the zero is at the end, which is fine
            sub_mask = attention_mask[rows_after_z, cols_before_z]
            assert torch.all(sub_mask == 0)

            # Check the rule for the zero
            sub_mask = attention_mask[rows_before_z, cols_after_z]
            assert torch.any(sub_mask == 1)


# enable test_attention_mask if we run this not from pytest
if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_attention_mask()
