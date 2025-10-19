import hashlib
import os

import torch
from generate_standard_inputs import generate_standard_inputs
from generate_standard_lm import generate_standard_lm

from dyna.model import DynaLM
from dyna.transition import BasicFFN

UP_OUTPUT_HASH = "92f694384f5bb74086f8413d3c9f3e380cde3f316e99d7c2b618f9c342fcef23"
ACTIVATION_OUTPUT_HASH = (
    "56306a193b0e8130896b92522ce9eae8599ebc2e8b49d73ea0c3b9164f33d57b"
)
DOWN_OUTPUT_HASH = "df6e86bd9de4b5e8ac2d34bd1b32c6ea3aa1bae39de0f8264716e6460ef685cc"


def test_basic_ffn():
    input = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, _, _ = model.embedding_stage(input, None, None, None)
    d_model = model.config.d_model
    d_ffn = model.config.d_ffn
    torch.manual_seed(42)

    ffn: BasicFFN = BasicFFN(d_model, d_ffn)
    collector = {}
    ffn_output, _ = ffn(embedding, None, collector)

    EXPECTED_KEYS = [
        "basic_ffn_proj_up",
        "basic_ffn_activation",
        "basic_ffn_proj_down",
    ]
    assert set(collector.keys()) == set(EXPECTED_KEYS), (
        f"Collector keys mismatch.\nExpected: {EXPECTED_KEYS}\n"
        f"Got: {list(collector.keys())}"
    )

    # print(f"UP_OUTPUT_HASH = \"{hashlib.sha256(
    #     collector['basic_ffn_proj_up'].detach().numpy().tobytes()
    # ).hexdigest()}\"")
    # print(f"ACTIVATION_OUTPUT_HASH = \"{hashlib.sha256(
    #     collector['basic_ffn_activation'].detach().numpy().tobytes()
    # ).hexdigest()}\"")
    # print(f"DOWN_OUTPUT_HASH = \"{hashlib.sha256(
    #     collector['basic_ffn_proj_down'].detach().numpy().tobytes()
    # ).hexdigest()}\"")

    # No non-deterministic operations so hashing will work
    up_output_hash = hashlib.sha256(
        collector["basic_ffn_proj_up"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        up_output_hash == UP_OUTPUT_HASH
    ), f"Projection Up Output Hash Mismatch: {up_output_hash}"

    activation_output_hash = hashlib.sha256(
        collector["basic_ffn_activation"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        activation_output_hash == ACTIVATION_OUTPUT_HASH
    ), f"Activation Output Hash Mismatch: {activation_output_hash}"

    down_output_hash = hashlib.sha256(
        collector["basic_ffn_proj_down"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        down_output_hash == DOWN_OUTPUT_HASH
    ), f"Projection Down Output Hash Mismatch: {down_output_hash}"

    # Same as down_output_hash.
    hash_value = hashlib.sha256(ffn_output.detach().numpy().tobytes()).hexdigest()
    assert hash_value == DOWN_OUTPUT_HASH, f"Final Output Hash Mismatch: {hash_value}"


if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_basic_ffn()
