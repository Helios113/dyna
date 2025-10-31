import hashlib
import os

import torch
from generate_standard_inputs import generate_standard_inputs
from generate_standard_lm import generate_standard_lm

from dyna.model import DynaLM

UP_OUTPUT_HASH = "cdc08b3d40261b542d47a0f18df3f44a528ce212a71f5ab7afb922ebb41acdc5"
ACTIVATION_OUTPUT_HASH = (
    "f30e8073d7ddf123da6a855267a35379a5de3300fe1aaa250e487d55e84bdbba"
)
DOWN_OUTPUT_HASH = "bdeb2c34d2b809420535b75dcc614f030bd4a2b0b1c023c09ccdadc3cafe8e62"


def test_basic_ffn():
    input = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, _, _ = model.embedding_stage(input, None, None, None)
    torch.manual_seed(42)
    ffn = model.transformer.body_layers[0].ffn
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

    print(
        f'UP_OUTPUT_HASH = "{
            hashlib.sha256(
                collector["basic_ffn_proj_up"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'ACTIVATION_OUTPUT_HASH = "{
            hashlib.sha256(
                collector["basic_ffn_activation"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'DOWN_OUTPUT_HASH = "{
            hashlib.sha256(
                collector["basic_ffn_proj_down"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )

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
