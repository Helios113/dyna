import hashlib
import os

from generate_standard_inputs import generate_standard_inputs
from generate_standard_lm import generate_standard_lm

TRANSFORMER_HEAD_OUTPUT_HASH = "missing"
TRANSFORMER_REINJECTION_EMBEDDINGS_HASH = "missing"
TRANSFORMER_LAYER_INDEX_HASH = "missing"
TRANSFORMER_BODY_OUTPUT_HASH = "missing"
TRANSFORMER_BODY_ENERGY_PER_SAMPLE_HASH = "missing"
TRANSFORMER_BODY_LAYER_INDEX_HASH = "missing"
TRANSFORMER_TAIL_OUTPUT_HASH = "missing"


def test_transformer():
    _ = generate_standard_inputs()
    _ = generate_standard_lm()

    # Initialization
    collector = {}
    # Output
    # transformer_output = None

    EXPECTED_KEYS = [
        "transformer_head_output",
        "transformer_head_reinjection_embeddings",
        "transformer_head_layer_index",
        "transformer_body_output",
        "transformer_body_energy_per_sample",
        "transformer_body_layer_index",
        "transformer_tail_output",
    ]
    assert set(collector.keys()) == set(EXPECTED_KEYS), (
        f"Collector keys mismatch.\nExpected: {EXPECTED_KEYS}\n"
        f"Got: {list(collector.keys())}"
    )

    # Uncomment to capture hashes
    print(f"HEAD_OUTPUT_HASH = \"{hashlib.sha256(
        collector['transformer_head_output'].detach().numpy().tobytes()
    ).hexdigest()}\"")
    print(f"REINJECTION_EMBEDDINGS_HASH = \"{hashlib.sha256(
        collector['transformer_head_reinjection_embeddings'].detach().numpy().tobytes()
    ).hexdigest()}\"")
    print(f"LAYER_INDEX_HASH = \"{hashlib.sha256(
        collector['transformer_head_layer_index'].detach().numpy().tobytes()
    ).hexdigest()}\"")
    print(f"BODY_OUTPUT_HASH = \"{hashlib.sha256(
        collector['transformer_body_output'].detach().numpy().tobytes()
    ).hexdigest()}\"")
    print(f"BODY_ENERGY_PER_SAMPLE_HASH = \"{hashlib.sha256(
        collector['transformer_body_energy_per_sample'].detach().numpy().tobytes()
    ).hexdigest()}\"")
    print(f"BODY_LAYER_INDEX_HASH = \"{hashlib.sha256(
        collector['transformer_body_layer_index'].detach().numpy().tobytes()
    ).hexdigest()}\"")
    print(f"TAIL_OUTPUT_HASH = \"{hashlib.sha256(
        collector['transformer_tail_output'].detach().numpy().tobytes()
    ).hexdigest()}\"")

    head_output_hash = hashlib.sha256(
        collector["transformer_head_output"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        head_output_hash == TRANSFORMER_HEAD_OUTPUT_HASH
    ), f"Head Output Hash Mismatch: {head_output_hash}"

    reinjection_embeddings_hash = hashlib.sha256(
        collector["transformer_head_reinjection_embeddings"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        reinjection_embeddings_hash == TRANSFORMER_REINJECTION_EMBEDDINGS_HASH
    ), f"Reinjection Embeddings Hash Mismatch: {reinjection_embeddings_hash}"

    layer_index_hash = hashlib.sha256(
        collector["transformer_head_layer_index"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        layer_index_hash == TRANSFORMER_LAYER_INDEX_HASH
    ), f"Layer Index Hash Mismatch: {layer_index_hash}"

    body_output_hash = hashlib.sha256(
        collector["transformer_body_output"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        body_output_hash == TRANSFORMER_BODY_OUTPUT_HASH
    ), f"Body Output Hash Mismatch: {body_output_hash}"

    body_energy_per_sample_hash = hashlib.sha256(
        collector["transformer_body_energy_per_sample"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        body_energy_per_sample_hash == TRANSFORMER_BODY_ENERGY_PER_SAMPLE_HASH
    ), f"Body Energy Per Sample Hash Mismatch: {body_energy_per_sample_hash}"

    body_layer_index_hash = hashlib.sha256(
        collector["transformer_body_layer_index"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        body_layer_index_hash == TRANSFORMER_BODY_LAYER_INDEX_HASH
    ), f"Body Layer Index Hash Mismatch: {body_layer_index_hash}"

    tail_output_hash = hashlib.sha256(
        collector["transformer_tail_output"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        tail_output_hash == TRANSFORMER_TAIL_OUTPUT_HASH
    ), f"Tail Output Hash Mismatch: {tail_output_hash}"

    # Smae as tail_output_hash.
    # final_output_hash = hashlib.sha256(
    #    transformer_output.detach().numpy().tobytes()
    # ).hexdigest()
    # assert (
    #    final_output_hash == TRANSFORMER_TAIL_OUTPUT_HASH
    # ), f"Final Output Hash Mismatch: {final_output_hash}"


if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_transformer()
