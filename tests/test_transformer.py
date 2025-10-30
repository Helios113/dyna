import hashlib
import os

from generate_standard_inputs import generate_standard_inputs
from generate_standard_lm import generate_standard_lm

from dyna.model import DynaLM

TRANSFORMER_HEAD_OUTPUT_HASH = (
    "8d1444d6de976930727c016b5f0782b35a3aaefe6fa1cedfb4d994c586d6f433"
)
TRANSFORMER_LAYER_INDEX_HASH = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
TRANSFORMER_BODY_OUTPUT_HASH = (
    "7631aa190daf3262dc051644303b9e256e14c0f8ed24a29d354236ee056a3905"
)
TRANSFORMER_BODY_LAYER_INDEX_HASH = (
    "61126de1b795b976f3ac878f48e88fa77a87d7308ba57c7642b9e1068403a496"
)
TRANSFORMER_TAIL_OUTPUT_HASH = (
    "7631aa190daf3262dc051644303b9e256e14c0f8ed24a29d354236ee056a3905"
)


def test_transformer():
    input_ids = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, attention_mask, src_len_mask = model.embedding_stage(
        input_ids, None, None, None
    )
    collector = {}
    model.transformer.reset_parameters()
    transformer_output, energy_per_sample = model.transformer(
        x=embedding,
        attention_mask=attention_mask,
        sequence_length=src_len_mask,
        e=embedding.clone(),
        input_ids=input_ids,
        collector=collector,
    )

    EXPECTED_KEYS = [
        "transformer_head_output",
        "transformer_head_layer_index",
        "transformer_body_output",
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

    layer_index_val = collector["transformer_head_layer_index"]
    print(f'LAYER_INDEX_HASH = "{hashlib.sha256(bytes(layer_index_val)).hexdigest()}"')

    print(f"BODY_OUTPUT_HASH = \"{hashlib.sha256(
        collector['transformer_body_output'].detach().numpy().tobytes()
    ).hexdigest()}\"")

    body_layer_index_val = collector["transformer_body_layer_index"]
    body_layer_index_hash = hashlib.sha256(bytes(body_layer_index_val)).hexdigest()
    print(f'BODY_LAYER_INDEX_HASH = "{body_layer_index_hash}"')

    print(f"TAIL_OUTPUT_HASH = \"{hashlib.sha256(
        collector['transformer_tail_output'].detach().numpy().tobytes()
    ).hexdigest()}\"")

    head_output_hash = hashlib.sha256(
        collector["transformer_head_output"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        head_output_hash == TRANSFORMER_HEAD_OUTPUT_HASH
    ), f"Head Output Hash Mismatch: {head_output_hash}"

    layer_index_hash = hashlib.sha256(bytes(layer_index_val)).hexdigest()
    assert (
        layer_index_hash == TRANSFORMER_LAYER_INDEX_HASH
    ), f"Layer Index Hash Mismatch: {layer_index_hash}"

    body_output_hash = hashlib.sha256(
        collector["transformer_body_output"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        body_output_hash == TRANSFORMER_BODY_OUTPUT_HASH
    ), f"Body Output Hash Mismatch: {body_output_hash}"

    body_layer_index_hash = hashlib.sha256(bytes(body_layer_index_val)).hexdigest()
    assert (
        body_layer_index_hash == TRANSFORMER_BODY_LAYER_INDEX_HASH
    ), f"Body Layer Index Hash Mismatch: {body_layer_index_hash}"

    tail_output_hash = hashlib.sha256(
        collector["transformer_tail_output"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        tail_output_hash == TRANSFORMER_TAIL_OUTPUT_HASH
    ), f"Tail Output Hash Mismatch: {tail_output_hash}"

    # Same as tail_output_hash.
    final_output_hash = hashlib.sha256(
        transformer_output.detach().numpy().tobytes()
    ).hexdigest()
    assert (
        final_output_hash == TRANSFORMER_TAIL_OUTPUT_HASH
    ), f"Final Output Hash Mismatch: {final_output_hash}"


if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_transformer()
