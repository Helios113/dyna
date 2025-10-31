import hashlib
import os

import torch
from generate_standard_inputs import generate_standard_inputs
from generate_standard_lm import generate_standard_lm

from dyna.model import DynaLM

Q_PROJ_HASH = "df0e53257db633db3201ac1c46157608b33986d2bc2e06637bdffe0f9450c5ee"
K_PROJ_HASH = "b88408082ae62e071ab43eb2ba92f3c4b999cc77eb147b07cb26fd8f16587d20"
V_PROJ_HASH = "d208218c47de320a55de6ca17fde4ea80840465c2495fe3b36e7dd50972d32ef"
Q_RESHAPED_HASH = "b8ee31164d94c161a02e31d951e473e1855bb6c0a406c99f790857655886b744"
K_RESHAPED_HASH = "ecebf74aacd032e4c6bf1c652ae9d562c89ec607882539125d6dc9a3ab371eb5"
V_RESHAPED_HASH = "7baa6945f3b30def6bd05dcee722623ed723cbeb8b1f7b10bf16457672002f25"
ATTENTION_OUTPUT_HASH = (
    "a8a3974ef8a881e05111afa6763ea1c9bfbee93ecdfdb4045e6e3a81cb3e6bed"
)
OUTPUT_PROJ_HASH = "6fcb5539e9ed8deb1b363ade922cd568dbcacb5b01d6f86c9c890bdbe243e5a9"


def test_basic_attention():
    input = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, attention_mask, sequence_length = model.embedding_stage(
        input, None, None, None
    )

    print(hashlib.sha256(embedding.detach().numpy().tobytes()), flush=True)
    torch.manual_seed(42)
    attention = model.transformer.body_layers[0].attention
    collector = {}
    attention_output, _ = attention(
        q_src=embedding,
        k_src=embedding,
        v_src=embedding,
        attention_mask=attention_mask,
        sequence_length=sequence_length,
        collector=collector,
    )

    EXPECTED_KEYS = [
        "basic_attn_q_proj",
        "basic_attn_k_proj",
        "basic_attn_v_proj",
        "basic_attn_q_reshaped",
        "basic_attn_k_reshaped",
        "basic_attn_v_reshaped",
        "basic_attn_attention_output",
        "basic_attn_output_proj",
    ]
    assert set(collector.keys()) == set(EXPECTED_KEYS), (
        f"Collector keys mismatch.\nExpected: {EXPECTED_KEYS}\n"
        f"Got: {list(collector.keys())}"
    )

    q_proj_hash = hashlib.sha256(
        collector["basic_attn_q_proj"].detach().numpy().tobytes()
    ).hexdigest()
    assert q_proj_hash == Q_PROJ_HASH, f"Query Projection Hash Mismatch: {q_proj_hash}"

    k_proj_hash = hashlib.sha256(
        collector["basic_attn_k_proj"].detach().numpy().tobytes()
    ).hexdigest()
    assert k_proj_hash == K_PROJ_HASH, f"Key Projection Hash Mismatch: {k_proj_hash}"

    v_proj_hash = hashlib.sha256(
        collector["basic_attn_v_proj"].detach().numpy().tobytes()
    ).hexdigest()
    assert v_proj_hash == V_PROJ_HASH, f"Value Projection Hash Mismatch: {v_proj_hash}"

    q_reshaped_hash = hashlib.sha256(
        collector["basic_attn_q_reshaped"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        q_reshaped_hash == Q_RESHAPED_HASH
    ), f"Query Reshaped Hash Mismatch: {q_reshaped_hash}"

    k_reshaped_hash = hashlib.sha256(
        collector["basic_attn_k_reshaped"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        k_reshaped_hash == K_RESHAPED_HASH
    ), f"Key Reshaped Hash Mismatch: {k_reshaped_hash}"

    v_reshaped_hash = hashlib.sha256(
        collector["basic_attn_v_reshaped"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        v_reshaped_hash == V_RESHAPED_HASH
    ), f"Value Reshaped Hash Mismatch: {v_reshaped_hash}"

    # Attention output hash
    attention_output_hash = hashlib.sha256(
        collector["basic_attn_attention_output"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        attention_output_hash == ATTENTION_OUTPUT_HASH
    ), f"Attention Output Hash Mismatch: {attention_output_hash}"

    # Output projection hash
    output_proj_hash = hashlib.sha256(
        collector["basic_attn_output_proj"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        output_proj_hash == OUTPUT_PROJ_HASH
    ), f"Output Projection Hash Mismatch: {output_proj_hash}"

    # Same as output projection
    final_output_hash = hashlib.sha256(
        attention_output.detach().numpy().tobytes()
    ).hexdigest()
    assert (
        final_output_hash == OUTPUT_PROJ_HASH
    ), f"Final Output Hash Mismatch: {final_output_hash}"


if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_basic_attention()
