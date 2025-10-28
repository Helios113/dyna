import hashlib
import os

import torch
from generate_standard_inputs import generate_standard_inputs
from generate_standard_lm import generate_standard_lm

from dyna.attention import BasicAttn
from dyna.model import DynaLM

Q_PROJ_HASH = "43505304badea2bd6d5f3e8acf631b0f1b0dbff1e8073ac162e1a65fd0833cea"
K_PROJ_HASH = "2945689855996ed47e6cc4b8d9b42eb9709d5f18022273cda83768197addb40e"
V_PROJ_HASH = "e571cf64614697d4317f4e2f3e5e881eee10381647332e15e3c2580e7d12e3ad"
Q_RESHAPED_HASH = "ce99e648c2488df6903bb6d36c5ec035dbf86a3ab6a0a578264b4da21647d99a"
K_RESHAPED_HASH = "a2a760b8854c0ae66d31009c7485e597b35849aebe121972641e50809734a5c9"
V_RESHAPED_HASH = "3c608ba93e6575952957ae7f72ce2ab14d3bcf359984d3853bbe5be80008d572"
ATTENTION_OUTPUT_HASH = (
    "dd35f81e96bda94b3ce392215508243fc51929434de37ac211e175c550c87d52"
)
OUTPUT_PROJ_HASH = "bb5c017a52f2e5363bf42091a47b05c8f9f14a2d86c851a95d9b22b8eadc03b6"


def test_basic_attention():
    input = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, attention_mask, sequence_length = model.embedding_stage(
        input, None, None, None
    )
    d_model = model.config.d_model
    n_heads = model.config.n_heads
    d_head = model.config.d_head
    torch.manual_seed(42)

    attention: BasicAttn = BasicAttn(d_model, n_heads, d_head)
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

    # print(f"Q_PROJ_HASH = \"{hashlib.sha256(
    #     collector['basic_attn_q_proj'].detach().numpy().tobytes()
    # ).hexdigest()}\"")
    # print(f"K_PROJ_HASH = \"{hashlib.sha256(
    #     collector['basic_attn_k_proj'].detach().numpy().tobytes()
    # ).hexdigest()}\"")
    # print(f"V_PROJ_HASH = \"{hashlib.sha256(
    #     collector['basic_attn_v_proj'].detach().numpy().tobytes()
    # ).hexdigest()}\"")
    # print(f"Q_RESHAPED_HASH = \"{hashlib.sha256(
    #     collector['basic_attn_q_reshaped'].detach().numpy().tobytes()
    # ).hexdigest()}\"")
    # print(f"K_RESHAPED_HASH = \"{hashlib.sha256(
    #     collector['basic_attn_k_reshaped'].detach().numpy().tobytes()
    # ).hexdigest()}\"")
    # print(f"V_RESHAPED_HASH = \"{hashlib.sha256(
    #     collector['basic_attn_v_reshaped'].detach().numpy().tobytes()
    # ).hexdigest()}\"")
    # print(f"ATTENTION_OUTPUT_HASH = \"{hashlib.sha256(
    #     collector['basic_attn_attention_output'].detach().numpy().tobytes()
    # ).hexdigest()}\"")
    # print(f"OUTPUT_PROJ_HASH = \"{hashlib.sha256(
    #     collector['basic_attn_output_proj'].detach().numpy().tobytes()
    # ).hexdigest()}\"")
    # print(f'FINAL_OUTPUT_HASH = "{hashlib.sha256(
    #     attention_output.detach().numpy().tobytes()
    # ).hexdigest()}")

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
