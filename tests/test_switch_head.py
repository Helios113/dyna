import hashlib
import os

import torch
from generate_standard_inputs import generate_standard_inputs
from generate_standard_lm import generate_standard_lm

from dyna.attention import SwitchHead
from dyna.model import DynaLM

# Hash for multiple experts scenario (n_experts_attn > 1)
Q_VAL_MULTIPLE_HASH = "missing, unable to run the file to get hash"
K_VAL_MULTIPLE_HASH = "missing, unable to run the file to get hash"
V_SEL_MULTIPLE_HASH = "missing, unable to run the file to get hash"
V_SEL_R_MULTIPLE_HASH = "missing, unable to run the file to get hash"
V_SEL_INDEX_MULTIPLE_HASH = "missing, unable to run the file to get hash"
O_SEL_MULTIPLE_HASH = "missing, unable to run the file to get hash"
O_SEL_R_MULTIPLE_HASH = "missing, unable to run the file to get hash"
O_SEL_INDEX_MULTIPLE_HASH = "missing, unable to run the file to get hash"
V_VAL_MULTIPLE_HASH = "missing, unable to run the file to get hash"
Q_VAL_O_MULTIPLE_HASH = "missing, unable to run the file to get hash"
K_VAL_O_MULTIPLE_HASH = "missing, unable to run the file to get hash"
Q_VAL_O_AFTER_DROPOUT_MULTIPLE_HASH = "missing, unable to run the file to get hash"
RES_BEFORE_TRANSPOSE_MULTIPLE_HASH = "missing, unable to run the file to get hash"
RES_AFTER_TRANSPOSE_MULTIPLE_HASH = "missing, unable to run the file to get hash"
OUTPUT_MULTIPLE_HASH = "missing, unable to run the file to get hash"
FINAL_OUTPUT_MULTIPLE_HASH = "missing, unable to run the file to get hash"
# Hash for single expert scenario (n_experts_attn = 1)
Q_VAL_SINGLE_HASH = "43505304badea2bd6d5f3e8acf631b0f1b0dbff1e8073ac162e1a65fd0833cea"
K_VAL_SINGLE_HASH = "2945689855996ed47e6cc4b8d9b42eb9709d5f18022273cda83768197addb40e"
O_GATE_SINGLE_HASH = "6939c2f20c1e952a413331ad4f829ebe0e21aa7d2ba37f6f366351193bd17379"
V_VAL_SINGLE_HASH = "438cc2034eabf6f5bdb4e9ad96af79807b38a29f76dfd7e00ddae3684aef478f"
V_VAL_O_SINGLE_HASH = "1650875d996602bc3bf2454a4ea9b38860b989fa71520d7d70c88c1bdd045265"
Q_VAL_O_SINGLE_HASH = "ce99e648c2488df6903bb6d36c5ec035dbf86a3ab6a0a578264b4da21647d99a"
K_VAL_O_SINGLE_HASH = "a2a760b8854c0ae66d31009c7485e597b35849aebe121972641e50809734a5c9"
Q_VAL_O_AFTER_DROPOUT_SINGLE_HASH = (
    "ce99e648c2488df6903bb6d36c5ec035dbf86a3ab6a0a578264b4da21647d99a"
)
RES_BEFORE_TRANSPOSE_SINGLE_HASH = (
    "c662b2475da2710264dd36bfceed800035cc30546a18a24c9c8990e04be1a9db"
)
RES_AFTER_TRANSPOSE_SINGLE_HASH = (
    "31c9322264e6c7e26437554c7125b564643ef6f4c692a7034dc02bfb1bc78248"
)
RES_AFTER_ELEMENT_MULTIPLICATION_SINGLE_HASH = (
    "ad612fc93c83ab7c23b649563ff77fb351625081c9c1f7af6d6bfdf67814494f"
)
RES_AFTER_VIEW_SINGLE_HASH = (
    "ad612fc93c83ab7c23b649563ff77fb351625081c9c1f7af6d6bfdf67814494f"
)
OUTPUT_SINGLE_HASH = "e92ef76fe06a93886a7433a9823044302d3428481a4fdd02b085f665b00263ef"
FINAL_OUTPUT_SINGLE_HASH = (
    "e92ef76fe06a93886a7433a9823044302d3428481a4fdd02b085f665b00263ef"
)


def test_switch_head_multiple_experts():
    """Test with multiple experts (n_experts_attn > 1)"""
    input = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, attention_mask, sequence_length = model.embedding_stage(
        input, None, None, None
    )
    d_model = model.config.d_model
    n_heads = model.config.n_heads
    d_head = model.config.d_head
    n_experts_attn = model.config.n_experts_attn
    k_attn = model.config.k_attn
    n_expert_shared_attn = model.config.n_expert_shared_attn
    dropout_expert_attn = model.config.dropout_expert_attn
    torch.manual_seed(42)

    switch_head: SwitchHead = SwitchHead(
        d_model=d_model,
        n_heads=n_heads,
        d_head=d_head,
        n_experts_attn=n_experts_attn,
        k_attn=k_attn,
        n_expert_shared_attn=n_expert_shared_attn,
        dropout_expert=dropout_expert_attn,
    )
    collector = {}
    switch_head_output, _ = switch_head(
        q_src=embedding,
        k_src=embedding,
        v_src=embedding,
        attention_mask=attention_mask,
        sequence_length=sequence_length,
        collector=collector,
    )

    EXPECTED_KEYS = [
        "switch_head_q_val",
        "switch_head_k_val",
        "switch_head_multiple_v_sel",
        "switch_head_multiple_v_sel_r",
        "switch_head_multiple_v_sel_index",
        "switch_head_multiple_o_sel",
        "switch_head_multiple_o_sel_r",
        "switch_head_multiple_o_sel_index",
        "switch_head_multiple_v_val",
        "switch_head_multiple_q_val_o",
        "switch_head_multiple_k_val_o",
        "switch_head_multiple_q_val_o_after_dropout",
        "switch_head_multiple_res_before_transpose",
        "switch_head_multiple_res_after_transpose",
        "switch_head_multiple_output",
    ]
    assert set(collector.keys()) == set(EXPECTED_KEYS), (
        f"Collector keys mismatch.\nExpected: {EXPECTED_KEYS}\n"
        f"Got: {list(collector.keys())}"
    )

    # Uncomment to capture hashes
    print(
        f'Q_VAL_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_q_val"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'K_VAL_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_k_val"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'V_SEL_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_v_sel"]
                .out_index.detach()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'V_SEL_R_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_v_sel_r"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'V_SEL_INDEX_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_v_sel_index"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'O_SEL_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_o_sel"]
                .out_index.detach()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'O_SEL_R_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_o_sel_r"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'O_SEL_INDEX_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_o_sel_index"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'V_VAL_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_v_val"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'Q_VAL_O_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_q_val_o"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'K_VAL_O_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_k_val_o"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'Q_VAL_O_AFTER_DROPOUT_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_q_val_o_after_dropout"]
                .detach()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'RES_BEFORE_TRANSPOSE_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_res_before_transpose"]
                .detach()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'RES_AFTER_TRANSPOSE_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_res_after_transpose"]
                .detach()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'OUTPUT_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_output"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'FINAL_OUTPUT_MULTIPLE_HASH = "{
            hashlib.sha256(switch_head_output.detach().numpy().tobytes()).hexdigest()
        }"'
    )

    q_val_hash = hashlib.sha256(
        collector["switch_head_q_val"].detach().numpy().tobytes()
    ).hexdigest()
    assert q_val_hash == Q_VAL_MULTIPLE_HASH, f"Query Value Hash Mismatch: {q_val_hash}"

    k_val_hash = hashlib.sha256(
        collector["switch_head_k_val"].detach().numpy().tobytes()
    ).hexdigest()
    assert k_val_hash == K_VAL_MULTIPLE_HASH, f"Key Value Hash Mismatch: {k_val_hash}"

    v_sel_hash = hashlib.sha256(
        collector["switch_head_multiple_v_sel"].out_index.detach().numpy().tobytes()
    ).hexdigest()
    assert (
        v_sel_hash == V_SEL_MULTIPLE_HASH
    ), f"Value Selection Hash Mismatch: {v_sel_hash}"

    v_sel_r_hash = hashlib.sha256(
        collector["switch_head_multiple_v_sel_r"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        v_sel_r_hash == V_SEL_R_MULTIPLE_HASH
    ), f"Value Selection Raw Hash Mismatch: {v_sel_r_hash}"

    v_sel_index_hash = hashlib.sha256(
        collector["switch_head_multiple_v_sel_index"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        v_sel_index_hash == V_SEL_INDEX_MULTIPLE_HASH
    ), f"Value Selection Index Hash Mismatch: {v_sel_index_hash}"

    o_sel_hash = hashlib.sha256(
        collector["switch_head_multiple_o_sel"].out_index.detach().numpy().tobytes()
    ).hexdigest()
    assert (
        o_sel_hash == O_SEL_MULTIPLE_HASH
    ), f"Output Selection Hash Mismatch: {o_sel_hash}"

    o_sel_r_hash = hashlib.sha256(
        collector["switch_head_multiple_o_sel_r"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        o_sel_r_hash == O_SEL_R_MULTIPLE_HASH
    ), f"Output Selection Raw Hash Mismatch: {o_sel_r_hash}"

    o_sel_index_hash = hashlib.sha256(
        collector["switch_head_multiple_o_sel_index"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        o_sel_index_hash == O_SEL_INDEX_MULTIPLE_HASH
    ), f"Output Selection Index Hash Mismatch: {o_sel_index_hash}"

    v_val_hash = hashlib.sha256(
        collector["switch_head_multiple_v_val"].detach().numpy().tobytes()
    ).hexdigest()
    assert v_val_hash == V_VAL_MULTIPLE_HASH, f"Value Hash Mismatch: {v_val_hash}"

    q_val_o_hash = hashlib.sha256(
        collector["switch_head_multiple_q_val_o"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        q_val_o_hash == Q_VAL_O_MULTIPLE_HASH
    ), f"Query Value Output Hash Mismatch: {q_val_o_hash}"

    k_val_o_hash = hashlib.sha256(
        collector["switch_head_multiple_k_val_o"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        k_val_o_hash == K_VAL_O_MULTIPLE_HASH
    ), f"Key Value Output Hash Mismatch: {k_val_o_hash}"

    q_val_o_after_dropout_hash = hashlib.sha256(
        collector["switch_head_multiple_q_val_o_after_dropout"]
        .detach()
        .numpy()
        .tobytes()
    ).hexdigest()
    assert (
        q_val_o_after_dropout_hash == Q_VAL_O_AFTER_DROPOUT_MULTIPLE_HASH
    ), f"Query Value After Dropout Output Hash Mismatch: {q_val_o_after_dropout_hash}"

    res_before_transpose_hash = hashlib.sha256(
        collector["switch_head_multiple_res_before_transpose"]
        .detach()
        .numpy()
        .tobytes()
    ).hexdigest()
    assert (
        res_before_transpose_hash == RES_BEFORE_TRANSPOSE_MULTIPLE_HASH
    ), f"Result Before Transpose Hash Mismatch: {res_before_transpose_hash}"

    res_after_transpose_hash = hashlib.sha256(
        collector["switch_head_multiple_res_after_transpose"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        res_after_transpose_hash == RES_AFTER_TRANSPOSE_MULTIPLE_HASH
    ), f"Result After Transpose Hash Mismatch: {res_after_transpose_hash}"

    output_hash = hashlib.sha256(
        collector["switch_head_multiple_output"].detach().numpy().tobytes()
    ).hexdigest()
    assert output_hash == OUTPUT_MULTIPLE_HASH, f"Output Hash Mismatch: {output_hash}"

    final_output_hash = hashlib.sha256(
        switch_head_output.detach().numpy().tobytes()
    ).hexdigest()
    assert (
        final_output_hash == FINAL_OUTPUT_MULTIPLE_HASH
    ), f"Final Output Hash Mismatch: {final_output_hash}"


def test_switch_head_single_expert():
    """Test with single expert (n_experts_attn = 1)"""
    input = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, attention_mask, sequence_length = model.embedding_stage(
        input, None, None, None
    )
    d_model = model.config.d_model
    n_heads = model.config.n_heads
    d_head = model.config.d_head
    torch.manual_seed(42)

    switch_head: SwitchHead = SwitchHead(
        d_model=d_model,
        n_heads=n_heads,
        d_head=d_head,
        n_experts_attn=1,
        k_attn=1,
        n_expert_shared_attn=0,
        dropout_expert=0.0,
    )
    collector = {}
    switch_head_output, _ = switch_head(
        q_src=embedding,
        k_src=embedding,
        v_src=embedding,
        attention_mask=attention_mask,
        sequence_length=sequence_length,
        collector=collector,
    )

    EXPECTED_KEYS = [
        "switch_head_q_val",
        "switch_head_k_val",
        "switch_head_single_o_gate",
        "switch_head_single_v_val",
        "switch_head_single_v_val_o",
        "switch_head_single_q_val_o",
        "switch_head_single_k_val_o",
        "switch_head_single_q_val_o_after_dropout",
        "switch_head_single_res_before_transpose",
        "switch_head_single_res_after_transpose",
        "switch_head_single_res_after_element_multiplication",
        "switch_head_single_res_after_view",
        "switch_head_single_output",
    ]
    assert set(collector.keys()) == set(EXPECTED_KEYS), (
        f"Collector keys mismatch.\nExpected: {EXPECTED_KEYS}\n"
        f"Got: {list(collector.keys())}"
    )

    # Uncomment to capture hashes
    print(
        f'Q_VAL_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_q_val"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'K_VAL_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_k_val"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'O_GATE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_o_gate"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'V_VAL_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_v_val"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'V_VAL_O_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_v_val_o"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'Q_VAL_O_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_q_val_o"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'K_VAL_O_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_k_val_o"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'Q_VAL_O_AFTER_DROPOUT_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_q_val_o_after_dropout"]
                .detach()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'RES_BEFORE_TRANSPOSE_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_res_before_transpose"]
                .detach()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'RES_AFTER_TRANSPOSE_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_res_after_transpose"]
                .detach()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'RES_AFTER_ELEMENT_MULTIPLICATION_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_res_after_element_multiplication"]
                .detach()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'RES_AFTER_VIEW_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_res_after_view"]
                .detach()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'OUTPUT_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_output"].detach().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'FINAL_OUTPUT_SINGLE_HASH = "{
            hashlib.sha256(switch_head_output.detach().numpy().tobytes()).hexdigest()
        }"'
    )

    q_val_single_hash = hashlib.sha256(
        collector["switch_head_q_val"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        q_val_single_hash == Q_VAL_SINGLE_HASH
    ), f"Query Value Single Hash Mismatch: {q_val_single_hash}"

    k_val_single_hash = hashlib.sha256(
        collector["switch_head_k_val"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        k_val_single_hash == K_VAL_SINGLE_HASH
    ), f"Key Value Single Hash Mismatch: {k_val_single_hash}"

    o_gate_hash = hashlib.sha256(
        collector["switch_head_single_o_gate"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        o_gate_hash == O_GATE_SINGLE_HASH
    ), f"Output Gate Hash Mismatch: {o_gate_hash}"

    v_val_single_hash = hashlib.sha256(
        collector["switch_head_single_v_val"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        v_val_single_hash == V_VAL_SINGLE_HASH
    ), f"Value Single Hash Mismatch: {v_val_single_hash}"

    v_val_o_single_hash = hashlib.sha256(
        collector["switch_head_single_v_val_o"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        v_val_o_single_hash == V_VAL_O_SINGLE_HASH
    ), f"Value Output Single Hash Mismatch: {v_val_o_single_hash}"

    q_val_o_single_hash = hashlib.sha256(
        collector["switch_head_single_q_val_o"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        q_val_o_single_hash == Q_VAL_O_SINGLE_HASH
    ), f"Query Value Output Single Hash Mismatch: {q_val_o_single_hash}"

    k_val_o_single_hash = hashlib.sha256(
        collector["switch_head_single_k_val_o"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        k_val_o_single_hash == K_VAL_O_SINGLE_HASH
    ), f"Key Value Output Single Hash Mismatch: {k_val_o_single_hash}"

    q_val_o_after_dropout_single_hash = hashlib.sha256(
        collector["switch_head_single_q_val_o_after_dropout"].detach().numpy().tobytes()
    ).hexdigest()
    assert q_val_o_after_dropout_single_hash == Q_VAL_O_AFTER_DROPOUT_SINGLE_HASH, (
        f"Query Value Output After Dropout Single Hash Mismatch: "
        f"{q_val_o_after_dropout_single_hash}"
    )

    res_before_transpose_single_hash = hashlib.sha256(
        collector["switch_head_single_res_before_transpose"].detach().numpy().tobytes()
    ).hexdigest()
    assert res_before_transpose_single_hash == RES_BEFORE_TRANSPOSE_SINGLE_HASH, (
        f"Result Before Transpose Single Hash Mismatch: "
        f"{res_before_transpose_single_hash}"
    )

    res_after_transpose_single_hash = hashlib.sha256(
        collector["switch_head_single_res_after_transpose"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        res_after_transpose_single_hash == RES_AFTER_TRANSPOSE_SINGLE_HASH
    ), f"Result After Transpose Single Hash Mismatch: {res_after_transpose_single_hash}"

    res_after_element_multiplication_hash = hashlib.sha256(
        collector["switch_head_single_res_after_element_multiplication"]
        .detach()
        .numpy()
        .tobytes()
    ).hexdigest()
    assert (
        res_after_element_multiplication_hash
        == RES_AFTER_ELEMENT_MULTIPLICATION_SINGLE_HASH
    ), (
        f"Result After Element Multiplication Hash Mismatch: "
        f"{res_after_element_multiplication_hash}"
    )

    res_after_view_hash = hashlib.sha256(
        collector["switch_head_single_res_after_view"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        res_after_view_hash == RES_AFTER_VIEW_SINGLE_HASH
    ), f"Result After View Hash Mismatch: {res_after_view_hash}"

    output_single_hash = hashlib.sha256(
        collector["switch_head_single_output"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        output_single_hash == OUTPUT_SINGLE_HASH
    ), f"Output Single Hash Mismatch: {output_single_hash}"

    final_output_single_hash = hashlib.sha256(
        switch_head_output.detach().numpy().tobytes()
    ).hexdigest()
    assert (
        final_output_single_hash == FINAL_OUTPUT_SINGLE_HASH
    ), f"Final Output Single Hash Mismatch: {final_output_single_hash}"


if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_switch_head_multiple_experts()
    test_switch_head_single_expert()
