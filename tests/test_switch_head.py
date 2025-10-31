import hashlib
import math
import os

import torch
from generate_standard_inputs import generate_standard_inputs
from generate_standard_lm import generate_standard_lm

from dyna.attention import SwitchHead
from dyna.model import DynaLM

# Hash for multiple experts scenario (n_experts_attn > 1)
Q_VAL_MULTIPLE_HASH = "7a088e3e217107aed7fb36c75bed7d8854f240ca3469c975b4c50759b8fc76b1"
K_VAL_MULTIPLE_HASH = "1e97c752a3807773c8f4378212b692b00b62d0e650b7739585f88f9a1514d6ae"
V_SEL_MULTIPLE_HASH = "828dd92d72d5a9b1343a5319a91ab6d9436e5199b4ed5be89fa21f0bace5e3f4"
V_SEL_R_MULTIPLE_HASH = (
    "7d45a21e543051eaadc87f2ee4d7e924b9f3b3aed60fc1b6999e4ca312d16eb1"
)
V_SEL_INDEX_MULTIPLE_HASH = (
    "5bc19dbdf34b3a3dd47821d222c13e8d044cd766ef7c35989beacf10639a32d0"
)
O_SEL_MULTIPLE_HASH = "8f778c7ccd7c3ac695756f505119c7b6226e90d99e946432108f00fe131b0675"
O_SEL_R_MULTIPLE_HASH = (
    "21dd5eb10e6790f26205f6fb4aae3043776af110b1453f61c02de181b38d811e"
)
O_SEL_INDEX_MULTIPLE_HASH = (
    "b2b98fed6cf0f1ac9bf3e179e0ccc94245e4e92acfe0fe2bc10d1c0b545a3f72"
)
V_VAL_MULTIPLE_HASH = "b5a19786e44e64b9927c79c0b6c54abb71943417e69a505abb26a8d8ab4e5ef1"
Q_VAL_O_MULTIPLE_HASH = (
    "6d60ef585cff39a21ac1fbb3110f12178f906dc97c430713518a4a1d8b3dcb17"
)
K_VAL_O_MULTIPLE_HASH = (
    "1891d902177c405cdb421ec0e64851e66d106c76cfc70b871dbbb21501524dac"
)
Q_VAL_O_AFTER_DROPOUT_MULTIPLE_HASH = (
    "6d60ef585cff39a21ac1fbb3110f12178f906dc97c430713518a4a1d8b3dcb17"
)
RES_BEFORE_TRANSPOSE_MULTIPLE_HASH = (
    "140cc18ac2e16314b96f0bcbb8014b2f225712069d291f366e0cb705b077af41"
)
RES_AFTER_TRANSPOSE_MULTIPLE_HASH = (
    "9ddcb97f88cd9d011433b8fbbf9fa1392233dc70f128ab2817bc850f2050dcc3"
)
OUTPUT_MULTIPLE_HASH = (
    "1131c235cf7b188bc5f8555bff6fd822eda68388a85238212d56571f34c4e6a6"
)
FINAL_OUTPUT_MULTIPLE_HASH = (
    "1131c235cf7b188bc5f8555bff6fd822eda68388a85238212d56571f34c4e6a6"
)
# Hash for single expert scenario (n_experts_attn = 1)
Q_VAL_SINGLE_HASH = "4c3e83312d0db43d6fc0b38a0dac51a15b92d58f12ab8075eaa0e4816fe4ef03"
K_VAL_SINGLE_HASH = "b893df6be2546eedf38d5e3aeaf32688522c304b226d925c723b99b219233dc7"
O_GATE_SINGLE_HASH = "17c5d84f4f7f648329e009a8e21fb85e7da3821545f0f21e80411e5bc03235da"
V_VAL_SINGLE_HASH = "fb83efd14a5b93644d1efe6bcb1229e161e96870d50ea1fa568e398d0ce11630"
V_VAL_O_SINGLE_HASH = "15bd835b8cbfe11b8ad78ceb1160adb6c2f1da7e25a0acf7de4626f1a253d0b2"
Q_VAL_O_SINGLE_HASH = "d7f22caf6d3e0bf719c6b73b60762e61b77e0ff797c466f0e33d36d0383c2454"
K_VAL_O_SINGLE_HASH = "fdba168f61e5ee03c0b38626b239e7882b05a37d16804d354b9087f39b554f4e"
Q_VAL_O_AFTER_DROPOUT_SINGLE_HASH = (
    "d7f22caf6d3e0bf719c6b73b60762e61b77e0ff797c466f0e33d36d0383c2454"
)
RES_BEFORE_TRANSPOSE_SINGLE_HASH = (
    "e5c7702daa4f076233c329f1632105af88204d8946b04df7b345143ead13888d"
)
RES_AFTER_TRANSPOSE_SINGLE_HASH = (
    "7b9f4cd636fbfd6abb3f79593d7d4ec14a50e1ee9ce562ca5cc859d27726643c"
)
RES_AFTER_ELEMENT_MULTIPLICATION_SINGLE_HASH = (
    "dafe65976078bb6dfc7744e9dc1c215e5e01cb8cc1d1108d0b8decd10d397285"
)
RES_AFTER_VIEW_SINGLE_HASH = (
    "dafe65976078bb6dfc7744e9dc1c215e5e01cb8cc1d1108d0b8decd10d397285"
)
OUTPUT_SINGLE_HASH = "d6f6a1379b5fd50784551b9e613f08ea1016f9c8cac03d7dc76b7aac01baa0e4"
FINAL_OUTPUT_SINGLE_HASH = (
    "d6f6a1379b5fd50784551b9e613f08ea1016f9c8cac03d7dc76b7aac01baa0e4"
)


def test_switch_head_multiple_experts():
    """Test with multiple experts (n_experts_attn > 1)"""
    input = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, attention_mask, sequence_length = model.embedding_stage(
        input, None, None, None
    )
    embedding = embedding.to("cuda")
    attention_mask = attention_mask.to("cuda")
    sequence_length = sequence_length.to("cuda")

    d_model = model.config.d_model
    n_heads = model.config.n_heads
    d_head = model.config.d_head
    n_experts_attn = model.config.n_experts_attn
    k_attn = model.config.k_attn
    n_expert_shared_attn = model.config.n_expert_shared_attn
    torch.manual_seed(42)

    switch_head: SwitchHead = SwitchHead(
        d_model=d_model,
        n_heads=n_heads,
        d_head=d_head,
        n_experts_attn=n_experts_attn,
        k_attn=k_attn,
        n_expert_shared_attn=n_expert_shared_attn,
        dropout_expert=0.0,
    ).to("cuda")
    # Ensure determinism for SwitchHead
    scale = (
        math.sqrt(2 / (model.config.n_repeats * model.config.total_depth_for_init))
        if model.config.enable_early_exit
        else math.sqrt(2 / model.config.total_depth_for_init)
    )
    switch_head.reset_parameters(scale)
    switch_head.eval()
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
                collector["switch_head_q_val"].detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'K_VAL_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_k_val"].detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'V_SEL_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_v_sel"]
                .out_index.detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'V_SEL_R_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_v_sel_r"]
                .detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'V_SEL_INDEX_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_v_sel_index"]
                .detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'O_SEL_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_o_sel"]
                .out_index.detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'O_SEL_R_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_o_sel_r"]
                .detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'O_SEL_INDEX_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_o_sel_index"]
                .detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'V_VAL_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_v_val"].detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'Q_VAL_O_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_q_val_o"]
                .detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'K_VAL_O_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_k_val_o"]
                .detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'Q_VAL_O_AFTER_DROPOUT_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_q_val_o_after_dropout"]
                .detach()
                .cpu()
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
                .cpu()
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
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'OUTPUT_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_output"]
                .detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'FINAL_OUTPUT_MULTIPLE_HASH = "{
            hashlib.sha256(
                switch_head_output.detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )

    q_val_hash = hashlib.sha256(
        collector["switch_head_q_val"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert q_val_hash == Q_VAL_MULTIPLE_HASH, f"Query Value Hash Mismatch: {q_val_hash}"

    k_val_hash = hashlib.sha256(
        collector["switch_head_k_val"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert k_val_hash == K_VAL_MULTIPLE_HASH, f"Key Value Hash Mismatch: {k_val_hash}"

    v_sel_hash = hashlib.sha256(
        collector["switch_head_multiple_v_sel"]
        .out_index.detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()
    assert (
        v_sel_hash == V_SEL_MULTIPLE_HASH
    ), f"Value Selection Hash Mismatch: {v_sel_hash}"

    v_sel_r_hash = hashlib.sha256(
        collector["switch_head_multiple_v_sel_r"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        v_sel_r_hash == V_SEL_R_MULTIPLE_HASH
    ), f"Value Selection Raw Hash Mismatch: {v_sel_r_hash}"

    v_sel_index_hash = hashlib.sha256(
        collector["switch_head_multiple_v_sel_index"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        v_sel_index_hash == V_SEL_INDEX_MULTIPLE_HASH
    ), f"Value Selection Index Hash Mismatch: {v_sel_index_hash}"

    o_sel_hash = hashlib.sha256(
        collector["switch_head_multiple_o_sel"]
        .out_index.detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()
    assert (
        o_sel_hash == O_SEL_MULTIPLE_HASH
    ), f"Output Selection Hash Mismatch: {o_sel_hash}"

    o_sel_r_hash = hashlib.sha256(
        collector["switch_head_multiple_o_sel_r"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        o_sel_r_hash == O_SEL_R_MULTIPLE_HASH
    ), f"Output Selection Raw Hash Mismatch: {o_sel_r_hash}"

    o_sel_index_hash = hashlib.sha256(
        collector["switch_head_multiple_o_sel_index"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        o_sel_index_hash == O_SEL_INDEX_MULTIPLE_HASH
    ), f"Output Selection Index Hash Mismatch: {o_sel_index_hash}"

    v_val_hash = hashlib.sha256(
        collector["switch_head_multiple_v_val"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert v_val_hash == V_VAL_MULTIPLE_HASH, f"Value Hash Mismatch: {v_val_hash}"

    q_val_o_hash = hashlib.sha256(
        collector["switch_head_multiple_q_val_o"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        q_val_o_hash == Q_VAL_O_MULTIPLE_HASH
    ), f"Query Value Output Hash Mismatch: {q_val_o_hash}"

    k_val_o_hash = hashlib.sha256(
        collector["switch_head_multiple_k_val_o"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        k_val_o_hash == K_VAL_O_MULTIPLE_HASH
    ), f"Key Value Output Hash Mismatch: {k_val_o_hash}"

    q_val_o_after_dropout_hash = hashlib.sha256(
        collector["switch_head_multiple_q_val_o_after_dropout"]
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()
    assert (
        q_val_o_after_dropout_hash == Q_VAL_O_AFTER_DROPOUT_MULTIPLE_HASH
    ), f"Query Value After Dropout Output Hash Mismatch: {q_val_o_after_dropout_hash}"

    res_before_transpose_hash = hashlib.sha256(
        collector["switch_head_multiple_res_before_transpose"]
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()
    assert (
        res_before_transpose_hash == RES_BEFORE_TRANSPOSE_MULTIPLE_HASH
    ), f"Result Before Transpose Hash Mismatch: {res_before_transpose_hash}"

    res_after_transpose_hash = hashlib.sha256(
        collector["switch_head_multiple_res_after_transpose"]
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()
    assert (
        res_after_transpose_hash == RES_AFTER_TRANSPOSE_MULTIPLE_HASH
    ), f"Result After Transpose Hash Mismatch: {res_after_transpose_hash}"

    output_hash = hashlib.sha256(
        collector["switch_head_multiple_output"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert output_hash == OUTPUT_MULTIPLE_HASH, f"Output Hash Mismatch: {output_hash}"

    final_output_hash = hashlib.sha256(
        switch_head_output.detach().cpu().numpy().tobytes()
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
    # Ensure determinism for SwitchHead
    scale = (
        math.sqrt(2 / (model.config.n_repeats * model.config.total_depth_for_init))
        if model.config.enable_early_exit
        else math.sqrt(2 / model.config.total_depth_for_init)
    )
    switch_head.reset_parameters(scale)
    switch_head.eval()
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
                collector["switch_head_q_val"].detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'K_VAL_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_k_val"].detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'O_GATE_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_o_gate"].detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'V_VAL_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_v_val"].detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'V_VAL_O_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_v_val_o"].detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'Q_VAL_O_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_q_val_o"].detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'K_VAL_O_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_k_val_o"].detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'Q_VAL_O_AFTER_DROPOUT_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_q_val_o_after_dropout"]
                .detach()
                .cpu()
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
                .cpu()
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
                .cpu()
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
                .cpu()
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
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'OUTPUT_SINGLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_single_output"].detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'FINAL_OUTPUT_SINGLE_HASH = "{
            hashlib.sha256(
                switch_head_output.detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )

    q_val_single_hash = hashlib.sha256(
        collector["switch_head_q_val"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        q_val_single_hash == Q_VAL_SINGLE_HASH
    ), f"Query Value Single Hash Mismatch: {q_val_single_hash}"

    k_val_single_hash = hashlib.sha256(
        collector["switch_head_k_val"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        k_val_single_hash == K_VAL_SINGLE_HASH
    ), f"Key Value Single Hash Mismatch: {k_val_single_hash}"

    o_gate_hash = hashlib.sha256(
        collector["switch_head_single_o_gate"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        o_gate_hash == O_GATE_SINGLE_HASH
    ), f"Output Gate Hash Mismatch: {o_gate_hash}"

    v_val_single_hash = hashlib.sha256(
        collector["switch_head_single_v_val"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        v_val_single_hash == V_VAL_SINGLE_HASH
    ), f"Value Single Hash Mismatch: {v_val_single_hash}"

    v_val_o_single_hash = hashlib.sha256(
        collector["switch_head_single_v_val_o"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        v_val_o_single_hash == V_VAL_O_SINGLE_HASH
    ), f"Value Output Single Hash Mismatch: {v_val_o_single_hash}"

    q_val_o_single_hash = hashlib.sha256(
        collector["switch_head_single_q_val_o"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        q_val_o_single_hash == Q_VAL_O_SINGLE_HASH
    ), f"Query Value Output Single Hash Mismatch: {q_val_o_single_hash}"

    k_val_o_single_hash = hashlib.sha256(
        collector["switch_head_single_k_val_o"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        k_val_o_single_hash == K_VAL_O_SINGLE_HASH
    ), f"Key Value Output Single Hash Mismatch: {k_val_o_single_hash}"

    q_val_o_after_dropout_single_hash = hashlib.sha256(
        collector["switch_head_single_q_val_o_after_dropout"]
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()
    assert q_val_o_after_dropout_single_hash == Q_VAL_O_AFTER_DROPOUT_SINGLE_HASH, (
        f"Query Value Output After Dropout Single Hash Mismatch: "
        f"{q_val_o_after_dropout_single_hash}"
    )

    res_before_transpose_single_hash = hashlib.sha256(
        collector["switch_head_single_res_before_transpose"]
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()
    assert res_before_transpose_single_hash == RES_BEFORE_TRANSPOSE_SINGLE_HASH, (
        f"Result Before Transpose Single Hash Mismatch: "
        f"{res_before_transpose_single_hash}"
    )

    res_after_transpose_single_hash = hashlib.sha256(
        collector["switch_head_single_res_after_transpose"]
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()
    assert (
        res_after_transpose_single_hash == RES_AFTER_TRANSPOSE_SINGLE_HASH
    ), f"""Result After Transpose Single Hash
    Mismatch: {res_after_transpose_single_hash}"""

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
        collector["switch_head_single_res_after_view"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        res_after_view_hash == RES_AFTER_VIEW_SINGLE_HASH
    ), f"Result After View Hash Mismatch: {res_after_view_hash}"

    output_single_hash = hashlib.sha256(
        collector["switch_head_single_output"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        output_single_hash == OUTPUT_SINGLE_HASH
    ), f"Output Single Hash Mismatch: {output_single_hash}"

    final_output_single_hash = hashlib.sha256(
        switch_head_output.detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        final_output_single_hash == FINAL_OUTPUT_SINGLE_HASH
    ), f"Final Output Single Hash Mismatch: {final_output_single_hash}"


if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_switch_head_multiple_experts()
    test_switch_head_single_expert()
