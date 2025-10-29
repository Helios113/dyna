import hashlib
import math
import os

import torch
from generate_standard_inputs import generate_standard_inputs
from generate_standard_lm import generate_standard_lm

from dyna.attention import SwitchHead
from dyna.model import DynaLM

# Hash for multiple experts scenario (n_experts_attn > 1)
Q_VAL_MULTIPLE_HASH = "ad61e918f9c5ec2e3a2706db4dcc0e2ca896ff6e9d38283e42c56ab4fc2321ce"
K_VAL_MULTIPLE_HASH = "153a852d66754bf6a8917f258521e412d5ab585dd8e44ec32abc211dacb750fd"
V_SEL_MULTIPLE_HASH = "82d1222ec44baa515a574a9ed67526e78461d3b6da113e4dda4647dcfb886639"
V_SEL_R_MULTIPLE_HASH = (
    "e9a918a344b560714908f8169d7163397a584b597f9968cb15932a23d2ad909a"
)
V_SEL_INDEX_MULTIPLE_HASH = (
    "ca67880ea4dc6e3b17623e30b9456bdeb5f5337b6de47fe6a95e27e04eb86977"
)
O_SEL_MULTIPLE_HASH = "ba1456b5f6bc098979fd4f34455cc326d4e7b6bd1dbe2446fdc4bdc5ac16b46b"
O_SEL_R_MULTIPLE_HASH = (
    "80396dcfa7d0e345f6f6513f8bffc8326cab6e854468f6b52f4ada03667e3508"
)
O_SEL_INDEX_MULTIPLE_HASH = (
    "00dedfc876c60d15d08d3eac38a7fb3892a47c2c0d4f6d91e3d204e3675bcef4"
)
V_VAL_MULTIPLE_HASH = "b311734b3a748332d2f2b28237e2833cfd5af66ec13cbdb42dfccdc20dc5909a"
Q_VAL_O_MULTIPLE_HASH = (
    "9d757a3dccf97a4cad8718b28d1d0ff365461e4d0bd7600077d03ee04aec1b55"
)
K_VAL_O_MULTIPLE_HASH = (
    "3856c54a60ccb30f1a057b6b5216f19e62149b0308d2ca75ca115814a045dd56"
)
Q_VAL_O_AFTER_DROPOUT_MULTIPLE_HASH = (
    "9d757a3dccf97a4cad8718b28d1d0ff365461e4d0bd7600077d03ee04aec1b55"
)
RES_BEFORE_TRANSPOSE_MULTIPLE_HASH = (
    "3ba111b2a1d86c15b56290a93c5cb68f40990986f15b1412421b613f116c6783"
)
RES_AFTER_TRANSPOSE_MULTIPLE_HASH = (
    "9d3bdb5149587482e3c04eefe1fb0e86db6d588141150852a6c07e0d5dbe151d"
)
OUTPUT_MULTIPLE_HASH = (
    "6c502ec07fe97220f42d57ad573c60b20a127ad9adfc02a23b5b58e6bb34e426"
)
FINAL_OUTPUT_MULTIPLE_HASH = (
    "6c502ec07fe97220f42d57ad573c60b20a127ad9adfc02a23b5b58e6bb34e426"
)
# Hash for single expert scenario (n_experts_attn = 1)
Q_VAL_SINGLE_HASH = "c572c07882ed7b20efe1d52ea9fc59f1027d0bd025a51250412b5ef26b8f7694"
K_VAL_SINGLE_HASH = "b51ce4dafb562c41d7d3b724fd4b257979acc443010282f9fef30914fa71b783"
O_GATE_SINGLE_HASH = "60c01a825d89b2de40ddd3999e0a94c88fc5462a4cbef4f19dcb7d7c83a01448"
V_VAL_SINGLE_HASH = "dfaaf6257ad7ba999fd5343c126f9e60e85a2a3c843b7cf2a85e5e7c58abd88d"
V_VAL_O_SINGLE_HASH = "593ae6aeea1cb4746b5c8c5e9026913d35760747d3856fb4864058947bc00ad6"
Q_VAL_O_SINGLE_HASH = "bba028d5a68e8957975c7154cc6339c487eac94fe864c045d51126a331df4b31"
K_VAL_O_SINGLE_HASH = "919e641359655d4d95418e13a61faef095fe99daefbec2b9bde4ef5949d226a8"
Q_VAL_O_AFTER_DROPOUT_SINGLE_HASH = (
    "bba028d5a68e8957975c7154cc6339c487eac94fe864c045d51126a331df4b31"
)
RES_BEFORE_TRANSPOSE_SINGLE_HASH = (
    "2532c14458e772aaa9b49df3f78db8663bc4b688b1b3eec3d0c4ac84d0fe9a21"
)
RES_AFTER_TRANSPOSE_SINGLE_HASH = (
    "f0a59227df8bc8657ddf7268f4d01a86b86c2fcee8d5219cc9807fecfb811c43"
)
RES_AFTER_ELEMENT_MULTIPLICATION_SINGLE_HASH = (
    "84ef940caa6155e86d78a5bafc5c8f422f6e5a0bdcf3b61f2e18bb38878df736"
)
RES_AFTER_VIEW_SINGLE_HASH = (
    "84ef940caa6155e86d78a5bafc5c8f422f6e5a0bdcf3b61f2e18bb38878df736"
)
OUTPUT_SINGLE_HASH = "04ad7ce85865733d414ecd83daff1d8fba118b1ac158afcfeea8c7f481475886"
FINAL_OUTPUT_SINGLE_HASH = (
    "04ad7ce85865733d414ecd83daff1d8fba118b1ac158afcfeea8c7f481475886"
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
    torch.manual_seed(42)

    switch_head: SwitchHead = SwitchHead(
        d_model=d_model,
        n_heads=n_heads,
        d_head=d_head,
        n_experts_attn=n_experts_attn,
        k_attn=k_attn,
        n_expert_shared_attn=n_expert_shared_attn,
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
                collector["switch_head_multiple_v_sel"].out_index.detach().numpy()
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
                collector["switch_head_multiple_o_sel"].out_index.detach().numpy()
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
                collector["switch_head_multiple_q_val_o_after_dropout"].detach().numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'RES_BEFORE_TRANSPOSE_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_res_before_transpose"].detach().numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'RES_AFTER_TRANSPOSE_MULTIPLE_HASH = "{
            hashlib.sha256(
                collector["switch_head_multiple_res_after_transpose"].detach().numpy()
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
        f'O_GATE_SINGLE_HASH = "{
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
