import hashlib
import math
import os

import torch
from generate_standard_inputs import generate_standard_inputs
from generate_standard_lm import generate_standard_lm

from dyna.model import DynaLM
from dyna.transition import SigmaMoE

AFFINITY_HASH = "5a22f9f716e6dcbd449d84d9338c48f6917f02b7eaae8cc060444dcfbc82726d"
SELECTION_INDEX_HASH = (
    "4ff98ff200c922c49902dc96f0d2c4211e16727c70356994e09121cf75ad79c3"
)
SELECTION_INDICES_HASH = (
    "a93766291575831cfcb9195bfababed2f1adf02edac4d31aa20208f8a1762fff"
)
SCORES_PRE_ACTIVATION_HASH = (
    "97e8dfbb6ea6143cbc8b50ea8d95095e3606910ab2c7f05479f5ede9e7af3440"
)
SCORES_POST_ACTIVATION_HASH = (
    "b0bc69b5d4bb62ad64a6faadb7740dbb84080004aeaf71a9a51fa8ce22f1b879"
)
OUTPUT_HASH = "535f6685ca998d09da47fcbac01dbf6528b2cde14964d1fd2b67b76f0bd84397"
FINAL_OUTPUT_HASH = "535f6685ca998d09da47fcbac01dbf6528b2cde14964d1fd2b67b76f0bd84397"


def test_sigma_moe():
    input = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, _, _ = model.embedding_stage(input, None, None, None)
    embedding = embedding.to("cuda")

    d_model = model.config.d_model
    n_experts_ffn = model.config.n_experts_ffn
    d_expert_ffn = model.config.d_expert_ffn
    n_expert_shared_ffn = model.config.n_expert_shared_ffn
    k_ffn = model.config.k_ffn
    torch.manual_seed(42)

    moe: SigmaMoE = SigmaMoE(
        d_model=d_model,
        n_experts_ffn=n_experts_ffn,
        d_expert_ffn=d_expert_ffn,
        n_expert_shared_ffn=n_expert_shared_ffn,
        k_ffn=k_ffn,
    ).to("cuda")
    # Ensure deterministic initialization consistent with model reset logic
    scale = (
        math.sqrt(2 / (model.config.n_repeats * model.config.total_depth_for_init))
        if model.config.enable_early_exit
        else math.sqrt(2 / model.config.total_depth_for_init)
    )
    moe.reset_parameters(scale)
    collector = {}
    moe_out, _ = moe(embedding, embedding, collector)

    EXPECTED_KEYS = [
        "sigma_moe_affinity",
        "sigma_moe_selection_index",
        "sigma_moe_selection_indices",
        "sigma_moe_scores_pre_activation",
        "sigma_moe_scores_post_activation",
        "sigma_moe_output",
    ]
    assert set(collector.keys()) == set(EXPECTED_KEYS), (
        f"Collector keys mismatch.\nExpected: {EXPECTED_KEYS}\n"
        f"Got: {list(collector.keys())}"
    )

    # Uncomment to capture hashes
    print(
        f'AFFINITY_HASH = "{
            hashlib.sha256(
                collector["sigma_moe_affinity"].detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'SELECTION_INDEX_HASH = "{
            hashlib.sha256(
                collector["sigma_moe_selection_index"].detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'SELECTION_INDICES_HASH = "{
            hashlib.sha256(
                collector["sigma_moe_selection_indices"]
                .sel.detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'SCORES_PRE_ACTIVATION_HASH = "{
            hashlib.sha256(
                collector["sigma_moe_scores_pre_activation"]
                .detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'SCORES_POST_ACTIVATION_HASH = "{
            hashlib.sha256(
                collector["sigma_moe_scores_post_activation"]
                .detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'OUTPUT_HASH = "{
            hashlib.sha256(
                collector["sigma_moe_output"].detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'FINAL_OUTPUT_HASH = "{
            hashlib.sha256(moe_out.detach().cpu().numpy().tobytes()).hexdigest()
        }"'
    )

    affinity_hash = hashlib.sha256(
        collector["sigma_moe_affinity"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert affinity_hash == AFFINITY_HASH, f"Affinity Hash Mismatch: {affinity_hash}"

    selection_index_hash = hashlib.sha256(
        collector["sigma_moe_selection_index"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        selection_index_hash == SELECTION_INDEX_HASH
    ), f"Selection Index Hash Mismatch: {selection_index_hash}"

    scores_pre_activation_hash = hashlib.sha256(
        collector["sigma_moe_scores_pre_activation"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        scores_pre_activation_hash == SCORES_PRE_ACTIVATION_HASH
    ), f"Scores Pre-Activation Hash Mismatch: {scores_pre_activation_hash}"

    scores_post_activation_hash = hashlib.sha256(
        collector["sigma_moe_scores_post_activation"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        scores_post_activation_hash == SCORES_POST_ACTIVATION_HASH
    ), f"Scores Post-Activation Hash Mismatch: {scores_post_activation_hash}"

    output_hash = hashlib.sha256(
        collector["sigma_moe_output"].detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert output_hash == OUTPUT_HASH, f"Output Hash Mismatch: {output_hash}"

    final_output_hash = hashlib.sha256(
        moe_out.detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        final_output_hash == FINAL_OUTPUT_HASH
    ), f"Final Output Hash Mismatch: {final_output_hash}"

    sel_indices_hash = hashlib.sha256(
        collector["sigma_moe_selection_indices"].sel.detach().cpu().numpy().tobytes()
    ).hexdigest()
    assert (
        sel_indices_hash == SELECTION_INDICES_HASH
    ), f"Selection Indices Hash Mismatch: {sel_indices_hash}"


if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_sigma_moe()
