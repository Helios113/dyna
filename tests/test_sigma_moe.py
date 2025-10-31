import hashlib
import math
import os

import torch
from generate_standard_inputs import generate_standard_inputs
from generate_standard_lm import generate_standard_lm

from dyna.model import DynaLM
from dyna.transition import SigmaMoE

AFFINITY_HASH = "04ffe4709fe0ada1949403b0c17af63090e6c577e03d6c6fcbe0bf691a699b72"
SELECTION_INDEX_HASH = (
    "e148557b0539c9500e28d0d2d5bf37be16fe23c74dbda8bbd4e76b10643646fc"
)
SELECTION_INDICES_HASH = (
    "2c14e489709d993ced62c6468733352504c13d8c0ab8707f1337c871b740c7cc"
)
SCORES_PRE_ACTIVATION_HASH = (
    "78c77b5d54f59402786c2981878955db0e02c35c7e40e3841d221829d20312c6"
)
SCORES_POST_ACTIVATION_HASH = (
    "1a4657603dec2c7e06fbf0fdc8a59a6c5fe30a0a59476a235086f7559a16146a"
)
OUTPUT_HASH = "02c63429a3f89b88a3ff288091fe1ba5ddc1215c8fc40a586dd06b283ba0c1c8"
FINAL_OUTPUT_HASH = "02c63429a3f89b88a3ff288091fe1ba5ddc1215c8fc40a586dd06b283ba0c1c8"


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

    print(collector["sigma_moe_affinity"])

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
                collector["sigma_moe_selection_indices"].sel.detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'SCORES_PRE_ACTIVATION_HASH = "{
            hashlib.sha256(
                collector["sigma_moe_scores_pre_activation"].detach().cpu().numpy().tobytes()
            ).hexdigest()
        }"'
    )
    print(
        f'SCORES_POST_ACTIVATION_HASH = "{
            hashlib.sha256(
                collector["sigma_moe_scores_post_activation"].detach().cpu().numpy().tobytes()
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
