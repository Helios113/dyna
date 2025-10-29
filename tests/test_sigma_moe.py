import hashlib
import os

import torch
from generate_standard_inputs import generate_standard_inputs
from generate_standard_lm import generate_standard_lm

from dyna.model import DynaLM
from dyna.transition import SigmaMoE

AFFINITY_HASH = "fbec379b5a8a89da07b6e91bb344b46525dad543b458bfd84a831a23da62e4eb"
SELECTION_INDEX_HASH = (
    "8a047be7bafefafa580491d2281cd66e0099db9b67dd47f79f1a4caf82c0a6b0"
)
SELECTION_INDICES_HASH = (
    "e5fdb97a87aefdd320458deaa11c08235fd8ed53b714f7022463017a67948527"
)
SCORES_PRE_ACTIVATION_HASH = (
    "a6d72ac7690f53be6ae46ba88506bd97302a093f7108472bd9efc3cefda06484"
)
SCORES_POST_ACTIVATION_HASH = (
    "a6d72ac7690f53be6ae46ba88506bd97302a093f7108472bd9efc3cefda06484"
)
OUTPUT_HASH = "ae84a0f26a285161c3563d74a85227c66b72ec929eadfbef4079466f5b80a19e"
FINAL_OUTPUT_HASH = "ae84a0f26a285161c3563d74a85227c66b72ec929eadfbef4079466f5b80a19e"


def test_sigma_moe():
    input = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, _, _ = model.embedding_stage(input, None, None, None)
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
    )
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
    print(f"AFFINITY_HASH = \"{hashlib.sha256(
        collector['sigma_moe_affinity'].detach().numpy().tobytes()
    ).hexdigest()}\"")
    print(f"SELECTION_INDEX_HASH = \"{hashlib.sha256(
        collector['sigma_moe_selection_index'].detach().numpy().tobytes()
    ).hexdigest()}\"")
    print(f"SELECTION_INDICES_HASH = \"{hashlib.sha256(
        collector['sigma_moe_selection_indices'].sel.detach().numpy().tobytes()
    ).hexdigest()}\"")
    print(f"SCORES_PRE_ACTIVATION_HASH = \"{hashlib.sha256(
        collector['sigma_moe_scores_pre_activation'].detach().numpy().tobytes()
    ).hexdigest()}\"")
    print(f"SCORES_POST_ACTIVATION_HASH = \"{hashlib.sha256(
        collector['sigma_moe_scores_post_activation'].detach().numpy().tobytes()
    ).hexdigest()}\"")
    print(f"OUTPUT_HASH = \"{hashlib.sha256(
        collector['sigma_moe_output'].detach().numpy().tobytes()
    ).hexdigest()}\"")
    print(f'FINAL_OUTPUT_HASH = "{hashlib.sha256(
        moe_out.detach().numpy().tobytes()
    ).hexdigest()}"')

    affinity_hash = hashlib.sha256(
        collector["sigma_moe_affinity"].detach().numpy().tobytes()
    ).hexdigest()
    assert affinity_hash == AFFINITY_HASH, f"Affinity Hash Mismatch: {affinity_hash}"

    selection_index_hash = hashlib.sha256(
        collector["sigma_moe_selection_index"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        selection_index_hash == SELECTION_INDEX_HASH
    ), f"Selection Index Hash Mismatch: {selection_index_hash}"

    scores_pre_activation_hash = hashlib.sha256(
        collector["sigma_moe_scores_pre_activation"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        scores_pre_activation_hash == SCORES_PRE_ACTIVATION_HASH
    ), f"Scores Pre-Activation Hash Mismatch: {scores_pre_activation_hash}"

    scores_post_activation_hash = hashlib.sha256(
        collector["sigma_moe_scores_post_activation"].detach().numpy().tobytes()
    ).hexdigest()
    assert (
        scores_post_activation_hash == SCORES_POST_ACTIVATION_HASH
    ), f"Scores Post-Activation Hash Mismatch: {scores_post_activation_hash}"

    output_hash = hashlib.sha256(
        collector["sigma_moe_output"].detach().numpy().tobytes()
    ).hexdigest()
    assert output_hash == OUTPUT_HASH, f"Output Hash Mismatch: {output_hash}"

    final_output_hash = hashlib.sha256(moe_out.detach().numpy().tobytes()).hexdigest()
    assert (
        final_output_hash == FINAL_OUTPUT_HASH
    ), f"Final Output Hash Mismatch: {final_output_hash}"

    sel_indices_hash = hashlib.sha256(
        collector["sigma_moe_selection_indices"].sel.detach().numpy().tobytes()
    ).hexdigest()
    assert (
        sel_indices_hash == SELECTION_INDICES_HASH
    ), f"Selection Indices Hash Mismatch: {sel_indices_hash}"


if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_sigma_moe()
