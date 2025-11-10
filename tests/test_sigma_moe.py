import json
import os

import torch

from dyna.model import DynaLM

from .generate_standard_inputs import generate_standard_inputs
from .generate_standard_lm import generate_standard_lm
from .graph_utils import generate_computation_graph, verify_same_computation


def test_sigma_moe():
    input_ids = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, attention_mask, sequence_length = model.embedding_stage(
        input_ids, None, None, None
    )
    embedding.requires_grad_(True)

    torch.manual_seed(42)
    sigma_moe_layer = model.transformer.body_layers[1]

    moe_out, _, _, _ = sigma_moe_layer(
        x=embedding,
        e=embedding.clone(),
        layer_index=1,
        reinjection_embeddings=embedding,
        attention_mask=attention_mask,
        sequence_length=sequence_length,
    )

    # Load the goal graph
    script_dir = os.path.dirname(__file__)
    graph_path = os.path.join(script_dir, "graph_jsons/standard_sigma_moe_graph.json")
    with open(graph_path) as f:
        goal_graph = json.load(f)

    graph = generate_computation_graph(moe_out)

    assert verify_same_computation(
        graph, goal_graph
    ), f"grph {graph} \n goal {goal_graph}"


if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_sigma_moe()
