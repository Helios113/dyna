import json
import os

import torch

from dyna.model import DynaLM

from .generate_standard_inputs import generate_standard_inputs
from .generate_standard_lm import generate_standard_lm
from .graph_utils import generate_computation_graph, verify_same_computation


def test_basic_ffn():
    input = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, _, _ = model.embedding_stage(input, None, None, None)
    embedding.requires_grad_(True)

    torch.manual_seed(42)
    ffn = model.transformer.body_layers[0].ffn

    ffn_output, _ = ffn(embedding, None)
    script_dir = os.path.dirname(__file__)
    graph_path = os.path.join(script_dir, "graph_jsons/standard_basic_ffn_graph.json")
    # Load the goal graph
    with open(graph_path) as f:
        goal_graph = json.load(f)

    graph = generate_computation_graph(ffn_output)
    assert verify_same_computation(graph, goal_graph)


if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_basic_ffn()
