import json
import os

import torch

from dyna.model import DynaLM
from tests.generate_standard_inputs import generate_standard_inputs
from tests.generate_standard_lm import generate_standard_lm
from tests.graph_utils import get_computation_graph


def test_basic_ffn():
    input = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, _, _ = model.embedding_stage(input, None, None, None)
    embedding.requires_grad_(True)

    torch.manual_seed(42)
    ffn = model.transformer.body_layers[0].ffn

    ffn_output, _ = ffn(embedding, None, {})

    # Load the goal graph
    with open("graph_jsons/standard_basic_ffn_graph.json") as f:
        goal_graph = json.load(f)

    graph = get_computation_graph(ffn_output)

    assert graph == goal_graph, "Basic FFN graph does not match the goal graph."


if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_basic_ffn()
