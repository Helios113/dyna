import json
import os

import torch

from dyna.model import DynaLM
from tests.generate_standard_inputs import generate_standard_inputs
from tests.generate_standard_lm import generate_standard_lm
from tests.graph_utils import get_computation_graph


def test_transformer():
    input_ids = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, attention_mask, src_len_mask = model.embedding_stage(
        input_ids, None, None, None
    )
    embedding.requires_grad_(True)

    torch.manual_seed(42)
    transformer_output, _ = model.transformer(
        x=embedding,
        attention_mask=attention_mask,
        sequence_length=src_len_mask,
        e=embedding.clone(),
        input_ids=input_ids,
    )

    # Load the goal graph
    script_dir = os.path.dirname(__file__)
    graph_path = os.path.join(script_dir, "graph_jsons/standard_transformer_graph.json")
    with open(graph_path) as f:
        goal_graph = json.load(f)

    graph = get_computation_graph(transformer_output)

    assert (
        graph == goal_graph
    ), "Transformer computation graph does not match the goal graph."


if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_transformer()
