import json
import os

import torch

from dyna.model import DynaLM
from tests.generate_standard_inputs import generate_standard_inputs
from tests.generate_switch_head_lm import (
    generate_switch_head_lm_single_expert,
    generate_switch_head_multiple_experts_lm,
)
from tests.graph_utils import generate_computation_graph


def test_switch_head_multiple_experts():
    """(n_experts_attn > 1)"""
    input_ids = generate_standard_inputs()
    model: DynaLM = generate_switch_head_multiple_experts_lm()

    embedding, attention_mask, sequence_length = model.embedding_stage(
        input_ids, None, None, None
    )
    model = model.to("cuda")  # type: ignore
    embedding = embedding.to("cuda")
    embedding.requires_grad_(True)
    attention_mask = attention_mask.to("cuda")
    sequence_length = sequence_length.to("cuda")

    torch.manual_seed(42)

    switch_head_layer = model.transformer.head_layers[0]
    switch_head_layer.eval()

    switch_head_output, _, _, _ = switch_head_layer(
        x=embedding,
        e=embedding.clone(),
        layer_index=0,
        reinjection_embeddings=embedding,
        attention_mask=attention_mask,
        sequence_length=sequence_length,
    )

    # Load the goal graph
    script_dir = os.path.dirname(__file__)
    graph_path = os.path.join(
        script_dir, "graph_jsons/standard_switch_head_multiple_experts_graph.json"
    )
    with open(graph_path) as f:
        goal_graph = json.load(f)

    graph = generate_computation_graph(switch_head_output)

    assert (
        graph == goal_graph
    ), "SwitchHead (multiple experts) graph does not match the goal graph."


def test_switch_head_single_expert():
    """(n_experts_attn = 1)"""
    input_ids = generate_standard_inputs()
    model: DynaLM = generate_switch_head_lm_single_expert()

    embedding, attention_mask, sequence_length = model.embedding_stage(
        input_ids, None, None, None
    )
    model = model.to("cuda")  # type: ignore
    embedding = embedding.to("cuda")
    embedding.requires_grad_(True)
    attention_mask = attention_mask.to("cuda")
    sequence_length = sequence_length.to("cuda")

    torch.manual_seed(42)

    switch_head_layer = model.transformer.head_layers[0]
    switch_head_layer.eval()

    switch_head_output, _, _, _ = switch_head_layer(
        x=embedding,
        e=embedding.clone(),
        layer_index=0,
        reinjection_embeddings=embedding,
        attention_mask=attention_mask,
        sequence_length=sequence_length,
    )

    # Load the goal graph
    script_dir = os.path.dirname(__file__)
    graph_path = os.path.join(
        script_dir, "graph_jsons/standard_switch_head_single_expert_graph.json"
    )
    with open(graph_path) as f:
        goal_graph = json.load(f)

    graph = generate_computation_graph(switch_head_output)

    assert graph == goal_graph, (
        "SwitchHead (single expert) computation graph does not match "
        "the goal graph graph."
    )


if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_switch_head_multiple_experts()
    test_switch_head_single_expert()
