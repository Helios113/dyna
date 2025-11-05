import json
from typing import Any

import torch


def generate_computation_graph(tensor: torch.Tensor) -> dict[str, Any]:
    """Generate computation graph structure for verification purposes.

    Args:
        tensor: PyTorch tensor with gradient function

    Returns:
        Dictionary containing the graph structure
    """
    if not tensor.requires_grad or tensor.grad_fn is None:
        return {"operations": [], "edges": []}

    operations = []
    edges = []
    visited = set()
    node_map = {}

    def traverse(grad_fn, node_id=0):
        if grad_fn is None or grad_fn in visited:
            return node_map.get(grad_fn, -1)
        visited.add(grad_fn)
        node_map[grad_fn] = node_id
        operations.append(grad_fn.__class__.__name__)

        current_id = node_id
        next_id = node_id + 1

        if hasattr(grad_fn, "next_functions"):
            for next_fn, _ in grad_fn.next_functions:
                child_id = traverse(next_fn, next_id)
                if child_id != -1:
                    edges.append([current_id, child_id])
                    next_id = max(next_id, child_id + 1)

        return current_id

    traverse(tensor.grad_fn)
    return {"operations": operations, "edges": edges}


def save_graph(graph: dict[str, Any], output_file: str):
    """Save graph to JSON file."""
    with open(output_file, "w") as f:
        json.dump(graph, f, indent=2)


def verify_same_computation(graph1: dict, graph2: dict) -> bool:
    """Verify if two graphs have the same computational structure."""
    return (
        graph1["operations"] == graph2["operations"]
        and graph1["edges"] == graph2["edges"]
    )
