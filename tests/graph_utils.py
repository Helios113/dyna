import torch


def get_computation_graph(tensor: torch.Tensor):
    if not tensor.requires_grad or tensor.grad_fn is None:
        return []

    graph = []
    visited = set()

    def traverse(grad_fn):
        if grad_fn is None or grad_fn in visited:
            return

        visited.add(grad_fn)
        op_name = grad_fn.__class__.__name__
        graph.append(op_name)

        if hasattr(grad_fn, "next_functions"):
            for next_fn, _ in grad_fn.next_functions:
                traverse(next_fn)

    traverse(tensor.grad_fn)
    return graph
