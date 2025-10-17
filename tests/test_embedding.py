import hashlib
import os

from generate_standard_inputs import generate_standard_inputs
from generate_standard_lm import generate_standard_lm

from dyna.model import DynaLM

DEFAULT_HASH = "8d1444d6de976930727c016b5f0782b35a3aaefe6fa1cedfb4d994c586d6f433"


def test_embeddings():
    input = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, _, _ = model.embedding_stage(input, None, None, None)

    # TODO: check gradients as well!
    # .grad_fn is not None
    # .grad if that not zero
    # Test gradients.py
    # grad(emb) * grad(transformer) * grad(task_head )

    # Hash the embedding using hashlib
    hash_value = hashlib.sha256(embedding.detach().cpu().numpy().tobytes()).hexdigest()
    assert hash_value == DEFAULT_HASH


if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    test_embeddings()
