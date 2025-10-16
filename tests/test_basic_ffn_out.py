import hashlib
import os

import torch
from generate_standard_inputs import generate_standard_inputs
from generate_standard_lm import generate_standard_lm

from dyna.model import DynaLM
from dyna.transition import BasicFFN

DEFAULT_HASH = "df6e86bd9de4b5e8ac2d34bd1b32c6ea3aa1bae39de0f8264716e6460ef685cc"


def test_basic_ffn():
    input = generate_standard_inputs()
    model: DynaLM = generate_standard_lm()

    embedding, _, _ = model.embedding_stage(input, None, None, None)
    d_model = model.config.d_model
    d_ffn = model.config.d_ffn
    torch.manual_seed(42)
    ffn: BasicFFN = BasicFFN(d_model, d_ffn)
    collector = []
    ffn_output, _ = ffn(embedding, None, collector)
    print(collector)

    # No non-deterministic operations so hashing will work
    hash_value = hashlib.sha256(ffn_output.detach().numpy().tobytes()).hexdigest()
    assert hash_value == DEFAULT_HASH


if __name__ == "__main__" and "PYTEST_CURRENT_TEST" not in os.environ:
    test_basic_ffn()
