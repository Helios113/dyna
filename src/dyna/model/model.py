from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dyna.config import ModelConfig


class ComposerDynaModel(nn.Module):
    def __init__(self, config: ModelConfig, tokenizer: Any | None = None):
        """Initialize ComposerDynaModel with config and optional tokenizer."""
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        # trivial single linear head to keep shape correctness in trainers
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)
        logits = self.lm_head(x)
        return type("Out", (), {"logits": logits})
