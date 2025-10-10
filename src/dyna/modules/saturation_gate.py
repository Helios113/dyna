from __future__ import annotations

import torch
from torch.nn import Module


class SaturationGate(Module):
    def __init__(self, d_model, init_bias=2.0):
        """Initialize SaturationGate with configurable parameters."""
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model // 2, 1, bias=True),  # Enable bias
        )
        # Initialize with positive bias to encourage continuation
        with torch.no_grad():
            self.linear[-1].bias.fill_(init_bias)

    def forward(self, x):
        z = self.linear(x.detach()).squeeze(-1)
        g_hard = (z > 0).float()
        g_soft = torch.sigmoid(z)
        g = g_hard + (g_soft - g_soft.detach())
        return g
