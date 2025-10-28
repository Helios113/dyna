from __future__ import annotations

import torch
from torch.nn import Module


class SaturationGate(Module):
    def __init__(self, d_model, init_bias=0.0):
        """Initialize SaturationGate with configurable parameters."""
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model // 2, 1),  # Enable bias
        )

    def forward(self, x):
        print("SaturationGate: Forward", x, flush=True)
        z = self.linear(x.detach()).squeeze(-1)
        g_hard = (z > 0).float()
        g_soft = torch.sigmoid(z)
        g = g_hard + (g_soft - g_soft.detach())
        print(f"SaturationGate: {g.mean().item():.4f}")
        print(f"SaturationGate: {g_soft.mean().item():.4f}")
        return g
