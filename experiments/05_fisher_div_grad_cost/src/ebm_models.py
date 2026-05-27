"""Small EBM energy networks shared across the PyTorch + JAX benchmarks.

We deliberately keep these dependency-light and small so that the dominant
cost is the autodiff over input x (which is the whole point of the
experiment), not framework / model bloat.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class TinyEBM(nn.Module):
    """E_theta : R^D -> R, a simple MLP energy with a sum-pool readout.

    Parameter count is roughly (D*H + H*H + H), with H = hidden_dim.
    """

    def __init__(self, dim: int, hidden_dim: int = 64, num_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.SiLU())
            last = hidden_dim
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
