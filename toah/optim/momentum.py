from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MomentumState:
    buf: torch.Tensor


class MomentumOptimizer:
    """A minimal momentum optimizer (for reproducible experiments)."""

    def __init__(self, param: torch.nn.Parameter, lr: float, momentum: float) -> None:
        self.param = param
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.state = MomentumState(buf=torch.zeros_like(param.data))

    @torch.no_grad()
    def step(self, grad: torch.Tensor) -> None:
        if grad is None:
            return
        self.state.buf.mul_(self.momentum).add_(grad, alpha=1.0)
        self.param.data.add_(self.state.buf, alpha=-self.lr)
