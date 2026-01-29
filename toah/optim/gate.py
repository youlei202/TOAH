from __future__ import annotations

import torch


class SmoothGate(torch.nn.Module):
    """A smooth gate alpha in (0,1) parameterized by a scalar logit."""

    def __init__(self, init_alpha: float, temperature: float) -> None:
        super().__init__()
        init_alpha = float(init_alpha)
        init_alpha = min(max(init_alpha, 1e-3), 1.0 - 1e-3)
        logit = torch.log(torch.tensor(init_alpha) / (1.0 - torch.tensor(init_alpha)))
        self.logit = torch.nn.Parameter(logit.view(()))
        self.temperature = float(temperature)

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.temperature * self.logit)
