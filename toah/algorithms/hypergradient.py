from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class NeumannConfig:
    steps: int
    eta: float


def _real_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Real-valued dot product for complex tensors."""
    return torch.real(torch.sum(a.conj() * b))


class HyperGradient:
    """Implicit hypergradient using a Neumann-series Hessian inverse approximation."""

    def __init__(self, cfg: NeumannConfig) -> None:
        self.cfg = cfg

    def hvp(
        self,
        grad_g_y: Sequence[torch.Tensor],
        y_params: Sequence[torch.Tensor],
        vec: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        scalar = torch.zeros((), device=grad_g_y[0].device, dtype=torch.float32)
        for gg, vv in zip(grad_g_y, vec):
            scalar = scalar + _real_dot(gg, vv)
        hvps = torch.autograd.grad(scalar, y_params, retain_graph=True, create_graph=True)
        return list(hvps)

    def inv_hessian_times(
        self,
        grad_g_y: Sequence[torch.Tensor],
        y_params: Sequence[torch.Tensor],
        v: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        eta = float(self.cfg.eta)
        steps = int(self.cfg.steps)

        v0 = [vv for vv in v]
        p = [eta * vv for vv in v0]
        cur = [vv for vv in v0]
        for _ in range(1, steps):
            hv = self.hvp(grad_g_y, y_params, cur)
            cur = [c - eta * h for c, h in zip(cur, hv)]
            p = [pp + eta * c for pp, c in zip(p, cur)]
        return p

    def compute(
        self,
        f: torch.Tensor,
        g: torch.Tensor,
        x_params: Sequence[torch.Tensor],
        y_params: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Hypergradient for min_x f(x, y*(x)), y*(x)=argmin_y g(x,y)."""
        grad_f_x = torch.autograd.grad(f, x_params, retain_graph=True, create_graph=True)
        grad_f_y = torch.autograd.grad(f, y_params, retain_graph=True, create_graph=True)
        grad_g_y = torch.autograd.grad(g, y_params, retain_graph=True, create_graph=True)

        p = self.inv_hessian_times(grad_g_y, y_params, list(grad_f_y))

        cross_scalar = torch.zeros((), device=f.device, dtype=torch.float32)
        for gg, pp in zip(grad_g_y, p):
            cross_scalar = cross_scalar + _real_dot(gg, pp)
        grad_cross_x = torch.autograd.grad(cross_scalar, x_params, retain_graph=True, create_graph=True)
        return [gx - gc for gx, gc in zip(grad_f_x, grad_cross_x)]
