from __future__ import annotations

import torch


def hermitian(x: torch.Tensor) -> torch.Tensor:
    """Hermitian (conjugate transpose) for 2D tensors."""
    return x.conj().transpose(-2, -1)


def fro_norm_sq(x: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius norm, returned as a real scalar."""
    return torch.sum(torch.real(x.conj() * x))


def safe_log2(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Numerically safe log2 for nonnegative tensors."""
    return torch.log(x.clamp_min(eps)) / torch.log(torch.tensor(2.0, device=x.device))
