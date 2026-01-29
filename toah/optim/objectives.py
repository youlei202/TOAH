from __future__ import annotations

import torch

from ..utils.complex import hermitian


def comm_sum_mse(hs: torch.Tensor, ws: torch.Tensor, vs: torch.Tensor, noise_var: float) -> torch.Tensor:
    """Lower-level objective: sum MSE over users."""
    k_users = hs.shape[0]
    w_comm = ws[:k_users]
    total = torch.zeros((), device=hs.device, dtype=torch.float32)
    for k in range(k_users):
        h = hs[k]
        v = vs[k].view(-1, 1)
        hw = h @ w_comm.transpose(0, 1)  # (N,K)
        cov = hw @ hermitian(hw) + noise_var * torch.eye(h.shape[0], device=hs.device, dtype=ws.dtype)
        desired = (hermitian(v) @ (h @ w_comm[k].view(-1, 1))).view(())
        mse = 1.0 - 2.0 * torch.real(desired) + torch.real((hermitian(v) @ (cov @ v))).view(())
        total = total + mse
    return total
