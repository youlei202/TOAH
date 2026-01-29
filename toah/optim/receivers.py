from __future__ import annotations

import torch

from ..utils.complex import hermitian


class Receivers(torch.nn.Module):
    """Per-user linear receivers v_k \in C^N."""

    def __init__(self, n_users: int, n_user_ant: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.k = int(n_users)
        self.n = int(n_user_ant)
        self.dtype = dtype

        init = torch.randn(self.k, self.n, dtype=dtype) + 1j * torch.randn(self.k, self.n, dtype=dtype)
        init = init / torch.linalg.vector_norm(init, dim=-1, keepdim=True).clamp_min(1e-12)
        self.v = torch.nn.Parameter(init)

    def forward(self) -> torch.Tensor:
        return self.v / torch.linalg.vector_norm(self.v, dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def set_mmse(self, hs: torch.Tensor, ws: torch.Tensor, noise_var: float) -> None:
        k_users = hs.shape[0]
        w_comm = ws[:k_users]
        vs = []
        for k in range(k_users):
            h = hs[k]
            hk_w = h @ w_comm.transpose(0, 1)  # (N,K)
            cov = hk_w @ hermitian(hk_w) + noise_var * torch.eye(h.shape[0], device=hs.device, dtype=ws.dtype)
            v = torch.linalg.solve(cov, hk_w[:, k])
            vs.append(v)
        v_mat = torch.stack(vs, dim=0)
        v_mat = v_mat / torch.linalg.vector_norm(v_mat, dim=-1, keepdim=True).clamp_min(1e-12)
        self.v.copy_(v_mat)
