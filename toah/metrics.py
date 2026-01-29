from __future__ import annotations

import torch

from .utils.complex import hermitian, safe_log2


def user_effective_channels(h: torch.Tensor, w: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute v^H H w for a batch of beams.

    Args:
        h: (N,M)
        w: (B,M)
        v: (N,)
    Returns:
        (B,) complex
    """
    hw = torch.matmul(h, w.transpose(0, 1))  # (N,B)
    return torch.matmul(hermitian(v.view(-1, 1)), hw).view(-1)


def sinr_and_rate(hs: torch.Tensor, ws: torch.Tensor, vs: torch.Tensor, noise_var: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-user SINR and Shannon rate.

    Args:
        hs: (K,N,M)
        ws: (K+1,M) where last beam is sensing beam
        vs: (K,N)
    Returns:
        sinr: (K,)
        rate: (K,)
    """
    k_users = hs.shape[0]
    w_comm = ws[:k_users]
    sinrs, rates = [], []
    for k in range(k_users):
        h = hs[k]
        v = vs[k]
        eff = user_effective_channels(h, w_comm, v)  # (K,)
        signal = torch.abs(eff[k]) ** 2
        interf = torch.sum(torch.abs(eff) ** 2) - signal
        noise = torch.tensor(float(noise_var), device=hs.device) * torch.sum(torch.real(v.conj() * v))
        sinr = signal / (interf + noise + 1e-12)
        rate = safe_log2(1.0 + sinr)
        sinrs.append(sinr)
        rates.append(rate)
    return torch.stack(sinrs, dim=0), torch.stack(rates, dim=0)


def sensing_scnr(a: torch.Tensor, ws: torch.Tensor, noise_var: float) -> torch.Tensor:
    """Sensing SCNR proxy.

    We use a single steering vector a and treat comm beams as interference.
    """
    proj = torch.matmul(a.conj().view(1, -1), ws.transpose(0, 1)).view(-1)  # (K+1,)
    sig = torch.abs(proj[-1]) ** 2
    interf = torch.sum(torch.abs(proj[:-1]) ** 2)
    noise = torch.tensor(float(noise_var), device=ws.device)
    return sig / (interf + noise + 1e-12)


def sensing_loss(a: torch.Tensor, ws: torch.Tensor, noise_var: float) -> torch.Tensor:
    """Sensing uncertainty proxy: u = 1/(SCNR+eps)."""
    scnr = sensing_scnr(a, ws, noise_var=noise_var)
    return 1.0 / (scnr + 1e-6)
