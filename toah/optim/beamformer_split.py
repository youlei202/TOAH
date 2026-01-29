from __future__ import annotations

import math

import torch


class SplitBeamformer(torch.nn.Module):
    """Beamformer with a fixed power split between comm beams and a sensing beam.

    We keep the power split fixed to reduce coupling and keep the experiment stable.
    """

    def __init__(
        self,
        n_users: int,
        n_tx_ant: int,
        tx_power_w: float,
        sense_power_frac: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.k = int(n_users)
        self.m = int(n_tx_ant)
        self.p = float(tx_power_w)
        self.rho = float(sense_power_frac)
        self.dtype = dtype

        scale = 1.0 / math.sqrt(self.m)
        init_c = (torch.randn(self.k, self.m, dtype=self.dtype) + 1j * torch.randn(
            self.k, self.m, dtype=self.dtype
        )) * scale
        init_s = (torch.randn(self.m, dtype=self.dtype) + 1j * torch.randn(self.m, dtype=self.dtype)) * scale

        self.u_comm = torch.nn.Parameter(init_c)
        self.u_sense = torch.nn.Parameter(init_s)

    def forward(self) -> torch.Tensor:
        p_comm = self.p * (1.0 - self.rho)
        p_sense = self.p * self.rho

        comm_norm = torch.linalg.vector_norm(self.u_comm.reshape(-1), ord=2).clamp_min(1e-12)
        w_comm = self.u_comm * math.sqrt(p_comm) / comm_norm

        sense_norm = torch.linalg.vector_norm(self.u_sense, ord=2).clamp_min(1e-12)
        w_sense = self.u_sense * math.sqrt(p_sense) / sense_norm

        return torch.cat([w_comm, w_sense.view(1, -1)], dim=0)
