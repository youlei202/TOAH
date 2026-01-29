from __future__ import annotations

import math

import torch


class ChannelModel:
    """A simple geometry-driven Rician MIMO downlink channel."""

    def __init__(
        self,
        n_tx_ant: int,
        n_user_ant: int,
        rician_k: float,
        pathloss_exp: float,
        ref_pathloss_db: float,
        ula_spacing_wavelength: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.m = int(n_tx_ant)
        self.n = int(n_user_ant)
        self.k = float(rician_k)
        self.pl_exp = float(pathloss_exp)
        self.pl0_db = float(ref_pathloss_db)
        self.d = float(ula_spacing_wavelength)
        self.device = device
        self.dtype = dtype
        self._m_idx = torch.arange(self.m, device=device, dtype=torch.float32)

    def tx_ula(self, sin_theta: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * self.d * self._m_idx * sin_theta
        a = torch.exp(1j * phase).to(dtype=self.dtype)
        return a / torch.sqrt(torch.tensor(float(self.m), device=self.device))

    def pathloss_lin(self, dist_m: torch.Tensor) -> torch.Tensor:
        pl_db = self.pl0_db - 10.0 * self.pl_exp * torch.log10(dist_m.clamp_min(1.0))
        return torch.pow(10.0, pl_db / 10.0)

    def generate_user_channel(self, uav: torch.Tensor, user: torch.Tensor) -> torch.Tensor:
        vec = user - uav
        dist = torch.linalg.vector_norm(vec)
        sin_theta = (vec[1] / dist).clamp(-1.0, 1.0)
        a_tx = self.tx_ula(sin_theta)  # (M,)

        # Random Rx steering
        a_rx = torch.randn(self.n, device=self.device, dtype=self.dtype) + 1j * torch.randn(
            self.n, device=self.device, dtype=self.dtype
        )
        a_rx = a_rx / torch.linalg.vector_norm(a_rx)

        h_los = a_rx.view(self.n, 1) @ a_tx.conj().view(1, self.m)

        h_nlos = (torch.randn(self.n, self.m, device=self.device, dtype=self.dtype) + 1j * torch.randn(
            self.n, self.m, device=self.device, dtype=self.dtype
        )) / math.sqrt(2.0)

        k_lin = torch.tensor(self.k, device=self.device)
        h = torch.sqrt(k_lin / (k_lin + 1.0)) * h_los + torch.sqrt(1.0 / (k_lin + 1.0)) * h_nlos
        h = h * torch.sqrt(self.pathloss_lin(dist))
        return h

    def generate_all(self, uav: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        hs = [self.generate_user_channel(uav, users[k]) for k in range(users.shape[0])]
        return torch.stack(hs, dim=0)  # (K,N,M)

    def sensing_steering(self, uav: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        vec = target - uav
        dist = torch.linalg.vector_norm(vec)
        sin_theta = (vec[1] / dist).clamp(-1.0, 1.0)
        return self.tx_ula(sin_theta)
