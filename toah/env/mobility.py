from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Positions:
    """3D positions in meters."""

    uav: torch.Tensor      # (3,)
    users: torch.Tensor    # (K,3)
    target: torch.Tensor   # (3,)


class MobilityModel:
    """Deterministic mobility for reproducible experiments.

    We keep users fixed to isolate the effect of UAV mobility and task shocks.
    """

    def __init__(
        self,
        area_size_m: float,
        uav_altitude_m: float,
        n_users: int,
        device: torch.device,
    ) -> None:
        self.area = float(area_size_m)
        self.alt = float(uav_altitude_m)
        self.k = int(n_users)
        self.device = device

        grid = torch.linspace(-0.4 * self.area, 0.4 * self.area, steps=self.k, device=device)
        xs = grid
        ys = torch.flip(grid, dims=[0])
        self.users0 = torch.stack([xs, ys, torch.zeros_like(xs)], dim=-1)

        self.target0 = torch.tensor([0.3 * self.area, 0.0, 0.0], device=device)

    def positions(self, t: int, n_slots: int, shock_sense_start: int, shock_sense_len: int) -> Positions:
        # UAV flies a circle around the origin.
        frac = float(t) / float(max(1, n_slots))
        ang = 2.0 * torch.pi * torch.tensor(frac, device=self.device)
        radius = 0.45 * self.area
        uav = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang), torch.tensor(self.alt, device=self.device)])

        users = self.users0

        # Target drifts. During sensing shock, it turns faster.
        v = 0.4 * self.area / max(1, n_slots)
        if shock_sense_start <= t < shock_sense_start + shock_sense_len:
            drift = torch.tensor([v, 2.5 * v, 0.0], device=self.device)
        else:
            drift = torch.tensor([v, 0.5 * v, 0.0], device=self.device)
        target = self.target0 + drift * float(t)
        return Positions(uav=uav, users=users, target=target)
