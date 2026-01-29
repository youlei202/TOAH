from __future__ import annotations

import torch

from .mobility import Positions


class IntersectionMobilityModel:
    """Deterministic intersection mobility for an RSU deployment.

    We keep the road geometry simple on purpose. Users move along two orthogonal
    lanes that cross at the origin. The RSU stays at the origin with a fixed
    height. This model is sufficient for studying online ISAC under strong and
    fast channel drift, which is common in mobile computing scenarios.

    The interface matches :class:`~toah.env.mobility.MobilityModel`.
    """

    def __init__(
        self,
        area_size_m: float,
        rsu_height_m: float,
        n_users: int,
        device: torch.device,
    ) -> None:
        self.area = float(area_size_m)
        self.h = float(rsu_height_m)
        self.k = int(n_users)
        self.device = device

        # Lane offsets to avoid all vehicles passing exactly through the origin.
        self.lane_offset = 0.06 * self.area

        kx = (self.k + 1) // 2
        ky = self.k - kx

        # Vehicles along the x-lane (moving +x).
        if kx > 0:
            xs0 = torch.linspace(-0.45 * self.area, -0.15 * self.area, steps=kx, device=device)
            ys0 = -self.lane_offset * torch.ones(kx, device=device)
            lane_x = torch.stack([xs0, ys0, torch.zeros_like(xs0)], dim=-1)
        else:
            lane_x = torch.zeros((0, 3), device=device)

        # Vehicles along the y-lane (moving +y).
        if ky > 0:
            ys1 = torch.linspace(-0.45 * self.area, -0.15 * self.area, steps=ky, device=device)
            xs1 = self.lane_offset * torch.ones(ky, device=device)
            lane_y = torch.stack([xs1, ys1, torch.zeros_like(ys1)], dim=-1)
        else:
            lane_y = torch.zeros((0, 3), device=device)

        self.users0 = torch.cat([lane_x, lane_y], dim=0)

        # A separate sensing target (not necessarily one of the communication users).
        self.target0 = torch.tensor([-0.25 * self.area, 0.25 * self.area, 0.0], device=device)

    def _wrap(self, val: torch.Tensor, low: float, high: float) -> torch.Tensor:
        span = high - low
        return torch.remainder(val - low, span) + low

    def positions(self, t: int, n_slots: int, shock_sense_start: int, shock_sense_len: int) -> Positions:
        # RSU at the origin.
        rsu = torch.tensor([0.0, 0.0, self.h], device=self.device)

        # Speed in meters per slot (deterministic, scaled by the simulation horizon).
        v = 0.9 * self.area / float(max(1, n_slots))

        users = self.users0.clone()
        kx = (self.k + 1) // 2

        # Update x-lane vehicles.
        if kx > 0:
            xs = users[:kx, 0]
            xs = self._wrap(xs + v * float(t), low=-0.5 * self.area, high=0.5 * self.area)
            users[:kx, 0] = xs

        # Update y-lane vehicles.
        if self.k - kx > 0:
            ys = users[kx:, 1]
            ys = self._wrap(ys + v * float(t), low=-0.5 * self.area, high=0.5 * self.area)
            users[kx:, 1] = ys

        # Sensing target drifts and makes a sharper turn during the sensing shock.
        if shock_sense_start <= t < shock_sense_start + shock_sense_len:
            drift = torch.tensor([0.4 * v, -2.6 * v, 0.0], device=self.device)
        else:
            drift = torch.tensor([0.9 * v, -0.6 * v, 0.0], device=self.device)
        target = self.target0 + drift * float(t)

        return Positions(uav=rsu, users=users, target=target)
