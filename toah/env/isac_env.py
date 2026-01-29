from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import EnvConfig
from ..metrics import sensing_loss, sinr_and_rate
from .channel import ChannelModel
from .mobility import MobilityModel, Positions


@dataclass
class EnvState:
    t: int
    hs: torch.Tensor         # (K,N,M)
    steering: torch.Tensor   # (M,)
    queues: torch.Tensor     # (K,)
    arrivals: torch.Tensor   # (K,)
    noise_sense: float


class ISACEnv:
    """Online ISAC environment with mobility, traffic shocks, and sensing shocks."""

    def __init__(self, cfg: EnvConfig, device: torch.device, dtype: torch.dtype) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        self.mob = MobilityModel(
            area_size_m=cfg.area_size_m,
            uav_altitude_m=cfg.uav_altitude_m,
            n_users=cfg.n_users,
            device=device,
        )
        self.chan = ChannelModel(
            n_tx_ant=cfg.n_tx_ant,
            n_user_ant=cfg.n_user_ant,
            rician_k=cfg.rician_k_factor,
            pathloss_exp=cfg.pathloss_exp,
            ref_pathloss_db=cfg.ref_pathloss_db,
            ula_spacing_wavelength=cfg.ula_spacing_wavelength,
            device=device,
            dtype=dtype,
        )

        self._t = 0
        self._queues = torch.zeros(cfg.n_users, device=device, dtype=torch.float32)
        self._state: EnvState | None = None

    def reset(self) -> EnvState:
        self._t = 0
        self._queues = torch.zeros(self.cfg.n_users, device=self.device, dtype=torch.float32)
        self._state = self._make_state()
        return self._state

    def _arrivals(self, t: int) -> torch.Tensor:
        base = self.cfg.arrival_rate
        lam = base * torch.ones(self.cfg.n_users, device=self.device, dtype=torch.float32)
        if self.cfg.shock_comm_start <= t < self.cfg.shock_comm_start + self.cfg.shock_comm_len:
            lam[0] = base * self.cfg.shock_comm_gain
        return torch.poisson(lam)

    def _noise_sense(self, t: int) -> float:
        noise = self.cfg.noise_sensing_w
        if self.cfg.shock_sensing_start <= t < self.cfg.shock_sensing_start + self.cfg.shock_sensing_len:
            noise = noise * self.cfg.shock_sensing_noise_gain
        return float(noise)

    def _make_state(self) -> EnvState:
        pos: Positions = self.mob.positions(
            t=self._t,
            n_slots=self.cfg.n_slots,
            shock_sense_start=self.cfg.shock_sensing_start,
            shock_sense_len=self.cfg.shock_sensing_len,
        )
        hs = self.chan.generate_all(pos.uav, pos.users)
        steering = self.chan.sensing_steering(pos.uav, pos.target)
        arrivals = self._arrivals(self._t)
        noise_sense = self._noise_sense(self._t)
        return EnvState(
            t=self._t,
            hs=hs,
            steering=steering,
            queues=self._queues.clone(),
            arrivals=arrivals,
            noise_sense=noise_sense,
        )

    @torch.no_grad()
    def step(self, ws: torch.Tensor, vs: torch.Tensor) -> tuple[EnvState, dict, bool]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        cur = self._state

        sinr, rate = sinr_and_rate(cur.hs, ws, vs, noise_var=self.cfg.noise_comm_w)
        served = self.cfg.service_scale * rate

        self._queues = torch.clamp(self._queues + cur.arrivals - served, min=0.0)

        sense_u = sensing_loss(cur.steering, ws, noise_var=cur.noise_sense)
        info = {
            "t": int(cur.t),
            "sinr": sinr.detach().cpu(),
            "rate": rate.detach().cpu(),
            "served": served.detach().cpu(),
            "queues": self._queues.detach().cpu(),
            "arrivals": cur.arrivals.detach().cpu(),
            "sense_u": float(sense_u.detach().cpu()),
            "noise_sense": float(cur.noise_sense),
        }

        self._t += 1
        done = self._t >= self.cfg.n_slots
        if done:
            nxt = EnvState(
                t=self._t,
                hs=cur.hs,
                steering=cur.steering,
                queues=self._queues.clone(),
                arrivals=torch.zeros_like(self._queues),
                noise_sense=float(cur.noise_sense),
            )
        else:
            nxt = self._make_state()
        self._state = nxt
        return nxt, info, done
