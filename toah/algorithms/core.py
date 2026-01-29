from __future__ import annotations

import torch

from ..config import AlgoConfig, EnvConfig
from ..env.isac_env import EnvState
from ..metrics import sensing_loss, sinr_and_rate
from ..optim.beamformer_split import SplitBeamformer
from ..optim.momentum import MomentumOptimizer
from ..optim.objectives import comm_sum_mse
from ..optim.receivers import Receivers
from ..utils.complex import fro_norm_sq
from .base import OnlineAlgorithm


class ISACCore(OnlineAlgorithm):
    """Shared components for all online algorithms in this repository."""

    def __init__(self, env_cfg: EnvConfig, algo_cfg: AlgoConfig, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__(env_cfg, algo_cfg, device, dtype)

        self.beamformer = SplitBeamformer(
            n_users=env_cfg.n_users,
            n_tx_ant=env_cfg.n_tx_ant,
            tx_power_w=env_cfg.tx_power_w,
            sense_power_frac=algo_cfg.sense_power_frac,
            dtype=dtype,
        ).to(device)

        self.receivers = Receivers(env_cfg.n_users, env_cfg.n_user_ant, dtype=dtype).to(device)

        self.opt_u_comm = MomentumOptimizer(self.beamformer.u_comm, lr=algo_cfg.lr_w, momentum=algo_cfg.momentum_w)
        self.opt_u_sense = MomentumOptimizer(self.beamformer.u_sense, lr=algo_cfg.lr_w, momentum=algo_cfg.momentum_w)
        self.opt_v = MomentumOptimizer(self.receivers.v, lr=algo_cfg.lr_v, momentum=algo_cfg.momentum_v)

    def reset(self, state: EnvState) -> None:
        with torch.no_grad():
            ws = self.beamformer()
            self.receivers.set_mmse(state.hs, ws, noise_var=self.env_cfg.noise_comm_w)

    def ws(self) -> torch.Tensor:
        return self.beamformer()

    def vs(self) -> torch.Tensor:
        return self.receivers()

    def ws_detach_sense(self) -> torch.Tensor:
        """Beamformer output that stops gradient through the sensing beam parameter."""
        u_comm = self.beamformer.u_comm
        u_sense = self.beamformer.u_sense.detach()
        p_comm = self.env_cfg.tx_power_w * (1.0 - self.algo_cfg.sense_power_frac)
        p_sense = self.env_cfg.tx_power_w * self.algo_cfg.sense_power_frac

        comm_norm = torch.linalg.vector_norm(u_comm.reshape(-1), ord=2).clamp_min(1e-12)
        w_comm = u_comm * torch.sqrt(torch.tensor(p_comm, device=u_comm.device)) / comm_norm

        sense_norm = torch.linalg.vector_norm(u_sense, ord=2).clamp_min(1e-12)
        w_sense = u_sense * torch.sqrt(torch.tensor(p_sense, device=u_comm.device)) / sense_norm
        return torch.cat([w_comm, w_sense.view(1, -1)], dim=0)

    def comm_drift_loss(self, state: EnvState, ws: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
        _, rate = sinr_and_rate(state.hs, ws, vs, noise_var=self.env_cfg.noise_comm_w)
        q_sum = torch.sum(state.queues)
        return self.algo_cfg.queue_weight * q_sum - self.algo_cfg.throughput_weight * torch.sum(state.queues * rate)

    def loss_sensing_led(self, state: EnvState, ws: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
        sense_u = sensing_loss(state.steering, ws, noise_var=state.noise_sense)
        comm = self.comm_drift_loss(state, ws, vs)
        return self.algo_cfg.sensing_weight * sense_u + self.algo_cfg.secondary_weight * comm

    def loss_comm_led(self, state: EnvState, ws: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
        sense_u = sensing_loss(state.steering, ws, noise_var=state.noise_sense)
        comm = self.comm_drift_loss(state, ws, vs)
        return comm + self.algo_cfg.secondary_weight * self.algo_cfg.sensing_weight * sense_u

    def lower_comm(self, state: EnvState, ws: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
        g = comm_sum_mse(state.hs, ws, vs, noise_var=self.env_cfg.noise_comm_w)
        return g + self.algo_cfg.reg_v * fro_norm_sq(self.receivers.v)

    def lower_sense(self, state: EnvState, ws: torch.Tensor) -> torch.Tensor:
        g = sensing_loss(state.steering, ws, noise_var=state.noise_sense)
        reg = self.algo_cfg.reg_sense * torch.real(torch.sum(self.beamformer.u_sense.conj() * self.beamformer.u_sense))
        return g + reg

    def track_comm_receivers(self, state: EnvState, steps: int) -> None:
        for _ in range(int(steps)):
            ws = self.ws()
            vs = self.vs()
            g = self.lower_comm(state, ws, vs)
            grad_v = torch.autograd.grad(g, self.receivers.v, retain_graph=False, create_graph=False)[0]
            self.opt_v.step(grad_v)

    def track_sensing_beam(self, state: EnvState, steps: int, scale: float = 1.0) -> None:
        if scale <= 0.0:
            return
        for _ in range(int(steps)):
            ws = self.ws()
            g = self.lower_sense(state, ws)
            grad_us = torch.autograd.grad(g, self.beamformer.u_sense, retain_graph=False, create_graph=False)[0]
            self.opt_u_sense.step(scale * grad_us)
