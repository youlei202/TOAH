from __future__ import annotations

import torch

from ..config import AlgoConfig, EnvConfig
from ..env.isac_env import EnvState
from ..optim.gate import SmoothGate
from ..optim.momentum import MomentumOptimizer
from .core import ISACCore
from .hypergradient import HyperGradient, NeumannConfig


class TOAH(ISACCore):
    """Task-Oriented Adaptive Hierarchy (TOAH) with implicit hypergradients."""

    def __init__(self, env_cfg: EnvConfig, algo_cfg: AlgoConfig, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__(env_cfg, algo_cfg, device, dtype)

        self.gate = SmoothGate(init_alpha=0.5, temperature=algo_cfg.alpha_temperature).to(device)
        self.opt_logit = MomentumOptimizer(self.gate.logit, lr=algo_cfg.lr_alpha, momentum=algo_cfg.momentum_alpha)
        self.hg = HyperGradient(NeumannConfig(steps=algo_cfg.neumann_steps, eta=algo_cfg.neumann_eta))
        self._alpha_prev = torch.tensor(0.5, device=device, dtype=torch.float32)

    @property
    def name(self) -> str:
        return "TOAH"

    def reset(self, state: EnvState) -> None:
        super().reset(state)
        with torch.no_grad():
            self._alpha_prev.fill_(0.5)

    def step(self, state: EnvState) -> dict:
        alpha = self.gate()

        # 1) follower tracking
        self.track_comm_receivers(state, steps=self.algo_cfg.lower_steps_per_slot)
        # Sensing beam tracks mainly in comm-led mode.
        self.track_sensing_beam(
            state,
            steps=self.algo_cfg.lower_steps_per_slot,
            scale=float((1.0 - alpha.detach()).clamp(0.0, 1.0).item()),
        )

        # 2) hypergradients for leader updates
        ws = self.ws()
        vs = self.vs()

        # Sensing-led: x=(u_comm,u_sense), y=v
        f_s = self.loss_sensing_led(state, ws, vs)
        g_c = self.lower_comm(state, ws, vs)
        hg_s = self.hg.compute(
            f=f_s,
            g=g_c,
            x_params=[self.beamformer.u_comm, self.beamformer.u_sense],
            y_params=[self.receivers.v],
        )

        # Comm-led: x=(u_comm), y=u_sense
        f_c = self.loss_comm_led(state, ws, vs)
        g_s = self.lower_sense(state, ws)
        hg_c = self.hg.compute(
            f=f_c,
            g=g_s,
            x_params=[self.beamformer.u_comm],
            y_params=[self.beamformer.u_sense],
        )

        grad_u_comm = alpha * hg_s[0] + (1.0 - alpha) * hg_c[0]
        grad_u_sense = alpha * hg_s[1]

        self.opt_u_comm.step(grad_u_comm)
        self.opt_u_sense.step(grad_u_sense)

        # 3) gate update with switching penalty
        ws2 = self.ws()
        vs2 = self.vs()
        f_s2 = self.loss_sensing_led(state, ws2, vs2)
        f_c2 = self.loss_comm_led(state, ws2, vs2)
        alpha2 = self.gate()

        switch = self.algo_cfg.switch_penalty * (alpha2 - self._alpha_prev.detach()) ** 2
        obj = alpha2 * f_s2 + (1.0 - alpha2) * f_c2 + switch
        grad_logit = torch.autograd.grad(obj, self.gate.logit, retain_graph=False, create_graph=False)[0]
        self.opt_logit.step(grad_logit)

        with torch.no_grad():
            self._alpha_prev.copy_(alpha2.detach())

        return {"ws": ws2.detach(), "vs": self.vs().detach(), "alpha": float(alpha2.detach().cpu())}
