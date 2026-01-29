from __future__ import annotations

import torch

from ..config import AlgoConfig, EnvConfig
from ..env.isac_env import EnvState
from ..optim.gate import SmoothGate
from ..optim.momentum import MomentumOptimizer
from .core import ISACCore
from .hypergradient import HyperGradient, NeumannConfig


class FixedSensingLeader(ISACCore):
    """Fixed hierarchy: sensing is always the leader (alpha=1)."""

    def __init__(self, env_cfg: EnvConfig, algo_cfg: AlgoConfig, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__(env_cfg, algo_cfg, device, dtype)
        self.hg = HyperGradient(NeumannConfig(steps=algo_cfg.neumann_steps, eta=algo_cfg.neumann_eta))

    @property
    def name(self) -> str:
        return "Fixed-SensingLeader"

    def step(self, state: EnvState) -> dict:
        # Followers
        self.track_comm_receivers(state, steps=self.algo_cfg.lower_steps_per_slot)

        ws = self.ws()
        vs = self.vs()

        f = self.loss_sensing_led(state, ws, vs)
        g = self.lower_comm(state, ws, vs)
        hg = self.hg.compute(
            f=f,
            g=g,
            x_params=[self.beamformer.u_comm, self.beamformer.u_sense],
            y_params=[self.receivers.v],
        )
        self.opt_u_comm.step(hg[0])
        self.opt_u_sense.step(hg[1])

        ws2 = self.ws()
        return {"ws": ws2.detach(), "vs": self.vs().detach(), "alpha": 1.0}


class FixedCommLeader(ISACCore):
    """Fixed hierarchy: communication is always the leader (alpha=0)."""

    def __init__(self, env_cfg: EnvConfig, algo_cfg: AlgoConfig, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__(env_cfg, algo_cfg, device, dtype)
        self.hg = HyperGradient(NeumannConfig(steps=algo_cfg.neumann_steps, eta=algo_cfg.neumann_eta))

    @property
    def name(self) -> str:
        return "Fixed-CommLeader"

    def step(self, state: EnvState) -> dict:
        # Followers
        self.track_comm_receivers(state, steps=self.algo_cfg.lower_steps_per_slot)
        self.track_sensing_beam(state, steps=self.algo_cfg.lower_steps_per_slot, scale=1.0)

        ws = self.ws()
        vs = self.vs()

        f = self.loss_comm_led(state, ws, vs)
        g = self.lower_sense(state, ws)
        hg = self.hg.compute(
            f=f,
            g=g,
            x_params=[self.beamformer.u_comm],
            y_params=[self.beamformer.u_sense],
        )
        self.opt_u_comm.step(hg[0])

        ws2 = self.ws()
        return {"ws": ws2.detach(), "vs": self.vs().detach(), "alpha": 0.0}


class HeuristicGate(ISACCore):
    """A heuristic gate based on queue threshold, then bilevel updates."""

    def __init__(self, env_cfg: EnvConfig, algo_cfg: AlgoConfig, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__(env_cfg, algo_cfg, device, dtype)
        self.hg = HyperGradient(NeumannConfig(steps=algo_cfg.neumann_steps, eta=algo_cfg.neumann_eta))

    @property
    def name(self) -> str:
        return "HeuristicGate"

    def _alpha(self, state: EnvState) -> float:
        return 0.0 if float(torch.max(state.queues).item()) > self.algo_cfg.heuristic_q_th else 1.0

    def step(self, state: EnvState) -> dict:
        alpha = self._alpha(state)

        self.track_comm_receivers(state, steps=self.algo_cfg.lower_steps_per_slot)
        if alpha < 0.5:
            self.track_sensing_beam(state, steps=self.algo_cfg.lower_steps_per_slot, scale=1.0)

        ws = self.ws()
        vs = self.vs()

        if alpha > 0.5:
            f = self.loss_sensing_led(state, ws, vs)
            g = self.lower_comm(state, ws, vs)
            hg = self.hg.compute(
                f=f,
                g=g,
                x_params=[self.beamformer.u_comm, self.beamformer.u_sense],
                y_params=[self.receivers.v],
            )
            self.opt_u_comm.step(hg[0])
            self.opt_u_sense.step(hg[1])
        else:
            f = self.loss_comm_led(state, ws, vs)
            g = self.lower_sense(state, ws)
            hg = self.hg.compute(
                f=f,
                g=g,
                x_params=[self.beamformer.u_comm],
                y_params=[self.beamformer.u_sense],
            )
            self.opt_u_comm.step(hg[0])

        ws2 = self.ws()
        return {"ws": ws2.detach(), "vs": self.vs().detach(), "alpha": float(alpha)}


class TOAHNoHG(ISACCore):
    """Ablation: keep adaptive gate but remove implicit hypergradient terms.

    We stop gradients through follower variables when updating leader variables.
    """

    def __init__(self, env_cfg: EnvConfig, algo_cfg: AlgoConfig, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__(env_cfg, algo_cfg, device, dtype)
        self.gate = SmoothGate(init_alpha=0.5, temperature=algo_cfg.alpha_temperature).to(device)
        self.opt_logit = MomentumOptimizer(self.gate.logit, lr=algo_cfg.lr_alpha, momentum=algo_cfg.momentum_alpha)
        self._alpha_prev = torch.tensor(0.5, device=device, dtype=torch.float32)

    @property
    def name(self) -> str:
        return "TOAH-NoHG"

    def reset(self, state: EnvState) -> None:
        super().reset(state)
        with torch.no_grad():
            self._alpha_prev.fill_(0.5)

    def step(self, state: EnvState) -> dict:
        alpha = self.gate()

        # Followers still track as in TOAH
        self.track_comm_receivers(state, steps=self.algo_cfg.lower_steps_per_slot)
        self.track_sensing_beam(
            state,
            steps=self.algo_cfg.lower_steps_per_slot,
            scale=float((1.0 - alpha.detach()).clamp(0.0, 1.0).item()),
        )

        # Leader gradients: detach follower variables
        ws = self.ws()
        vs_detached = self.vs().detach()

        f_s = self.loss_sensing_led(state, ws, vs_detached)
        grad_s = torch.autograd.grad(f_s, [self.beamformer.u_comm, self.beamformer.u_sense], retain_graph=True)

        # Comm-led: stop gradient through the sensing-beam parameter (follower)
        ws_c = self.ws_detach_sense()
        f_c = self.loss_comm_led(state, ws_c, vs_detached)
        grad_c = torch.autograd.grad(f_c, [self.beamformer.u_comm], retain_graph=True)

        grad_u_comm = alpha * grad_s[0] + (1.0 - alpha) * grad_c[0]
        grad_u_sense = alpha * grad_s[1]

        self.opt_u_comm.step(grad_u_comm)
        self.opt_u_sense.step(grad_u_sense)

        # Gate update
        ws2 = self.ws()
        vs2 = self.vs().detach()
        f_s2 = self.loss_sensing_led(state, ws2, vs2)
        f_c2 = self.loss_comm_led(state, self.ws_detach_sense(), vs2)
        alpha2 = self.gate()
        switch = self.algo_cfg.switch_penalty * (alpha2 - self._alpha_prev.detach()) ** 2
        obj = alpha2 * f_s2 + (1.0 - alpha2) * f_c2 + switch
        grad_logit = torch.autograd.grad(obj, self.gate.logit, retain_graph=False)[0]
        self.opt_logit.step(grad_logit)
        with torch.no_grad():
            self._alpha_prev.copy_(alpha2.detach())

        return {"ws": self.ws().detach(), "vs": self.vs().detach(), "alpha": float(alpha2.detach().cpu())}


class AOHeuristic(ISACCore):
    """Alternating optimization baseline with heuristic mode selection.

    Per slot, we alternate between follower updates and leader updates for a few rounds.
    This is intentionally slower than single-loop methods.
    """

    def __init__(self, env_cfg: EnvConfig, algo_cfg: AlgoConfig, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__(env_cfg, algo_cfg, device, dtype)
        self.outer_rounds = 5

    @property
    def name(self) -> str:
        return "AO-Heuristic"

    def _alpha(self, state: EnvState) -> float:
        return 0.0 if float(torch.max(state.queues).item()) > self.algo_cfg.heuristic_q_th else 1.0

    def step(self, state: EnvState) -> dict:
        alpha = self._alpha(state)

        for _ in range(self.outer_rounds):
            # Inner loop: followers to (approximate) convergence
            self.track_comm_receivers(state, steps=self.algo_cfg.lower_steps_ao)
            if alpha < 0.5:
                self.track_sensing_beam(state, steps=self.algo_cfg.lower_steps_ao, scale=1.0)

            ws = self.ws()
            vs = self.vs().detach()  # AO treats follower fixed during leader step

            if alpha > 0.5:
                f = self.loss_sensing_led(state, ws, vs)
                grad = torch.autograd.grad(f, [self.beamformer.u_comm, self.beamformer.u_sense], retain_graph=False)
                self.opt_u_comm.step(grad[0])
                self.opt_u_sense.step(grad[1])
            else:
                f = self.loss_comm_led(state, ws, vs)
                grad = torch.autograd.grad(f, [self.beamformer.u_comm], retain_graph=False)
                self.opt_u_comm.step(grad[0])

        return {"ws": self.ws().detach(), "vs": self.vs().detach(), "alpha": float(alpha)}
