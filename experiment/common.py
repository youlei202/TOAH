"""experiment/common.py

Shared helpers for experiments.

We keep experiment scripts runnable without command-line arguments. Each script
imports these helpers to resolve device/dtype, enforce reproducibility, and run
an episode with consistent logging.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch

from toah.config import AlgoConfig, EnvConfig
from toah.env import EnvState, ISACEnv
from toah.utils.seed import set_seed


def resolve_device(spec: str) -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda")
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device spec: {spec}")


def resolve_dtype(spec: str) -> torch.dtype:
    if spec == "complex64":
        return torch.complex64
    if spec == "complex128":
        return torch.complex128
    raise ValueError(f"Unknown dtype spec: {spec}")


def scale_env_horizon(env_cfg: EnvConfig, n_slots: int, shock_len_frac: float = 0.08) -> EnvConfig:
    """Create a new EnvConfig with a different horizon, while keeping shocks aligned."""
    t = int(n_slots)
    comm_start = int(round(0.25 * t))
    sense_start = int(round(0.65 * t))
    shock_len = int(max(1, round(shock_len_frac * t)))
    return replace(
        env_cfg,
        n_slots=t,
        shock_comm_start=comm_start,
        shock_comm_len=shock_len,
        shock_sensing_start=sense_start,
        shock_sensing_len=shock_len,
    )


def apply_csi_error(state: EnvState, std: float, base_seed: int) -> EnvState:
    """Return a new EnvState with a noisy channel realization.

    We treat ``state.hs`` as the true channel, and we construct a noisy estimate
    that the algorithm uses for decision making. We keep the noise generation
    independent of the algorithm by seeding it with (base_seed, t).
    """
    if std <= 0.0:
        return state

    t = int(state.t)

    # A per-slot deterministic generator makes CSI errors comparable across methods.
    gen = torch.Generator(device=state.hs.device)
    gen.manual_seed(int(base_seed) + 1000003 * t)

    # Complex Gaussian noise (independent real/imag parts).
    shape = state.hs.shape
    noise_r = torch.randn(shape, device=state.hs.device, dtype=torch.float32, generator=gen)
    noise_i = torch.randn(shape, device=state.hs.device, dtype=torch.float32, generator=gen)
    noise = (noise_r + 1j * noise_i) / math.sqrt(2.0)
    noise = noise.to(dtype=state.hs.dtype)

    # Scale relative to the average channel magnitude.
    scale = torch.mean(torch.abs(state.hs)).clamp_min(1e-6)
    hs_est = state.hs + std * scale * noise

    return EnvState(
        t=state.t,
        hs=hs_est,
        steering=state.steering,
        queues=state.queues,
        arrivals=state.arrivals,
        noise_sense=state.noise_sense,
    )


def episode_out_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "result" / "table"


def run_episode(
    env_cfg: EnvConfig,
    algo_cfg: AlgoConfig,
    AlgoCls: Type[Any],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    method_kwargs: Optional[Dict[str, Any]] = None,
    csi_error_std: float = 0.0,
    q_deadline: float = 40.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run one online episode and log per-slot metrics."""
    method_kwargs = method_kwargs or {}

    set_seed(seed, deterministic=True)

    env = ISACEnv(env_cfg, device=device, dtype=dtype)
    state = env.reset()

    algo = AlgoCls(env_cfg, algo_cfg, device=device, dtype=dtype, **method_kwargs)
    algo.reset(state)

    # IMPORTANT: Keep per-slot metrics on-device during the run.
    # Converting tensors to CPU (or calling .item()) inside the slot loop
    # forces CPU↔GPU synchronization and can slow down experiments by orders
    # of magnitude. We only convert to CPU once per run at the end.

    alpha_prev: Optional[torch.Tensor] = None
    alpha_tv = torch.zeros((), device=device, dtype=torch.float32)
    switch_count = torch.zeros((), device=device, dtype=torch.int32)
    deadline_viol = torch.zeros((), device=device, dtype=torch.int32)

    ts_t: List[int] = []
    ts_alpha: List[torch.Tensor] = []
    ts_sumq: List[torch.Tensor] = []
    ts_maxq: List[torch.Tensor] = []
    ts_sense: List[torch.Tensor] = []
    ts_sumrate: List[torch.Tensor] = []
    ts_noise: List[float] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for _ in range(int(env_cfg.n_slots)):
        state_in = apply_csi_error(state, std=csi_error_std, base_seed=seed + 777)
        act = algo.step(state_in)
        state, info, done = env.step(act["ws"], act["vs"])

        alpha = act["alpha"]
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(float(alpha), device=device, dtype=torch.float32)
        else:
            alpha = alpha.to(device=device, dtype=torch.float32)

        if alpha_prev is not None:
            alpha_tv = alpha_tv + torch.abs(alpha - alpha_prev)
            prev_side = (alpha_prev > 0.5)
            side = (alpha > 0.5)
            switch_count = switch_count + (prev_side != side).to(torch.int32)
        alpha_prev = alpha

        queues = info["queues"].to(dtype=torch.float32)
        sum_q = torch.sum(queues)
        max_q = torch.max(queues)
        deadline_viol = deadline_viol + (max_q > q_deadline).to(torch.int32)

        rate = info["rate"].to(dtype=torch.float32)
        sum_rate = torch.sum(rate)
        sense_u = info["sense_u"].to(dtype=torch.float32)

        ts_t.append(int(info["t"]))
        ts_alpha.append(alpha.detach())
        ts_sumq.append(sum_q.detach())
        ts_maxq.append(max_q.detach())
        ts_sense.append(sense_u.detach())
        ts_sumrate.append(sum_rate.detach())
        ts_noise.append(float(info["noise_sense"]))

        if done:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time_s = float(time.perf_counter() - t_start)

    alpha_np = torch.stack(ts_alpha).cpu().numpy()
    sumq_np = torch.stack(ts_sumq).cpu().numpy()
    maxq_np = torch.stack(ts_maxq).cpu().numpy()
    sense_np = torch.stack(ts_sense).cpu().numpy()
    sumrate_np = torch.stack(ts_sumrate).cpu().numpy()

    df_ts = pd.DataFrame(
        {
            "t": np.asarray(ts_t, dtype=np.int64),
            "alpha": alpha_np,
            "sum_queue": sumq_np,
            "max_queue": maxq_np,
            "sense_u": sense_np,
            "sum_rate": sumrate_np,
            "noise_sense": np.asarray(ts_noise, dtype=np.float64),
        }
    )

    avg_sumq = float(np.mean(sumq_np))
    p95_maxq = float(np.nanpercentile(maxq_np, 95))
    avg_sense = float(np.mean(sense_np))
    p95_sense = float(np.nanpercentile(sense_np, 95))
    n_slots_eff = max(1, len(ts_t))
    avg_slot_ms = 1000.0 * total_time_s / float(n_slots_eff)

    summary = {
        "seed": int(seed),
        "method": str(getattr(algo, "name", AlgoCls.__name__)),
        "avg_sum_queue": avg_sumq,
        "p95_max_queue": p95_maxq,
        "avg_sense_u": avg_sense,
        "p95_sense_u": p95_sense,
        "alpha_total_variation": float(alpha_tv.detach().cpu().item()),
        "switch_count": int(switch_count.detach().cpu().item()),
        "deadline_violation_rate": float(deadline_viol.detach().cpu().item()) / float(max(1, len(df_ts))),
        "avg_slot_ms": avg_slot_ms,
        "total_time_s": total_time_s,
        "device": str(device),
        "csi_error_std": float(csi_error_std),
        "q_deadline": float(q_deadline),
    }
    return df_ts, summary
