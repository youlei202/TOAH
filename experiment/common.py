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

    alpha_prev: Optional[float] = None
    alpha_tv = 0.0
    switch_count = 0
    deadline_viol = 0

    rows: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    done = False
    while not done:
        t0 = time.perf_counter()

        state_in = apply_csi_error(state, std=csi_error_std, base_seed=seed + 777)
        act = algo.step(state_in)

        state, info, done = env.step(act["ws"], act["vs"])
        t1 = time.perf_counter()

        alpha = float(act["alpha"])
        if alpha_prev is not None:
            alpha_tv += abs(alpha - alpha_prev)
            if (alpha_prev <= 0.5 < alpha) or (alpha <= 0.5 < alpha_prev):
                switch_count += 1
        alpha_prev = alpha

        queues = torch.as_tensor(info["queues"])
        sum_q = float(torch.sum(queues).item())
        max_q = float(torch.max(queues).item())
        if max_q > q_deadline:
            deadline_viol += 1

        sum_rate = float(torch.sum(torch.as_tensor(info["rate"])).item())

        rows.append(
            {
                "t": int(info["t"]),
                "alpha": alpha,
                "sum_queue": sum_q,
                "max_queue": max_q,
                "sense_u": float(info["sense_u"]),
                "sum_rate": sum_rate,
                "noise_sense": float(info["noise_sense"]),
                "slot_ms": 1000.0 * (t1 - t0),
            }
        )

    total_time_s = float(time.perf_counter() - t_start)

    df_ts = pd.DataFrame(rows)
    avg_sumq = float(df_ts["sum_queue"].mean())
    p95_maxq = float(np.percentile(df_ts["max_queue"], 95))
    avg_sense = float(df_ts["sense_u"].mean())
    p95_sense = float(np.percentile(df_ts["sense_u"], 95))
    avg_slot_ms = float(df_ts["slot_ms"].mean())

    summary = {
        "seed": int(seed),
        "method": str(getattr(algo, "name", AlgoCls.__name__)),
        "avg_sum_queue": avg_sumq,
        "p95_max_queue": p95_maxq,
        "avg_sense_u": avg_sense,
        "p95_sense_u": p95_sense,
        "alpha_total_variation": float(alpha_tv),
        "switch_count": int(switch_count),
        "deadline_violation_rate": float(deadline_viol) / float(max(1, len(df_ts))),
        "avg_slot_ms": avg_slot_ms,
        "total_time_s": total_time_s,
        "device": str(device),
        "csi_error_std": float(csi_error_std),
        "q_deadline": float(q_deadline),
    }
    return df_ts, summary
