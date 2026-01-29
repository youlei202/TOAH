"""00_task_switch_stress_test.py

Flagship experiment:
  - mobility-driven time-varying channels
  - a traffic burst (comm shock)
  - a sensing shock (noise increase + target maneuver)

We compare TOAH and baselines under strict reproducibility.
Run:
    python 00_task_switch_stress_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow `python experiment/00_xxx.py` without installing as a package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd
import torch

from toah.config import AlgoConfig, EnvConfig, ExperimentConfig
from toah.env import ISACEnv
from toah.algorithms import (
    AOHeuristic,
    FixedCommLeader,
    FixedSensingLeader,
    HeuristicGate,
    TOAH,
    TOAHNoHG,
)
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


def main() -> None:
    # -------------------------
    # Defaults (no CLI args)
    # -------------------------
    exp_cfg = ExperimentConfig(seed=123, n_runs=5, device="auto", dtype="complex64")
    env_cfg = EnvConfig()
    algo_cfg = AlgoConfig()

    device = resolve_device(exp_cfg.device)
    dtype = resolve_dtype(exp_cfg.dtype)

    out_dir = Path(__file__).resolve().parents[1] / "result" / "table"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        (TOAH, {}),
        (FixedSensingLeader, {}),
        (FixedCommLeader, {}),
        (TOAHNoHG, {}),
        (HeuristicGate, {}),
        (AOHeuristic, {}),
    ]

    rows_ts = []
    rows_sum = []

    for run in range(exp_cfg.n_runs):
        run_seed = exp_cfg.seed + 1000 * run
        print(f"\n=== Run {run} (seed={run_seed}) ===")

        for method_idx, (Cls, kwargs) in enumerate(methods):
            # Reset RNG for fair comparisons across methods in the same run.
            set_seed(run_seed, deterministic=True)

            env = ISACEnv(env_cfg, device=device, dtype=dtype)
            state = env.reset()

            algo = Cls(env_cfg, algo_cfg, device=device, dtype=dtype, **kwargs)
            algo.reset(state)

            alpha_prev = None
            alpha_tv = 0.0
            t_start = time.perf_counter()

            done = False
            while not done:
                t0 = time.perf_counter()
                act = algo.step(state)
                state, info, done = env.step(act["ws"], act["vs"])
                t1 = time.perf_counter()

                alpha = float(act["alpha"])
                if alpha_prev is not None:
                    alpha_tv += abs(alpha - alpha_prev)
                alpha_prev = alpha

                sum_q = float(torch.sum(torch.tensor(info["queues"])).item())
                max_q = float(torch.max(torch.tensor(info["queues"])).item())
                sum_rate = float(torch.sum(torch.tensor(info["rate"])).item())

                rows_ts.append(
                    {
                        "run": run,
                        "seed": run_seed,
                        "method": algo.name,
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

            t_end = time.perf_counter()
            total_s = t_end - t_start

            # Summaries
            df_m = pd.DataFrame([r for r in rows_ts if r["run"] == run and r["method"] == algo.name])
            avg_sumq = float(df_m["sum_queue"].mean())
            p95_maxq = float(np.percentile(df_m["max_queue"], 95))
            avg_sense = float(df_m["sense_u"].mean())
            p95_sense = float(np.percentile(df_m["sense_u"], 95))
            avg_slot_ms = float(df_m["slot_ms"].mean())

            rows_sum.append(
                {
                    "run": run,
                    "seed": run_seed,
                    "method": algo.name,
                    "avg_sum_queue": avg_sumq,
                    "p95_max_queue": p95_maxq,
                    "avg_sense_u": avg_sense,
                    "p95_sense_u": p95_sense,
                    "alpha_total_variation": float(alpha_tv),
                    "total_time_s": float(total_s),
                    "avg_slot_ms": avg_slot_ms,
                    "device": str(device),
                }
            )

            print(
                f"{algo.name:>18s} | avg(sumQ)={avg_sumq:8.2f}  "
                f"p95(maxQ)={p95_maxq:8.2f}  avg(u)={avg_sense:8.4f}  "
                f"p95(u)={p95_sense:8.4f}  tv(alpha)={alpha_tv:6.2f}  "
                f"avg_slot_ms={avg_slot_ms:7.2f}"
            )

    df_ts = pd.DataFrame(rows_ts)
    df_sum = pd.DataFrame(rows_sum)

    ts_path = out_dir / "00_task_switch_timeseries.csv"
    sum_path = out_dir / "00_task_switch_summary.csv"
    df_ts.to_csv(ts_path, index=False)
    df_sum.to_csv(sum_path, index=False)

    print(f"\nSaved time series to: {ts_path}")
    print(f"Saved summary to:    {sum_path}")


if __name__ == "__main__":
    main()
