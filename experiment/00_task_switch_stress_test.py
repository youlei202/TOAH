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


import pandas as pd
import torch

from toah.config import AlgoConfig, EnvConfig, ExperimentConfig
from toah.algorithms import (
    AOHeuristic,
    FixedCommLeader,
    FixedSensingLeader,
    HeuristicGate,
    TOAH,
    TOAHNoHG,
)

from experiment.common import resolve_device, resolve_dtype, run_episode


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

    rows_ts: list[dict] = []
    rows_sum: list[dict] = []

    for run in range(exp_cfg.n_runs):
        run_seed = exp_cfg.seed + 1000 * run
        print(f"\n=== Run {run} (seed={run_seed}) ===")

        for _, (Cls, kwargs) in enumerate(methods):
            # IMPORTANT: run_episode keeps all tensors on-device during the slot loop.
            # This avoids per-slot CPU↔GPU synchronization that can slow the experiment
            # by orders of magnitude.
            df_ts, summary = run_episode(
                env_cfg=env_cfg,
                algo_cfg=algo_cfg,
                AlgoCls=Cls,
                device=device,
                dtype=dtype,
                seed=run_seed,
                method_kwargs=kwargs,
            )

            # Enrich with run metadata and append
            df_ts.insert(0, "method", summary["method"])
            df_ts.insert(0, "seed", run_seed)
            df_ts.insert(0, "run", run)
            rows_ts.append(df_ts)

            summary_row = {"run": run, **summary}
            rows_sum.append(summary_row)

            print(
                f"{summary['method']:>18s} | avg(sumQ)={summary['avg_sum_queue']:8.2f}  "
                f"p95(maxQ)={summary['p95_max_queue']:8.2f}  avg(u)={summary['avg_sense_u']:8.4f}  "
                f"p95(u)={summary['p95_sense_u']:8.4f}  tv(alpha)={summary['alpha_total_variation']:6.2f}  "
                f"avg_slot_ms={summary['avg_slot_ms']:7.2f}"
            )

    df_ts = pd.concat(rows_ts, ignore_index=True) if len(rows_ts) > 0 else pd.DataFrame()
    df_sum = pd.DataFrame(rows_sum)

    ts_path = out_dir / "00_task_switch_timeseries.csv"
    sum_path = out_dir / "00_task_switch_summary.csv"
    df_ts.to_csv(ts_path, index=False)
    df_sum.to_csv(sum_path, index=False)

    print(f"\nSaved time series to: {ts_path}")
    print(f"Saved summary to:    {sum_path}")


if __name__ == "__main__":
    main()
