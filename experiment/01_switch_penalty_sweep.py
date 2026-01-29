"""01_switch_penalty_sweep.py

Ablation: switching penalty sweep (lambda_sw).

We run TOAH with different switching penalties and measure:
  - communication backlog metrics
  - sensing uncertainty metrics
  - gate overhead (total variation and switch counts)
  - per-slot runtime

Run:
    python 01_switch_penalty_sweep.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from toah.config import AlgoConfig, EnvConfig, ExperimentConfig
from toah.algorithms import TOAH
from experiment.common import resolve_device, resolve_dtype, run_episode


def main() -> None:
    exp_cfg = ExperimentConfig(seed=123, n_runs=5, device="auto", dtype="complex64")
    env_cfg = EnvConfig()
    base_algo = AlgoConfig()

    device = resolve_device(exp_cfg.device)
    dtype = resolve_dtype(exp_cfg.dtype)

    out_dir = ROOT / "result" / "table"
    out_dir.mkdir(parents=True, exist_ok=True)

    switch_penalties = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

    rows_ts = []
    rows_sum = []

    for run in range(exp_cfg.n_runs):
        run_seed = exp_cfg.seed + 1000 * run
        print(f"\n=== Run {run} (seed={run_seed}) ===")

        for lam in switch_penalties:
            algo_cfg = replace(base_algo, switch_penalty=float(lam))
            df_ts, summary = run_episode(
                env_cfg=env_cfg,
                algo_cfg=algo_cfg,
                AlgoCls=TOAH,
                device=device,
                dtype=dtype,
                seed=run_seed,
                csi_error_std=0.0,
                q_deadline=40.0,
            )

            df_ts.insert(0, "run", run)
            df_ts.insert(1, "seed", run_seed)
            df_ts.insert(2, "switch_penalty", float(lam))
            rows_ts.append(df_ts)

            summary["run"] = int(run)
            summary["switch_penalty"] = float(lam)
            rows_sum.append(summary)

            print(
                f"TOAH | lam={lam:>4.2f} | avg(sumQ)={summary['avg_sum_queue']:8.2f}  "
                f"p95(maxQ)={summary['p95_max_queue']:8.2f}  avg(u)={summary['avg_sense_u']:8.4f}  "
                f"tv(alpha)={summary['alpha_total_variation']:6.2f}  switches={summary['switch_count']:3d}  "
                f"avg_slot_ms={summary['avg_slot_ms']:7.2f}"
            )

    df_ts_all = pd.concat(rows_ts, ignore_index=True)
    df_sum_all = pd.DataFrame(rows_sum)

    ts_path = out_dir / "01_switch_penalty_timeseries.csv"
    sum_path = out_dir / "01_switch_penalty_summary.csv"
    df_ts_all.to_csv(ts_path, index=False)
    df_sum_all.to_csv(sum_path, index=False)

    print(f"\nSaved time series to: {ts_path}")
    print(f"Saved summary to:    {sum_path}")


if __name__ == "__main__":
    main()
