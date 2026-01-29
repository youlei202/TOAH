"""05_rsu_intersection_stress_test.py

Second scenario: RSU-enabled ISAC near an intersection.

We reuse the task-switch stress-test idea, but we replace UAV mobility with an RSU
at a road intersection and moving vehicles. This scenario produces stronger
channel drift and emphasizes the mobile computing requirement.

Run:
    python 05_rsu_intersection_stress_test.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
    exp_cfg = ExperimentConfig(seed=123, n_runs=5, device="auto", dtype="complex64")

    # RSU scenario: lower height, smaller area, strong mobility.
    env_cfg = replace(
        EnvConfig(),
        scenario="rsu_intersection",
        area_size_m=200.0,
        uav_altitude_m=6.0,
        n_slots=1000,
    )

    algo_cfg = AlgoConfig()

    device = resolve_device(exp_cfg.device)
    dtype = resolve_dtype(exp_cfg.dtype)

    out_dir = ROOT / "result" / "table"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        TOAH,
        FixedSensingLeader,
        FixedCommLeader,
        TOAHNoHG,
        HeuristicGate,
        AOHeuristic,
    ]

    rows_ts = []
    rows_sum = []

    for run in range(exp_cfg.n_runs):
        run_seed = exp_cfg.seed + 1000 * run
        print(f"\n=== Run {run} (seed={run_seed}) ===")

        for Cls in methods:
            df_ts, summary = run_episode(
                env_cfg=env_cfg,
                algo_cfg=algo_cfg,
                AlgoCls=Cls,
                device=device,
                dtype=dtype,
                seed=run_seed,
                csi_error_std=0.0,
                q_deadline=40.0,
            )

            df_ts.insert(0, "run", run)
            df_ts.insert(1, "seed", run_seed)
            df_ts.insert(2, "method", summary["method"])
            rows_ts.append(df_ts)

            summary["run"] = int(run)
            summary["scenario"] = "rsu_intersection"
            rows_sum.append(summary)

            print(
                f"{summary['method']:<18s} | avg(sumQ)={summary['avg_sum_queue']:8.2f}  "
                f"p95(maxQ)={summary['p95_max_queue']:8.2f}  avg(u)={summary['avg_sense_u']:8.4f}  "
                f"tv(alpha)={summary['alpha_total_variation']:6.2f}  avg_slot_ms={summary['avg_slot_ms']:7.2f}"
            )

    df_ts_all = pd.concat(rows_ts, ignore_index=True)
    df_sum_all = pd.DataFrame(rows_sum)

    ts_path = out_dir / "05_rsu_intersection_timeseries.csv"
    sum_path = out_dir / "05_rsu_intersection_summary.csv"
    df_ts_all.to_csv(ts_path, index=False)
    df_sum_all.to_csv(sum_path, index=False)

    print(f"\nSaved time series to: {ts_path}")
    print(f"Saved summary to:    {sum_path}")


if __name__ == "__main__":
    main()
