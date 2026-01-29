"""02_truncation_depth_ablation.py

Ablation: truncation depths for online bilevel optimization.

We vary:
  - Neumann truncation depth (neumann_steps) for implicit hypergradients
  - follower tracking steps per slot (lower_steps_per_slot)

We report performance and runtime. This experiment is designed for tables and
heatmaps.

Run:
    python 02_truncation_depth_ablation.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from toah.config import AlgoConfig, EnvConfig, ExperimentConfig
from toah.algorithms import TOAH
from experiment.common import resolve_device, resolve_dtype, run_episode, scale_env_horizon


def main() -> None:
    exp_cfg = ExperimentConfig(seed=123, n_runs=3, device="auto", dtype="complex64")

    # Shorter horizon for a grid sweep. We keep shock locations aligned.
    env_cfg = scale_env_horizon(EnvConfig(), n_slots=600)

    base_algo = AlgoConfig()

    device = resolve_device(exp_cfg.device)
    dtype = resolve_dtype(exp_cfg.dtype)

    out_dir = ROOT / "result" / "table"
    out_dir.mkdir(parents=True, exist_ok=True)

    neumann_steps_list = [1, 2, 3, 5, 8, 10]
    lower_steps_list = [1, 2, 3, 5, 8]

    rows = []

    for run in range(exp_cfg.n_runs):
        run_seed = exp_cfg.seed + 1000 * run
        print(f"\n=== Run {run} (seed={run_seed}) ===")

        for ns in neumann_steps_list:
            for ls in lower_steps_list:
                algo_cfg = replace(
                    base_algo,
                    neumann_steps=int(ns),
                    lower_steps_per_slot=int(ls),
                )
                _, summary = run_episode(
                    env_cfg=env_cfg,
                    algo_cfg=algo_cfg,
                    AlgoCls=TOAH,
                    device=device,
                    dtype=dtype,
                    seed=run_seed,
                    csi_error_std=0.0,
                    q_deadline=40.0,
                )
                summary["run"] = int(run)
                summary["neumann_steps"] = int(ns)
                summary["lower_steps_per_slot"] = int(ls)
                rows.append(summary)

                print(
                    f"TOAH | ns={ns:2d} ls={ls:2d} | avg(sumQ)={summary['avg_sum_queue']:8.2f}  "
                    f"avg(u)={summary['avg_sense_u']:8.4f}  avg_slot_ms={summary['avg_slot_ms']:7.2f}"
                )

    df = pd.DataFrame(rows)
    out_path = out_dir / "02_truncation_depth_ablation.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
