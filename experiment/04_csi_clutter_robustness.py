"""04_csi_clutter_robustness.py

Reproducibility / robustness experiment.

We inject:
  - CSI estimation errors (std as a fraction of average channel magnitude)
  - sensing clutter variance (noise_sensing_w scaled by a factor)

We compare TOAH with fixed-hierarchy baselines and a heuristic gate.

Run:
    python 04_csi_clutter_robustness.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from toah.config import AlgoConfig, EnvConfig, ExperimentConfig
from toah.algorithms import FixedCommLeader, FixedSensingLeader, HeuristicGate, TOAH, TOAHNoHG
from experiment.common import resolve_device, resolve_dtype, run_episode, scale_env_horizon


def main() -> None:
    exp_cfg = ExperimentConfig(seed=123, n_runs=3, device="auto", dtype="complex64")

    base_env = scale_env_horizon(EnvConfig(), n_slots=400)
    base_algo = AlgoConfig()

    device = resolve_device(exp_cfg.device)
    dtype = resolve_dtype(exp_cfg.dtype)

    out_dir = ROOT / "result" / "table"
    out_dir.mkdir(parents=True, exist_ok=True)

    csi_list = [0.0, 0.05, 0.10, 0.20]
    clutter_scales = [1.0, 2.0, 4.0]

    methods = [
        TOAH,
        FixedSensingLeader,
        FixedCommLeader,
        TOAHNoHG,
        HeuristicGate,
    ]

    rows = []

    for run in range(exp_cfg.n_runs):
        run_seed = exp_cfg.seed + 1000 * run
        print(f"\n=== Run {run} (seed={run_seed}) ===")

        for cs in clutter_scales:
            env_cfg = replace(base_env, noise_sensing_w=float(base_env.noise_sensing_w) * float(cs))

            for csi_std in csi_list:
                for Cls in methods:
                    _, summary = run_episode(
                        env_cfg=env_cfg,
                        algo_cfg=base_algo,
                        AlgoCls=Cls,
                        device=device,
                        dtype=dtype,
                        seed=run_seed,
                        csi_error_std=float(csi_std),
                        q_deadline=40.0,
                    )
                    summary["run"] = int(run)
                    summary["clutter_scale"] = float(cs)
                    summary["csi_error_std"] = float(csi_std)
                    rows.append(summary)

                    print(
                        f"{summary['method']:<18s} | clutter={cs:>3.1f} csi={csi_std:>4.2f} | "
                        f"avg(sumQ)={summary['avg_sum_queue']:8.2f}  avg(u)={summary['avg_sense_u']:8.4f}"
                    )

    df = pd.DataFrame(rows)
    out_path = out_dir / "04_csi_clutter_robustness.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
