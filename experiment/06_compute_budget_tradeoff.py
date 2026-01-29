"""06_compute_budget_tradeoff.py

Compute-budget tradeoff: TOAH vs alternating optimization (AO).

We vary the AO inner-loop budget (lower_steps_ao) to emulate different compute
budgets. We compare the performance and per-slot runtime against TOAH, which is
single-loop and designed for online tracking.

Run:
    python 06_compute_budget_tradeoff.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from toah.config import AlgoConfig, EnvConfig, ExperimentConfig
from toah.algorithms import AOHeuristic, TOAH
from experiment.common import resolve_device, resolve_dtype, run_episode, scale_env_horizon


def main() -> None:
    exp_cfg = ExperimentConfig(seed=123, n_runs=3, device="auto", dtype="complex64")

    env_cfg = scale_env_horizon(EnvConfig(), n_slots=400)
    base_algo = AlgoConfig()

    device = resolve_device(exp_cfg.device)
    dtype = resolve_dtype(exp_cfg.dtype)

    out_dir = ROOT / "result" / "table"
    out_dir.mkdir(parents=True, exist_ok=True)

    ao_budgets = [2, 5, 10, 20, 40, 80]  # follower GD steps per outer round

    rows = []

    for run in range(exp_cfg.n_runs):
        run_seed = exp_cfg.seed + 1000 * run
        print(f"\n=== Run {run} (seed={run_seed}) ===")

        # TOAH once per run
        _, s_toah = run_episode(
            env_cfg=env_cfg,
            algo_cfg=base_algo,
            AlgoCls=TOAH,
            device=device,
            dtype=dtype,
            seed=run_seed,
            csi_error_std=0.0,
            q_deadline=40.0,
        )
        s_toah["run"] = int(run)
        s_toah["budget"] = "TOAH-default"
        s_toah["ao_inner_steps"] = 0
        s_toah["ao_outer_rounds"] = 0
        rows.append(s_toah)

        print(
            f"TOAH | avg(sumQ)={s_toah['avg_sum_queue']:8.2f}  avg(u)={s_toah['avg_sense_u']:8.4f}  "
            f"avg_slot_ms={s_toah['avg_slot_ms']:7.2f}"
        )

        # AO at different budgets
        for steps in ao_budgets:
            algo_cfg = replace(base_algo, lower_steps_ao=int(steps))
            _, s_ao = run_episode(
                env_cfg=env_cfg,
                algo_cfg=algo_cfg,
                AlgoCls=AOHeuristic,
                device=device,
                dtype=dtype,
                seed=run_seed,
                method_kwargs={"outer_rounds": 1},
                csi_error_std=0.0,
                q_deadline=40.0,
            )
            s_ao["run"] = int(run)
            s_ao["budget"] = f"AO-steps{steps}"
            s_ao["ao_inner_steps"] = int(steps)
            s_ao["ao_outer_rounds"] = 1
            rows.append(s_ao)

            print(
                f"AO | steps={steps:3d} | avg(sumQ)={s_ao['avg_sum_queue']:8.2f}  avg(u)={s_ao['avg_sense_u']:8.4f}  "
                f"avg_slot_ms={s_ao['avg_slot_ms']:7.2f}"
            )

    df = pd.DataFrame(rows)
    out_path = out_dir / "06_compute_budget_tradeoff.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
