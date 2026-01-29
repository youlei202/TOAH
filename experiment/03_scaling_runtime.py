"""03_scaling_runtime.py

Mobile feasibility: runtime and memory scaling with (M, K).

We vary:
  - M: number of UAV/RSU transmit antennas
  - K: number of users

We run a short horizon without shocks to estimate the per-slot wall-clock time
and the peak GPU memory.

Run:
    python 03_scaling_runtime.py
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
    exp_cfg = ExperimentConfig(seed=123, n_runs=3, device="auto", dtype="complex64")

    base_env = EnvConfig()
    base_algo = AlgoConfig()

    device = resolve_device(exp_cfg.device)
    dtype = resolve_dtype(exp_cfg.dtype)

    out_dir = ROOT / "result" / "table"
    out_dir.mkdir(parents=True, exist_ok=True)

    m_list = [8, 16, 32, 64]
    k_list = [2, 4, 8, 12]

    rows = []

    for run in range(exp_cfg.n_runs):
        run_seed = exp_cfg.seed + 1000 * run
        print(f"\n=== Run {run} (seed={run_seed}) ===")

        for m in m_list:
            for k in k_list:
                env_cfg = replace(
                    base_env,
                    scenario="uav",
                    n_tx_ant=int(m),
                    n_users=int(k),
                    n_slots=200,
                    shock_comm_len=0,
                    shock_sensing_len=0,
                )

                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                _, summary = run_episode(
                    env_cfg=env_cfg,
                    algo_cfg=base_algo,
                    AlgoCls=TOAH,
                    device=device,
                    dtype=dtype,
                    seed=run_seed,
                    csi_error_std=0.0,
                    q_deadline=40.0,
                )

                peak_mem_mb = None
                if device.type == "cuda":
                    peak_mem_mb = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))

                summary["run"] = int(run)
                summary["M"] = int(m)
                summary["K"] = int(k)
                summary["n_slots"] = int(env_cfg.n_slots)
                summary["peak_mem_mb"] = float(peak_mem_mb) if peak_mem_mb is not None else float("nan")
                rows.append(summary)

                print(
                    f"TOAH | M={m:2d} K={k:2d} | avg_slot_ms={summary['avg_slot_ms']:7.2f}  "
                    f"peak_mem_mb={summary['peak_mem_mb']:8.1f}"
                )

    df = pd.DataFrame(rows)
    out_path = out_dir / "03_scaling_runtime.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
