from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    """Simulation environment configuration."""

    # Topology
    n_users: int = 4
    n_user_ant: int = 2
    n_tx_ant: int = 16
    area_size_m: float = 250.0
    uav_altitude_m: float = 80.0

    # Time
    n_slots: int = 1000
    slot_s: float = 0.001

    # Power / noise
    tx_power_w: float = 1.0
    noise_comm_w: float = 1e-3
    noise_sensing_w: float = 1e-3

    # Channel model (simple Rician + distance path loss)
    rician_k_factor: float = 5.0
    pathloss_exp: float = 2.2
    ref_pathloss_db: float = -30.0
    carrier_freq_hz: float = 28e9
    ula_spacing_wavelength: float = 0.5

    # Arrivals / queues
    arrival_rate: float = 1.0
    service_scale: float = 10.0

    # Communication shock (traffic burst on user 0)
    shock_comm_start: int = 250
    shock_comm_len: int = 80
    shock_comm_gain: float = 8.0

    # Sensing shock (higher sensing noise + target maneuver)
    shock_sensing_start: int = 650
    shock_sensing_len: int = 80
    shock_sensing_noise_gain: float = 8.0


@dataclass(frozen=True)
class AlgoConfig:
    """Algorithm hyperparameters."""

    # Learning rates
    lr_w: float = 2e-2
    lr_v: float = 5e-2
    lr_alpha: float = 2e-2

    # Momentum
    momentum_w: float = 0.9
    momentum_v: float = 0.9
    momentum_alpha: float = 0.9

    # Lower-level tracking
    lower_steps_per_slot: int = 3
    lower_steps_ao: int = 60

    # Hypergradient approximation (Neumann series)
    neumann_steps: int = 5
    neumann_eta: float = 0.5

    # Gate smoothing / switching penalty
    switch_penalty: float = 0.5
    alpha_temperature: float = 6.0

    # Loss weights
    sensing_weight: float = 1.0
    queue_weight: float = 1.0
    throughput_weight: float = 2.0

    # Secondary term weight used by leader when it is not prioritized
    secondary_weight: float = 0.2

    # Small regularization for stability
    reg_v: float = 1e-3
    reg_sense: float = 1e-3

    # Fixed sensing power split (simple but stable)
    sense_power_frac: float = 0.2

    # Heuristic gate threshold
    heuristic_q_th: float = 20.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment configuration."""

    seed: int = 123
    n_runs: int = 5
    device: str = "auto"  # "auto", "cpu", "cuda"
    dtype: str = "complex64"  # "complex64" or "complex128"
