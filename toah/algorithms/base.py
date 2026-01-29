from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..config import AlgoConfig, EnvConfig
from ..env.isac_env import EnvState


class OnlineAlgorithm(ABC):
    """Base interface for online ISAC algorithms."""

    def __init__(self, env_cfg: EnvConfig, algo_cfg: AlgoConfig, device: torch.device, dtype: torch.dtype) -> None:
        self.env_cfg = env_cfg
        self.algo_cfg = algo_cfg
        self.device = device
        self.dtype = dtype

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def reset(self, state: EnvState) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self, state: EnvState) -> dict:
        """Run one online step.

        Returns:
            A dict with keys: ws, vs, alpha.
        """
        raise NotImplementedError
