from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set RNG seeds for reproducible experiments.

    Notes
    -----
    When ``deterministic=True`` and CUDA is available, PyTorch may require the
    environment variable ``CUBLAS_WORKSPACE_CONFIG`` to be set for deterministic
    GEMM kernels (CUDA >= 10.2). We set a safe default if the user did not set
    it already, so that experiments can run out of the box on modern GPUs.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if torch.cuda.is_available():
            # Required by PyTorch deterministic mode for some CuBLAS ops.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            # Reduce small numeric drift across GPUs by disabling TF32.
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
            except Exception:
                pass
            try:
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older PyTorch versions may not support this flag.
            pass
