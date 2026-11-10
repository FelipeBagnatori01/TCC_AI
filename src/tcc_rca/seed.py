"""Global seeding (Python, NumPy, PyTorch) for reproducible runs (R5)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed every RNG in use and force deterministic PyTorch kernels."""
    random.seed(seed)
    np.random.seed(seed)  # legacy global RNG: seeded for third-party code
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
