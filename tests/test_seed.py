from __future__ import annotations

import random

import numpy as np
import torch

from tcc_rca.seed import set_global_seed


def _draw() -> tuple[float, float, float]:
    return (random.random(), float(np.random.rand()), float(torch.rand(1).item()))


def test_same_seed_same_draws() -> None:
    set_global_seed(7)
    first = _draw()
    set_global_seed(7)
    assert _draw() == first


def test_different_seed_different_draws() -> None:
    set_global_seed(7)
    first = _draw()
    set_global_seed(8)
    assert _draw() != first
