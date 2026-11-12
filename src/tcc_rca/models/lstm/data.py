"""Per-fold LSTM inputs: normalization fitted on training rows only, and window tensors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from tcc_rca.dataset.precursor import local_z
from tcc_rca.series.transforms import RobustScaler

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]

Normalization = Literal["global", "local"]


@dataclass(frozen=True)
class FoldInputs:
    """Normalized matrix (rows x kept features) with NaN where the value is unknown."""

    features: list[str]
    normalization: Normalization
    matrix: Float32Array  # NaN = missing (the loss ignores it; the model sees 0)


def normalize(
    values: pl.DataFrame,
    features: Sequence[str],
    fit_rows: BoolArray,
    normalization: Normalization,
    global_clip: float,
    local_window: int,
    local_cap: float,
) -> FoldInputs:
    """``global``: robust z with median/IQR of the fold's normal training minutes, clipped.
    ``local``: causal z against the previous ``local_window`` minutes (no fitting at all)."""
    feats = list(features)
    if normalization == "global":
        scaler = RobustScaler().fit(values, feats, fit_rows)
        z = scaler.transform(values.select(feats)).to_numpy().astype(np.float64)
        z = np.clip(z, -global_clip, global_clip)
    elif normalization == "local":
        z = local_z(values.select("day", *feats), feats, local_window, local_cap)
    else:
        raise ValueError(f"unknown normalization {normalization!r}")
    return FoldInputs(feats, normalization, z.astype(np.float32))


def window_tensor(matrix: Float32Array, end_rows: IntArray, window: int) -> Float32Array:
    """(n_windows, window, n_features) with the rows ``end - window + 1 .. end`` of each window."""
    offsets = np.arange(-(window - 1), 1, dtype=np.int64)
    out: Float32Array = matrix[end_rows[:, None] + offsets[None, :]]
    return out
