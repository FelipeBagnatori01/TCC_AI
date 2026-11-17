"""Per-fold scaling and window statistics for the E4 baselines.

Every statistic that is *fitted* (mean, std, median, IQR, selection) uses only the normal minutes
of the fold's training days. Window statistics are pure functions of the window rows.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import polars as pl

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

STATS = ("mean", "std", "min", "max", "last")
A = TypeVar("A", FloatArray, npt.NDArray[np.float32])


@dataclass(frozen=True)
class ZScore:
    """Classic z-score per feature; a zero std falls back to 1.0 (as in ADR 0005)."""

    features: list[str]
    mean: FloatArray
    std: FloatArray

    @classmethod
    def fit(cls, values: pl.DataFrame, features: list[str], fit_rows: BoolArray) -> ZScore:
        if fit_rows.dtype != np.bool_ or fit_rows.shape != (values.height,):
            raise ValueError("fit_rows must be a boolean mask with one entry per row")
        if not fit_rows.any():
            raise ValueError("fit_rows selects no rows")
        block = values.select(features).to_numpy().astype(np.float64)[fit_rows]
        with np.errstate(all="ignore"):
            mean = np.nanmean(block, axis=0)
            std = np.nanstd(block, axis=0)
        if np.isnan(mean).any():
            raise ValueError("a feature has no training value; select features first")
        std = np.where(std > 0.0, std, 1.0)
        return cls(features, mean, std)

    def transform(self, values: pl.DataFrame) -> FloatArray:
        x = values.select(self.features).to_numpy().astype(np.float64)
        z: FloatArray = (x - self.mean) / self.std
        return z


def window_rows(end_rows: IntArray, window: int) -> IntArray:
    """Matrix rows of each window, shape (n_windows, window); windows end at ``end_rows``."""
    offsets = np.arange(-(window - 1), 1, dtype=np.int64)
    rows: IntArray = end_rows[:, None] + offsets[None, :]
    return rows


def last_steps(per_step: A, n: int | None) -> A:
    """The last ``n`` steps (axis 1) of ``(windows, steps)`` scores; ``None`` keeps them all.

    ADR 0019: a positive window means "an onset in the last D minutes", so the unsupervised
    window scores read only the last D + 1 minutes; ``None`` is the whole-window control.
    """
    if n is None:
        return per_step
    if n > per_step.shape[1]:
        raise ValueError(f"cannot score the last {n} steps of {per_step.shape[1]}")
    out: A = per_step[:, -n:]
    return out


def window_max_abs(
    z: FloatArray, end_rows: IntArray, window: int, score_minutes: int | None = None
) -> FloatArray:
    """max over the window's last ``score_minutes`` minutes (all when ``None``) and over
    features of |z| (missing counts as 0)."""
    per_minute = np.nan_to_num(np.abs(z), nan=0.0).max(axis=1)
    out: FloatArray = last_steps(per_minute[window_rows(end_rows, window)], score_minutes).max(
        axis=1
    )
    return out


def window_statistics(
    x: FloatArray, end_rows: IntArray, window: int, chunk: int = 256
) -> npt.NDArray[np.float32]:
    """Per feature: mean, std, min, max, last over the window -> (n_windows, 5 * n_features).

    Columns are grouped by statistic (all means, then all stds, ...). A feature missing on the
    whole window yields NaN (LightGBM accepts it; Isolation Forest callers fill it). Windows are
    processed in chunks to bound memory.
    """
    out = np.empty((end_rows.size, len(STATS) * x.shape[1]), dtype=np.float32)
    for start in range(0, end_rows.size, chunk):
        sl = slice(start, start + chunk)
        blocks = x[window_rows(end_rows[sl], window)]  # (n, window, n_features)
        with warnings.catch_warnings():  # all-NaN slices are expected (missing features)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            parts = [
                np.nanmean(blocks, axis=1),
                np.nanstd(blocks, axis=1),
                np.nanmin(blocks, axis=1),
                np.nanmax(blocks, axis=1),
                blocks[:, -1, :],
            ]
        out[sl] = np.concatenate(parts, axis=1)
    return out
