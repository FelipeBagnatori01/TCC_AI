"""Is there a precursor signal? A learning-free check (ADR 0007).

Each minute gets an anomaly score: the mean of its ``top_k`` largest |robust z| values, with the
selector and scaler fitted on the normal minutes of the initial block only. For each lead ``L``
the scores ``L`` minutes before every evaluated onset (L < 0: after the onset) are compared with
the scores of normal minutes of the same days by AUROC. A precursor exists for lead ``L > 0``
only if the AUROC confidence interval lies above 0.5.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from tcc_rca.evaluation.metrics import auroc

FloatArray = npt.NDArray[np.float64]


def bootstrap_auroc_ci(
    positives: FloatArray, negatives: FloatArray, n: int, seed: int, level: float = 0.95
) -> tuple[float, float]:
    """Percentile CI, resampling positives and negatives independently."""
    rng = np.random.default_rng(seed)
    stats = np.empty(n)
    for b in range(n):
        stats[b] = auroc(
            rng.choice(positives, positives.size, replace=True),
            rng.choice(negatives, negatives.size, replace=True),
        )
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))


def local_abs_z(
    values: pl.DataFrame, features: Sequence[str], window: int, cap: float
) -> FloatArray:
    """|z| of each value against the PREVIOUS ``window`` minutes of the same day (causal).

    Center: rolling median of the past; scale: rolling std of the past, floored at 0.1% of the
    median's magnitude (+1) so near-constant series do not blow up; |z| capped at ``cap``.
    Robust to slow drift across days, which breaks a scaler fitted on the initial block (Bank).
    Uses no label and no future value.
    """

    def z(f: str) -> pl.Expr:
        past = pl.col(f).shift(1)
        center = past.rolling_median(window, min_samples=10).over("day")
        spread = past.rolling_std(window, min_samples=10).over("day")
        scale = pl.max_horizontal(spread, (center.abs() + 1.0) * 1e-3)
        return ((pl.col(f) - center) / scale).alias(f)

    out = values.select(*[z(f) for f in features]).to_numpy().astype(np.float64)
    out = np.where(np.isfinite(out), np.abs(out), np.nan)
    clipped: FloatArray = np.minimum(out, cap)
    return clipped


def minute_scores(z: FloatArray, top_k: int) -> FloatArray:
    """Mean of the ``top_k`` largest |z| per row; missing values count as 0 (no evidence)."""
    a = np.nan_to_num(np.abs(z), nan=0.0)
    k = min(top_k, a.shape[1])
    top: FloatArray = np.partition(a, a.shape[1] - k, axis=1)[:, -k:]
    means: FloatArray = top.mean(axis=1)
    return means


@dataclass(frozen=True)
class LeadResult:
    method: str  # "global" (initial-block scaler) or "local" (causal rolling baseline)
    lead_minutes: int
    group: str  # "all" or a failure reason
    n_pos: int
    n_neg: int
    auroc: float
    ci_low: float
    ci_high: float


def lead_samples(
    scores: FloatArray,
    minute_of_row: npt.NDArray[np.datetime64],
    onsets: npt.NDArray[np.datetime64],
    lead_minutes: int,
) -> FloatArray:
    """Scores at ``onset - lead`` for every onset whose target minute exists in the matrix."""
    lookup = {t: i for i, t in enumerate(minute_of_row.tolist())}
    targets = (onsets - np.timedelta64(lead_minutes, "m")).tolist()
    rows = [lookup[t] for t in targets if t in lookup]
    return scores[np.array(rows, dtype=np.int64)] if rows else np.empty(0)
