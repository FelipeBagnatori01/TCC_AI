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

FloatArray = npt.NDArray[np.float64]


def _average_ranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < values.size:  # ties share the mean of their 1-based ranks
        j = i
        while j + 1 < values.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(
    positives: Sequence[float] | FloatArray, negatives: Sequence[float] | FloatArray
) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie), via the Mann-Whitney U statistic."""
    pos, neg = np.asarray(positives, dtype=np.float64), np.asarray(negatives, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("AUROC needs at least one positive and one negative")
    if np.isnan(pos).any() or np.isnan(neg).any():
        raise ValueError("AUROC scores must not contain NaN")
    ranks = _average_ranks(np.concatenate([pos, neg]))
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


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


def minute_scores(z: FloatArray, top_k: int) -> FloatArray:
    """Mean of the ``top_k`` largest |z| per row; missing values count as 0 (no evidence)."""
    a = np.nan_to_num(np.abs(z), nan=0.0)
    k = min(top_k, a.shape[1])
    top: FloatArray = np.partition(a, a.shape[1] - k, axis=1)[:, -k:]
    means: FloatArray = top.mean(axis=1)
    return means


@dataclass(frozen=True)
class LeadResult:
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
