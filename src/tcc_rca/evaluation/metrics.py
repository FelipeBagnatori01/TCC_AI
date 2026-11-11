"""Evaluation metrics shared by every stage (E4-E8). Each one has a hand-computed test.

Detection is scored on labelled windows (label 1/0; ignored windows are never passed in) and on
failure events. No point-adjust is applied anywhere (it inflates detection results).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


# ---------------------------------------------------------------- threshold-free ranking metrics
def _average_ranks(values: FloatArray) -> FloatArray:
    """1-based ranks; tied values share the mean of their ranks (vectorized)."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts).astype(np.float64)
    mean_rank: FloatArray = (ends - counts + 1 + ends) / 2.0
    ranks: FloatArray = mean_rank[inverse]
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


def auc_roc(y_true: Sequence[int] | npt.NDArray[np.int_], scores: FloatArray) -> float:
    y = np.asarray(y_true).astype(bool)
    return auroc(scores[y], scores[~y])


def auc_pr(y_true: Sequence[int] | npt.NDArray[np.int_], scores: FloatArray) -> float:
    """Average precision (step-wise area under the precision-recall curve)."""
    y = np.asarray(y_true).astype(int)
    if y.sum() == 0:
        raise ValueError("AUC-PR needs at least one positive")
    return float(average_precision_score(y, scores))


# ---------------------------------------------------------------------- thresholded detection
@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def precision_recall_f1(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> PRF:
    """Precision is 0 when nothing is predicted positive (no free credit)."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)


def best_f1_threshold(y_true: npt.ArrayLike, scores: FloatArray) -> float:
    """Threshold (alarm when score >= threshold) maximizing F1; ties -> the highest threshold."""
    y = np.asarray(y_true).astype(bool)
    if not y.any():
        raise ValueError("best-F1 threshold needs at least one positive")
    best_t, best_f1 = math.inf, -1.0
    for t in np.unique(scores)[::-1]:  # descending: the first maximum is the highest threshold
        f1 = precision_recall_f1(y, scores >= t).f1
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


@dataclass(frozen=True)
class EventDetection:
    detected: bool
    delay_minutes: float | None  # first alarm - onset; negative = anticipation


def event_detection(
    onset_minute: float,
    alarm_minutes: Sequence[float] | FloatArray,
    lookback_minutes: float,
    max_delay_minutes: float,
) -> EventDetection:
    """A failure is detected by the first alarm in ``[onset - lookback, onset + max_delay]``."""
    a = np.asarray(alarm_minutes, dtype=np.float64)
    inside = a[(a >= onset_minute - lookback_minutes) & (a <= onset_minute + max_delay_minutes)]
    if inside.size == 0:
        return EventDetection(False, None)
    return EventDetection(True, float(inside.min() - onset_minute))


def false_alarms_per_hour(alarms_on_normal_windows: BoolArray) -> float:
    """Alarms raised on normal (label 0) windows per hour; one window ends per minute."""
    n = alarms_on_normal_windows.size
    if n == 0:
        raise ValueError("no normal windows")
    return float(alarms_on_normal_windows.sum()) / (n / 60.0)


def chance_event_recall(alarm_rate_per_minute: float, span_minutes: int) -> float:
    """Event recall of alarms raised at random at the given per-window rate.

    Each of the ``span_minutes`` windows in the search interval alarms independently with
    probability ``alarm_rate_per_minute``; the failure counts as detected if any of them does.
    """
    if not 0.0 <= alarm_rate_per_minute <= 1.0:
        raise ValueError("alarm rate per minute must be a probability")
    return 1.0 - (1.0 - alarm_rate_per_minute) ** span_minutes


# --------------------------------------------------------------------------------- localization
def pessimistic_rank(gold: str, scores: Mapping[str, float]) -> int:
    """1-based rank of ``gold``; candidates tied with it rank ahead (no credit from ordering).

    A gold component absent from ``scores`` gets rank ``len(scores) + 1``.
    """
    if gold not in scores:
        return len(scores) + 1
    g = scores[gold]
    if math.isnan(g):
        return len(scores)
    ahead = sum(1 for c, s in scores.items() if c != gold and not math.isnan(s) and s >= g)
    return 1 + ahead


def hits_at_k(ranks: Sequence[int] | npt.NDArray[np.int_], k: int) -> float:
    r = np.asarray(ranks)
    return float((r <= k).mean())


def mean_reciprocal_rank(ranks: Sequence[int] | npt.NDArray[np.int_]) -> float:
    return float((1.0 / np.asarray(ranks, dtype=np.float64)).mean())


# ------------------------------------------------------------------------------- paired testing
def mcnemar_exact(a_correct: npt.ArrayLike, b_correct: npt.ArrayLike) -> float:
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs)."""
    a, b = np.asarray(a_correct).astype(bool), np.asarray(b_correct).astype(bool)
    if a.shape != b.shape:
        raise ValueError("paired outcomes must have the same shape")
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return float(min(1.0, 2.0 * tail))
