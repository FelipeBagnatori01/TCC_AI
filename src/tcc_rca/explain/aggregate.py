"""Aggregation of attributions and explanation-quality metrics (E6, ADR 0016).

Attributions are signed contributions to the explained output (anomaly score or failure
probability). A feature's importance is the sum over time of its contributions; a component's is
the sum of the POSITIVE importances of its features (what pushed the output up), or the mean of
its three largest (less sensitive to how many KPIs a component has, as in E4/E5).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import kendalltau

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


def feature_importance(attr: Float32Array | FloatArray) -> FloatArray:
    """(W, F) time x feature attributions -> (F,) summed over time; (F,) is returned as is."""
    a = np.asarray(attr, dtype=np.float64)
    out: FloatArray = a.sum(axis=0) if a.ndim == 2 else a
    return out


def component_scores(
    importance: FloatArray,
    columns: Mapping[str, IntArray],
    candidates: Sequence[str],
) -> dict[str, dict[str, float]]:
    """``sum``: sum of positive feature importances; ``top3``: mean of the 3 largest (clipped
    at 0). Candidates without features in this fold score 0."""
    pos = np.clip(importance, 0.0, None)
    out: dict[str, dict[str, float]] = {"sum": {}, "top3": {}}
    for c in candidates:
        cols = columns.get(c)
        vals = pos[cols] if cols is not None else np.zeros(1)
        out["sum"][c] = float(vals.sum())
        out["top3"][c] = float(np.sort(vals)[::-1][:3].mean())
    return out


def category_scores(importance: FloatArray, categories: Sequence[str]) -> dict[str, float]:
    """Sum of positive importances per KPI category (the suggested reason category is the max)."""
    pos = np.clip(importance, 0.0, None)
    out: dict[str, float] = {}
    for cat in sorted(set(categories)):
        mask = np.array([c == cat for c in categories])
        out[cat] = float(pos[mask].sum())
    return out


def top_category(scores: Mapping[str, float]) -> str | None:
    best = max(scores.values(), default=0.0)
    if best <= 0.0:
        return None
    return min(c for c, v in scores.items() if v == best)


# --------------------------------------------------------------------------------- fidelity
def deletion_insertion(
    predict: Callable[[Float32Array], FloatArray],
    window: Float32Array,
    reference: Float32Array,
    order: IntArray,
    steps: Sequence[int],
) -> tuple[FloatArray, FloatArray]:
    """Output after deleting (replacing by reference) or inserting (into the reference) the first
    ``k`` features of ``order``, for each ``k`` in ``steps`` (0 included first)."""
    ks = [0, *steps]
    deleted, inserted = [], []
    for k in ks:
        idx = order[:k]
        d = window.copy()
        d[:, idx] = reference[:, idx]
        i = reference.copy()
        i[:, idx] = window[:, idx]
        deleted.append(d)
        inserted.append(i)
    return predict(np.stack(deleted)), predict(np.stack(inserted))


def normalized_curve_area(values: FloatArray) -> float:
    """Mean of a curve normalized by its first point (deletion: lower = more faithful)."""
    start = values[0]
    if start == 0:
        return float("nan")
    return float(np.mean(values / start))


# ----------------------------------------------------------------------- agreement / stability
def top_k(importance: FloatArray, k: int) -> IntArray:
    order: IntArray = np.argsort(-importance, kind="mergesort")[:k].astype(np.int64)
    return order


def jaccard_top_k(a: FloatArray, b: FloatArray, k: int) -> float:
    sa, sb = set(top_k(a, k).tolist()), set(top_k(b, k).tolist())
    return len(sa & sb) / len(sa | sb)


def kendall_top_k(a: FloatArray, b: FloatArray, k: int) -> float:
    """Kendall tau of the two importance vectors restricted to the union of their top-k."""
    union = np.array(sorted(set(top_k(a, k).tolist()) | set(top_k(b, k).tolist())))
    if union.size < 2:
        return float("nan")
    tau = kendalltau(a[union], b[union]).statistic
    return float(tau)
