"""Component localizers on raw KPI values around an estimated onset (ADR 0020).

* ``robust_feature_scores`` (C4): BARO's robust scorer (Pham et al., FSE 2024), re-implemented:
  per KPI, |x - median| / IQR of a reference before the onset, maximum after the onset; a zero
  IQR becomes 1, as scikit-learn's ``RobustScaler`` (used by BARO) does;
* ``nsigma_feature_scores`` (C5): the same with mean and standard deviation (N-Sigma);
* ``seasonal_feature_scores`` (C6): |x - same minute of the previous usable day| / training IQR;
* ``personalized_pagerank`` (C7): random walk with restart on the dependency graph;
* ``reciprocal_rank_fusion`` (C8).

Per-feature scores become component scores by the maximum over the component's KPIs (BARO
ranks metrics; the first metric of a component gives its position). Missing values are ignored;
a component with no finite score gets 0.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@contextmanager
def _quiet() -> Iterator[None]:
    """Silence 'All-NaN slice' warnings: an all-missing KPI simply scores 0."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _nan_stat(fn: str, block: FloatArray) -> FloatArray:
    with _quiet():
        if fn == "median":
            out: FloatArray = np.nanmedian(block, axis=0)
        elif fn == "iqr":
            q75, q25 = np.nanpercentile(block, [75, 25], axis=0)
            out = q75 - q25
        elif fn == "mean":
            out = np.nanmean(block, axis=0)
        else:
            out = np.nanstd(block, axis=0)
    return out


def _safe_scale(scale: FloatArray) -> FloatArray:
    out: FloatArray = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return out


def _post_max(dev: FloatArray) -> FloatArray:
    with _quiet():
        out: FloatArray = np.nanmax(np.abs(dev), axis=0)
    return np.where(np.isfinite(out), out, 0.0)


def robust_feature_scores(
    reference: FloatArray, post: FloatArray, fallback_median: FloatArray, fallback_iqr: FloatArray
) -> FloatArray:
    """C4 per-feature score. ``reference`` (rows before the onset, possibly empty: then the
    fallback statistics are used) and ``post`` (onset .. window end) are raw (rows, F) blocks."""
    if reference.shape[0]:
        med, iqr = _nan_stat("median", reference), _nan_stat("iqr", reference)
        med = np.where(np.isfinite(med), med, fallback_median)
        iqr = np.where(np.isfinite(iqr), iqr, fallback_iqr)
    else:
        med, iqr = fallback_median, fallback_iqr
    return _post_max((post - med) / _safe_scale(iqr))


def nsigma_feature_scores(
    reference: FloatArray, post: FloatArray, fallback_mean: FloatArray, fallback_std: FloatArray
) -> FloatArray:
    """C5 per-feature score: |x - mean| / std of the reference, maximum after the onset."""
    if reference.shape[0]:
        mean, std = _nan_stat("mean", reference), _nan_stat("std", reference)
        mean = np.where(np.isfinite(mean), mean, fallback_mean)
        std = np.where(np.isfinite(std), std, fallback_std)
    else:
        mean, std = fallback_mean, fallback_std
    return _post_max((post - mean) / _safe_scale(std))


def seasonal_feature_scores(
    post: FloatArray, previous: FloatArray, scale: FloatArray
) -> FloatArray:
    """C6 per-feature score: |x - x(previous day, same minute)| / training IQR, maximum after
    the onset. ``previous`` has the same shape as ``post`` (NaN where that minute is absent)."""
    return _post_max((post - previous) / _safe_scale(scale))


def training_statistics(block: FloatArray) -> dict[str, FloatArray]:
    """Median, IQR, mean and std per feature over the fold's normal training minutes (the
    fallback reference); non-finite statistics become 0 (median/mean) or 1 (scales)."""
    out = {k: _nan_stat(k, block) for k in ("median", "iqr", "mean", "std")}
    for k in ("median", "mean"):
        out[k] = np.where(np.isfinite(out[k]), out[k], 0.0)
    for k in ("iqr", "std"):
        out[k] = _safe_scale(out[k])
    return out


def component_max(
    per_feature: FloatArray, columns: Mapping[str, IntArray], candidates: Sequence[str]
) -> dict[str, float]:
    """Component score = maximum of its features' scores (0 without features)."""
    return {c: float(per_feature[columns[c]].max()) if c in columns else 0.0 for c in candidates}


def personalized_pagerank(
    edges: Sequence[tuple[str, str]],
    personalization: Mapping[str, float],
    damping: float,
    backward_weight: float,
    iterations: int = 200,
    tol: float = 1e-12,
) -> dict[str, float]:
    """Random walk with restart: from a node, move along its outgoing edges (weight 1, from the
    affected component to the one it depends on) or backwards (``backward_weight``); restart to
    the personalization distribution with probability ``1 - damping``. Dangling nodes restart.

    Nodes = every endpoint plus every personalized node. A zero personalization becomes uniform.
    """
    nodes = sorted({n for e in edges for n in e} | set(personalization))
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    w = np.zeros((n, n), dtype=np.float64)
    for src, dst in edges:
        if src == dst:
            continue
        w[index[src], index[dst]] += 1.0
        w[index[dst], index[src]] += backward_weight
    p = np.array([max(personalization.get(x, 0.0), 0.0) for x in nodes], dtype=np.float64)
    p = p / p.sum() if p.sum() > 0 else np.full(n, 1.0 / n)
    out_sum = w.sum(axis=1)
    trans = np.divide(w, out_sum[:, None], out=np.zeros_like(w), where=out_sum[:, None] > 0)
    dangling = out_sum == 0
    r = p.copy()
    for _ in range(iterations):
        new = damping * (r @ trans + r[dangling].sum() * p) + (1.0 - damping) * p
        if np.abs(new - r).sum() < tol:
            r = new
            break
        r = new
    return {x: float(r[index[x]]) for x in nodes}


def ranking(scores: Mapping[str, float]) -> list[str]:
    """Components by decreasing score, ties alphabetical (deterministic)."""
    return sorted(scores, key=lambda c: (-scores[c], c))


def reciprocal_rank_fusion(
    score_sets: Sequence[Mapping[str, float]], candidates: Sequence[str], k: int
) -> dict[str, float]:
    """Sum over rankings of 1 / (k + position), positions from 1 (Cormack et al., 2009)."""
    out = dict.fromkeys(candidates, 0.0)
    for scores in score_sets:
        for pos, c in enumerate(ranking({c: scores.get(c, 0.0) for c in candidates}), 1):
            out[c] += 1.0 / (k + pos)
    return out
