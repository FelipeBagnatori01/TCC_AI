"""Cluster bootstrap confidence intervals (no number is reported without dispersion).

Units that are not independent (overlapping windows of the same half hour) are resampled
together as a cluster; independent units (failures, queries) are their own cluster.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Estimate:
    value: float
    ci_low: float
    ci_high: float
    n_valid_resamples: int

    def as_dict(self, digits: int = 4) -> dict[str, float | int]:
        return {
            "value": round(self.value, digits),
            "ci_low": round(self.ci_low, digits),
            "ci_high": round(self.ci_high, digits),
            "n_valid_resamples": self.n_valid_resamples,
        }


def clusters_from_labels(labels: Sequence[object] | npt.NDArray[np.generic]) -> list[IntArray]:
    """Row indices grouped by cluster label, in order of first appearance."""
    groups: dict[object, list[int]] = {}
    for i, lab in enumerate(np.asarray(labels).tolist()):
        groups.setdefault(lab, []).append(i)
    return [np.array(v, dtype=np.int64) for v in groups.values()]


def cluster_bootstrap(
    statistic: Callable[[IntArray], float],
    clusters: Sequence[IntArray],
    n_resamples: int,
    seed: int,
    level: float = 0.95,
) -> Estimate:
    """Percentile CI of ``statistic(row_indices)``, resampling whole clusters with replacement.

    Resamples where the statistic is undefined (raises ``ValueError`` or returns NaN, e.g. no
    positive left) are skipped and counted out of ``n_valid_resamples``.
    """
    if not clusters:
        raise ValueError("no clusters to resample")
    full = np.concatenate(list(clusters))
    value = statistic(full)
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(n_resamples):
        picks = rng.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[i] for i in picks])
        try:
            s = statistic(rows)
        except ValueError:
            continue
        if not np.isnan(s):
            stats.append(s)
    if not stats:
        return Estimate(value, float("nan"), float("nan"), 0)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(np.array(stats), [alpha, 1.0 - alpha])
    return Estimate(value, float(lo), float(hi), len(stats))
