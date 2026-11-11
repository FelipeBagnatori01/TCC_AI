"""E4 localization heuristic: rank components by their largest |z| in the failure window."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from tcc_rca.ingestion.cases import Failure
from tcc_rca.series.catalog import split_feature

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def component_columns(features: Sequence[str]) -> dict[str, IntArray]:
    """Component -> column indices of its features in ``features``."""
    cols: dict[str, list[int]] = {}
    for j, f in enumerate(features):
        cols.setdefault(split_feature(f)[0], []).append(j)
    return {c: np.array(v, dtype=np.int64) for c, v in cols.items()}


def zscore_component_scores(
    z: FloatArray, rows: IntArray, columns: dict[str, IntArray], candidates: Sequence[str]
) -> dict[str, float]:
    """max |z| over the window rows and the component's features; missing values count as 0.

    A candidate with no selected feature in this fold scores 0 (no evidence against it).
    """
    block = np.nan_to_num(np.abs(z[rows]), nan=0.0)
    per_feature = block.max(axis=0)
    return {c: float(per_feature[columns[c]].max()) if c in columns else 0.0 for c in candidates}


def local_component_scores(
    local_z: FloatArray,
    rows: IntArray,
    columns: dict[str, IntArray],
    candidates: Sequence[str],
    top: int = 3,
) -> dict[str, float]:
    """Mean of the component's ``top`` largest window-max causal local |z| (ADR 0014).

    Local |z| is capped, so a plain maximum would tie many components at the cap; averaging
    the top three features keeps the ranking informative. Missing values count as 0.
    """
    per_feature = np.nan_to_num(local_z[rows], nan=0.0).max(axis=0)
    out: dict[str, float] = {}
    for c in candidates:
        if c not in columns:
            out[c] = 0.0
            continue
        vals = np.sort(per_feature[columns[c]])[::-1][:top]
        out[c] = float(vals.mean())
    return out


def prior_component_scores(
    training_failures: Sequence[Failure], candidates: Sequence[str]
) -> dict[str, float]:
    """Reference ranking without telemetry: how often each candidate was the root cause in the
    fold's training/validation failures (the localization analogue of the constant baseline)."""
    counts = Counter(f.component for f in training_failures)
    return {c: float(counts.get(c, 0)) for c in candidates}
