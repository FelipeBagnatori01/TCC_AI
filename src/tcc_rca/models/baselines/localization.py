"""E4 localization heuristic: rank components by their largest |z| in the failure window."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

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
