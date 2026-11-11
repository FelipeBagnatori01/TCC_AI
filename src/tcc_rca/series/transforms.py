"""Feature selection and per-feature robust scaling, fitted ONLY on training rows (R2).

Both transformers take the full matrix plus a boolean ``train_rows`` mask; every statistic is
computed from the masked rows alone, so validation/test values cannot influence the fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import polars as pl


def _train_block(
    frame: pl.DataFrame, features: list[str], train_rows: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    if train_rows.dtype != np.bool_ or train_rows.shape != (frame.height,):
        raise ValueError("train_rows must be a boolean mask with one entry per row")
    if not train_rows.any():
        raise ValueError("train_rows selects no rows")
    block = frame.select(features).to_numpy().astype(np.float64)
    return block[train_rows]


@dataclass
class FeatureSelector:
    """Drops features that are mostly missing, constant or near-constant in the training rows."""

    max_missing_frac: float
    near_constant_mode_share: float
    exempt: frozenset[str] = frozenset()  # features never dropped for (near-)constancy
    kept: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)  # feature -> reason (Portuguese)
    # Exempt features kept only because of the exemption -> the reason they would have been
    # dropped. ``kept`` minus these keys is exactly the selection without exemption (control).
    rescued: dict[str, str] = field(default_factory=dict)

    def _constancy_reason(self, present: npt.NDArray[np.float64]) -> str | None:
        values, counts = np.unique(present, return_counts=True)
        if values.size <= 1:
            return "constante no treino"
        share = counts.max() / present.size
        if share >= self.near_constant_mode_share:
            return f"quase constante no treino (moda em {share:.2%})"
        return None

    def fit(
        self, frame: pl.DataFrame, features: list[str], train_rows: npt.NDArray[np.bool_]
    ) -> FeatureSelector:
        block = _train_block(frame, features, train_rows)
        self.kept, self.dropped, self.rescued = [], {}, {}
        for j, feature in enumerate(features):
            col = block[:, j]
            present = col[~np.isnan(col)]
            missing = 1.0 - present.size / col.size
            if missing > self.max_missing_frac:
                self.dropped[feature] = f"ausente em {missing:.1%} das linhas de treino"
                continue
            reason = self._constancy_reason(present)
            if reason is None:
                self.kept.append(feature)
            elif feature in self.exempt:
                self.kept.append(feature)
                self.rescued[feature] = reason
            else:
                self.dropped[feature] = reason
        return self

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        index = [c for c in frame.columns if c in ("day", "time_utc")]
        return frame.select(*index, *self.kept)


@dataclass
class RobustScaler:
    """(x - median) / IQR per feature; falls back to std, then 1.0, when the spread is zero."""

    center: dict[str, float] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    fallback: dict[str, str] = field(default_factory=dict)

    def fit(
        self, frame: pl.DataFrame, features: list[str], train_rows: npt.NDArray[np.bool_]
    ) -> RobustScaler:
        block = _train_block(frame, features, train_rows)
        self.center, self.scale, self.fallback = {}, {}, {}
        for j, feature in enumerate(features):
            col = block[:, j]
            col = col[~np.isnan(col)]
            if col.size == 0:
                raise ValueError(f"{feature}: no training values to fit the scaler")
            q1, median, q3 = np.percentile(col, [25, 50, 75])
            spread = float(q3 - q1)
            if spread == 0.0:
                spread = float(col.std())
                self.fallback[feature] = "std"
            if spread == 0.0:
                spread = 1.0
                self.fallback[feature] = "unit"
            self.center[feature] = float(median)
            self.scale[feature] = spread
        return self

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            [((pl.col(f) - self.center[f]) / self.scale[f]).alias(f) for f in self.center]
        )

    def inverse_transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            [(pl.col(f) * self.scale[f] + self.center[f]).alias(f) for f in self.center]
        )
