"""Tabular window models compared with the LSTM (ADR 0018): LightGBM, Isolation Forest and L1
logistic regression over window statistics.

Every model reads the SAME normalized window the LSTM reads (fold transform, missing = 0) through
``tensor_statistics`` (mean, std, min, max, last per series, the E4 statistics), so SHAP/LIME
explanations can be expressed per ``component::kpi`` series for every model. Scores: failure
probability (LightGBM, logistic) or anomaly score ``-score_samples`` (Isolation Forest); higher is
always "more likely a failure window". Fitting uses only the rows the caller passes (training
windows of the fold).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from tcc_rca.models.baselines.features import STATS

Float32Array = npt.NDArray[np.float32]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
TabularKind = Literal["lightgbm", "isolation_forest", "logistic"]


def tensor_statistics(windows: Float32Array) -> Float32Array:
    """(n, W, F) windows -> (n, 5 * F) statistics, column layout of ``window_statistics``
    (all means, then all stds, mins, maxs, lasts). Inputs are expected NaN-free (filled)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        parts = [
            np.nanmean(windows, axis=1),
            np.nanstd(windows, axis=1),
            np.nanmin(windows, axis=1),
            np.nanmax(windows, axis=1),
            windows[:, -1, :],
        ]
    out: Float32Array = np.concatenate(parts, axis=1).astype(np.float32)
    return out


def series_attribution(attr: FloatArray | Float32Array, n_series: int) -> FloatArray:
    """(n, 5 * F) attributions over statistics -> (n, F): the 5 statistics of a series summed."""
    a = np.asarray(attr, dtype=np.float64)
    if a.shape[1] != len(STATS) * n_series:
        raise ValueError("attribution width does not match 5 statistics per series")
    out: FloatArray = a.reshape(a.shape[0], len(STATS), n_series).sum(axis=1)
    return out


@dataclass
class TabularModel:
    kind: TabularKind
    model: Any

    def score_stats(self, stats: Float32Array) -> FloatArray:
        if self.kind == "isolation_forest":
            out: FloatArray = -np.asarray(self.model.score_samples(stats), dtype=np.float64)
            return out
        proba = np.asarray(self.model.predict_proba(stats), dtype=np.float64)
        pos: FloatArray = proba[:, 1]
        return pos

    def score_windows(self, windows: Float32Array) -> FloatArray:
        """Score of filled windows ``(n, W, F)``: the function LIME and fidelity perturb."""
        return self.score_stats(tensor_statistics(windows))


def _check_labels(y: npt.NDArray[np.int64]) -> int:
    pos = int(y.sum())
    if pos == 0 or pos == y.size:
        raise ValueError("a supervised model needs positive and negative training windows")
    return pos


def fit_lightgbm(
    x: Float32Array,
    y: npt.NDArray[np.int64],
    *,
    num_leaves: int,
    n_estimators: int,
    learning_rate: float,
    min_child_samples: int,
    colsample_bytree: float,
    seed: int,
) -> TabularModel:
    """Same recipe as the E4 LightGBM (class balancing, 1 thread, deterministic), with the
    searched ``num_leaves`` / ``n_estimators``."""
    import lightgbm as lgb  # lazy: its OpenMP runtime must not meet multithreaded torch (ADR 0015)

    pos = _check_labels(y)
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=(y.size - pos) / pos,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        num_threads=1,
        verbose=-1,
    )
    model.fit(x, y)
    return TabularModel("lightgbm", model)


def fit_isolation_forest(
    x_normal: Float32Array, *, n_estimators: int, max_samples: int, seed: int
) -> TabularModel:
    """Unsupervised: fitted on NORMAL training windows only (as the LSTM forecaster)."""
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=min(max_samples, x_normal.shape[0]),
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x_normal)
    return TabularModel("isolation_forest", model)


def fit_logistic(
    x: Float32Array, y: npt.NDArray[np.int64], *, c: float, max_iter: int, seed: int
) -> TabularModel:
    """L1-penalized logistic regression with balanced classes (liblinear)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    _check_labels(y)
    model = LogisticRegression(
        l1_ratio=1.0,  # pure L1 (``penalty`` is deprecated since scikit-learn 1.8)
        C=c,
        solver="liblinear",
        class_weight="balanced",
        max_iter=max_iter,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)  # counted by the caller
        model.fit(x, y)
    return TabularModel("logistic", model)


def converged(model: TabularModel) -> bool | None:
    """Whether liblinear stopped before ``max_iter`` (None for the tree models)."""
    if model.kind != "logistic":
        return None
    return bool(np.max(model.model.n_iter_) < model.model.max_iter)
