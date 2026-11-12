"""E4 detection baselines: Isolation Forest and LightGBM on window statistics, plus thresholds.

The z-score baseline needs no model: its window score is ``features.window_max_abs``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import IsolationForest

from tcc_rca.config import LightGBMConfig
from tcc_rca.evaluation.metrics import best_f1_threshold

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]


def isolation_forest_scores(
    train: Float32Array, everything: Float32Array, n_estimators: int, seed: int
) -> FloatArray:
    """Fit on normal training windows only; higher score = more anomalous."""
    model = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=1)
    model.fit(np.nan_to_num(train, nan=0.0))  # 0 = the training median after robust scaling
    scores: FloatArray = -model.score_samples(np.nan_to_num(everything, nan=0.0))
    return scores.astype(np.float64)


def lightgbm_scores(
    x_train: Float32Array,
    y_train: npt.NDArray[np.int_],
    everything: Float32Array,
    cfg: LightGBMConfig,
    seed: int,
) -> FloatArray:
    """Supervised window classifier with class balancing; returns P(failure window)."""
    # Imported here, not at module level: once LightGBM has trained, its OpenMP runtime makes
    # multithreaded PyTorch crash in the same process (macOS). Keeping the import local means the
    # LSTM command never loads it (ADR 0015).
    import lightgbm as lgb

    pos = int(y_train.sum())
    if pos == 0 or pos == y_train.size:
        raise ValueError("LightGBM needs both positive and negative training windows")
    model = lgb.LGBMClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        colsample_bytree=cfg.colsample_bytree,
        scale_pos_weight=(y_train.size - pos) / pos,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        # One thread: with PyTorch's libomp already loaded (Run seeds torch), multithreaded
        # LightGBM aborts with "OMP: pthread_mutex_init failed" on macOS (ADR 0010).
        num_threads=1,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    proba = np.asarray(model.predict_proba(everything), dtype=np.float64)
    out: FloatArray = proba[:, 1]
    return out


def choose_threshold(
    val_labels: npt.NDArray[np.int_], val_scores: FloatArray, fallback_quantile: float
) -> tuple[float, str]:
    """Best F1 on validation windows; without validation positives, a normal-score quantile."""
    if val_labels.size == 0:
        raise ValueError("no labelled validation windows")
    if val_labels.sum() > 0:
        return best_f1_threshold(val_labels, val_scores), "best_f1_val"
    normal = val_scores[val_labels == 0]
    return float(np.quantile(normal, fallback_quantile)), f"q{fallback_quantile:g}_val_normal"
