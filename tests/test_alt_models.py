"""Alternative explainable models (ADR 0018): statistics, exact SHAP, training rows, TCN."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tcc_rca.config import Config
from tcc_rca.evaluation.alt_models import Candidate, TabularFitted, TcnFitted, fit, grid
from tcc_rca.explain.attributions import lime_groups
from tcc_rca.explain.tabular import shap_series, shap_values_stats
from tcc_rca.models.baselines.features import window_statistics
from tcc_rca.models.lstm.train import TrainSettings, classifier_scores, fit_window_classifier
from tcc_rca.models.tabular import (
    fit_isolation_forest,
    fit_lightgbm,
    fit_logistic,
    series_attribution,
    tensor_statistics,
)
from tcc_rca.models.tcn import ClassifierTCN

W, F = 6, 5


@pytest.fixture(autouse=True, scope="module")
def _single_torch_thread():  # type: ignore[no-untyped-def]
    """LightGBM is fitted in this process too (see tests/test_lstm.py)."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _windows(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, W, F)).astype(np.float32)


def _labelled(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Failure windows have series 2 shifted up over the last minutes."""
    x = _windows(n, seed)
    y = (np.arange(n) % 4 == 0).astype(np.int64)
    x[y == 1, -3:, 2] += 4.0
    return x, y


# ------------------------------------------------------------------------------ statistics
def test_tensor_statistics_equal_the_e4_window_statistics() -> None:
    matrix = np.random.default_rng(0).normal(size=(30, F))
    matrix[3:9, 1] = np.nan  # a series missing for part of a window
    ends = np.array([W - 1, 10, 29], dtype=np.int64)
    expected = window_statistics(matrix, ends, W)
    tensors = np.stack([matrix[e - W + 1 : e + 1] for e in ends]).astype(np.float32)
    np.testing.assert_allclose(tensor_statistics(tensors), expected, rtol=1e-5, atol=1e-6)


def test_series_attribution_sums_the_five_statistics_of_each_series() -> None:
    attr = np.zeros((1, 5 * 3))
    attr[0, [0, 3, 6, 9, 12]] = 1.0  # every statistic of series 0
    attr[0, 5] = 2.0  # std of series 2
    np.testing.assert_array_equal(series_attribution(attr, 3), [[5.0, 0.0, 2.0]])
    with pytest.raises(ValueError, match="5 statistics"):
        series_attribution(np.zeros((1, 7)), 3)


# ---------------------------------------------------------------------------- exact SHAP
def test_tree_and_linear_shap_add_up_to_the_explained_output() -> None:
    x, y = _labelled(200, 0)
    stats, bg = tensor_statistics(x), tensor_statistics(x[y == 0][:40])
    lgbm = fit_lightgbm(
        stats,
        y,
        num_leaves=7,
        n_estimators=30,
        learning_rate=0.1,
        min_child_samples=5,
        colsample_bytree=1.0,
        seed=0,
    )
    values, expected = shap_values_stats(lgbm, stats[:5], bg)
    raw = lgbm.model.predict(stats[:5], raw_score=True)
    np.testing.assert_allclose(values.sum(axis=1) + expected, raw, rtol=1e-4, atol=1e-4)
    logistic = fit_logistic(stats, y, c=1.0, max_iter=200, seed=0)
    values, expected = shap_values_stats(logistic, stats[:5], bg)
    np.testing.assert_allclose(
        values.sum(axis=1) + expected, logistic.model.decision_function(stats[:5]), rtol=1e-4
    )
    # the shifted series carries the attribution of a failure window, for both models
    for model in (lgbm, logistic):
        per_series = shap_series(model, x[:1], x[y == 0][:40])[0]
        assert int(np.argmax(per_series)) == 2


def test_isolation_forest_shap_points_to_the_anomalous_series_with_a_positive_sign() -> None:
    normal = _windows(300, 1)
    model = fit_isolation_forest(
        tensor_statistics(normal), n_estimators=100, max_samples=256, seed=0
    )
    odd = _windows(1, 2)
    odd[0, :, 4] += 8.0
    assert model.score_windows(odd)[0] > np.median(model.score_windows(normal))
    per_series = shap_series(model, odd, normal[:50])[0]
    assert int(np.argmax(per_series)) == 4
    assert per_series[4] > 0  # positive = towards "more anomalous", as every other explanation


def test_lime_on_a_tabular_model_uses_the_same_series_perturbation() -> None:
    x, y = _labelled(200, 3)
    model = fit_lightgbm(
        tensor_statistics(x),
        y,
        num_leaves=7,
        n_estimators=30,
        learning_rate=0.1,
        min_child_samples=5,
        colsample_bytree=1.0,
        seed=0,
    )
    reference = np.median(x[y == 0], axis=0).astype(np.float32)
    weights = lime_groups(model.score_windows, x[0], reference, 300, 25.0, seed=0)
    assert int(np.argmax(weights)) == 2


# ------------------------------------------------------------------- protocol and training
def test_grid_has_eight_configurations_per_model_smallest_first(config: Config) -> None:
    for model in ("lightgbm", "isolation_forest", "logistic", "tcn"):
        cands = grid(config, model)
        assert len(cands) == 8
        assert len(set(cands)) == 8
        assert Candidate.from_dict(model, cands[3].as_dict()) == cands[3]
    first = grid(config, "lightgbm")[0]
    assert first.normalization == "global"
    assert dict(first.params) == {"n_estimators": 300, "num_leaves": 15}


class _RecordingView:
    """Stand-in for FoldView: records which windows the fit reads."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        n = y.size
        role = np.array(["train"] * (n - 40) + ["val"] * 20 + ["test"] * 20)
        self.masks = {
            "train_labelled": role == "train",
            "train_normal": (role == "train") & (y == 0),
            "val_labelled": role == "val",
            "val_normal": (role == "val") & (y == 0),
            "test": role == "test",
        }
        self.fd = SimpleNamespace(y=y)
        self.matrix = np.zeros((1, F), dtype=np.float32)
        self.x, self.read = x, []  # type: ignore[var-annotated]

    def stats(self, mask: str) -> np.ndarray:
        self.read.append(mask)
        return tensor_statistics(self.x[self.masks[mask]])

    def masked_tensors(self, mask: str) -> np.ndarray:
        self.read.append(mask)
        return self.x[self.masks[mask]]


@pytest.mark.parametrize(
    ("model", "params", "allowed"),
    [
        ("lightgbm", {"num_leaves": 7, "n_estimators": 20}, {"train_labelled"}),
        ("isolation_forest", {"n_estimators": 20, "max_samples": 64}, {"train_normal"}),
        ("logistic", {"c": 1.0}, {"train_labelled"}),
        # validation only for early stopping, as the LSTM classifier
        ("tcn", {"channels": 4, "blocks": 2}, {"train_labelled", "val_labelled"}),
    ],
)
def test_fit_reads_training_windows_only(
    config: Config, model: str, params: dict[str, Any], allowed: set[str]
) -> None:
    x, y = _labelled(200, 4)
    view = _RecordingView(x, y)
    cfg = config.model_copy(update={"lstm": config.lstm.model_copy(update={"max_epochs": 2})})
    cand = Candidate(model, "local", tuple(sorted(params.items())))
    fitted, _ = fit(cand, view, seed=0, config=cfg)  # type: ignore[arg-type]
    assert set(view.read) <= allowed
    assert "test" not in view.read
    expected = TcnFitted if model == "tcn" else TabularFitted
    assert isinstance(fitted, expected)


def test_tcn_is_reproducible_and_learns_the_shifted_series() -> None:
    x, y = _labelled(256, 5)
    xv, yv = _labelled(64, 6)
    s = TrainSettings(8, 2, 0.0, 0.01, 32, 15, 50, seed=3)

    def train() -> ClassifierTCN:
        model, _ = fit_window_classifier(lambda: ClassifierTCN(F, 8, 2, 0.0), x, y, xv, yv, s)
        return model

    a, b = train(), train()
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb)
    scores = classifier_scores(a, xv)
    assert scores[yv == 1].mean() > scores[yv == 0].mean() + 0.3
    assert yv[0] == 1  # failure window: series 2 shifted
    attr = TcnFitted(a, shap_samples=50).shap(xv[:1], x[y == 0][:20], seed=0)
    assert attr.shape == (1, F)
    assert int(np.argmax(attr[0])) == 2


def test_tcn_is_causal() -> None:
    torch.manual_seed(0)
    model = ClassifierTCN(F, 4, 2, 0.0)
    conv_out = model.blocks(torch.zeros(1, F, W))
    x = torch.zeros(1, F, W)
    x[0, :, -1] = 5.0  # change only the last step
    changed = model.blocks(x)
    torch.testing.assert_close(changed[..., :-1], conv_out[..., :-1])  # earlier steps unaffected
