"""E5 LSTM (ADR 0015): determinism, learning sanity, normalization and native ranking."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

from tcc_rca.evaluation.lstm import _component_scores
from tcc_rca.models.baselines.localization import component_columns
from tcc_rca.models.lstm.data import normalize, window_tensor
from tcc_rca.models.lstm.train import (
    ForecastScorer,
    TrainSettings,
    classifier_scores,
    train_classifier,
    train_forecaster,
)

W, F = 8, 4


@pytest.fixture(autouse=True, scope="module")
def _single_torch_thread():  # type: ignore[no-untyped-def]
    """LightGBM (fitted earlier in the same pytest process by test_baselines) leaves an OpenMP
    runtime that crashes multithreaded PyTorch on macOS; the CLI runs them in separate processes
    (ADR 0015). One thread here keeps the suite safe without changing what is tested."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _sine_windows(n: int, seed: int, noise: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, (n, 1, F))
    t = np.arange(W)[None, :, None]
    x = np.sin(0.6 * t + phase) + noise * rng.normal(size=(n, W, F))
    return x.astype(np.float32)


def _settings(seed: int = 0, epochs: int = 25) -> TrainSettings:
    return TrainSettings(
        hidden=16,
        layers=1,
        dropout=0.0,
        learning_rate=0.01,
        batch_size=32,
        max_epochs=epochs,
        patience=50,
        seed=seed,
    )


@pytest.fixture(scope="module")
def trained() -> tuple[ForecastScorer, np.ndarray]:
    train = _sine_windows(256, 0)
    model, _ = train_forecaster(train, _sine_windows(64, 1), _settings())
    return ForecastScorer.fit(model, train, top_k=2), train


def test_score_is_deterministic_and_pure(trained) -> None:  # type: ignore[no-untyped-def]
    scorer, _ = trained
    w = _sine_windows(1, 7)[0]
    before = {k: v.clone() for k, v in scorer.model.state_dict().items()}
    first = scorer.score(torch.from_numpy(w))
    assert scorer.score(w) == first  # same input (tensor or array) -> same value
    assert scorer.score(w[None]) == first  # (1, W, F) accepted
    after = scorer.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)  # no side effects
    assert not scorer.model.training


def test_training_is_reproducible_with_the_same_seed() -> None:
    train, val = _sine_windows(128, 0), _sine_windows(32, 1)
    a, ha = train_forecaster(train, val, _settings(seed=3, epochs=4))
    b, hb = train_forecaster(train, val, _settings(seed=3, epochs=4))
    assert ha.train_loss == hb.train_loss
    assert all(
        torch.equal(x, y)
        for x, y in zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    )


def test_forecaster_scores_an_anomalous_window_higher(trained) -> None:  # type: ignore[no-untyped-def]
    scorer, _ = trained
    normal = _sine_windows(50, 9)
    broken = normal.copy()
    broken[:, -1, 0] += 4.0  # a jump in the last step of feature 0
    assert np.median(scorer.score_windows(broken)) > 3 * np.median(scorer.score_windows(normal))
    errs = scorer.normalized_errors(broken)
    assert errs.shape == (50, W - 1, F)
    assert (errs[:, -1, 0] > errs[:, -1, 1:].max(axis=1)).mean() > 0.9  # the feature is pinpointed


def test_missing_targets_do_not_count(trained) -> None:  # type: ignore[no-untyped-def]
    scorer, _ = trained
    w = _sine_windows(1, 11)
    w_missing = w.copy()
    w_missing[0, -1, 0] = np.nan
    assert scorer.normalized_errors(w_missing)[0, -1, 0] == 0.0


def test_classifier_learns_and_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, W, F)).astype(np.float32)
    y = (rng.uniform(size=200) < 0.2).astype(np.int64)
    x[y == 1, -2:, 1] += 3.0
    model, _ = train_classifier(x, y, x[:60], y[:60], _settings(seed=1, epochs=15))
    p = classifier_scores(model, x)
    assert p[y == 1].mean() > p[y == 0].mean() + 0.3
    again, _ = train_classifier(x, y, x[:60], y[:60], _settings(seed=1, epochs=15))
    assert np.array_equal(classifier_scores(again, x), p)
    with pytest.raises(ValueError, match="positive and negative"):
        train_classifier(x, np.zeros(200, dtype=np.int64), x, y, _settings())


def test_forecaster_refuses_empty_training() -> None:
    with pytest.raises(ValueError, match="no normal training windows"):
        train_forecaster(
            np.zeros((0, W, F), np.float32), np.zeros((0, W, F), np.float32), _settings()
        )


def test_global_normalization_fits_training_rows_only() -> None:
    frame = pl.DataFrame({"day": ["d"] * 6, "a::x": [1.0, 2.0, 3.0, 4.0, 1e6, -1e6]})
    train = np.array([True] * 4 + [False] * 2)
    out = normalize(
        frame, ["a::x"], train, "global", global_clip=10.0, local_window=3, local_cap=5.0
    )
    # median 2.5, IQR 1.5 of the four training rows; huge test values are clipped at +-10
    assert out.matrix[:4, 0].tolist() == pytest.approx([-1.0, -1 / 3, 1 / 3, 1.0])
    assert out.matrix[4:, 0].tolist() == [10.0, -10.0]


def test_window_tensor_rows() -> None:
    m = np.arange(12, dtype=np.float32).reshape(6, 2)
    out = window_tensor(m, np.array([2, 5]), 3)
    assert out[0].tolist() == [[0, 1], [2, 3], [4, 5]]
    assert out[1, -1].tolist() == [10, 11]


def test_native_ranking_sum_and_top3_hand_computed() -> None:
    feats = ["os_1::a", "os_1::b", "os_1::c", "os_1::d", "db_1::a"]
    errors = np.array([[1.0, 0.5, 0.0, 2.0, 3.0], [4.0, 0.5, 1.0, 0.0, 0.0]], dtype=np.float32)
    # per-feature max over steps: [4, 0.5, 1, 2, 3]
    out = _component_scores(errors, component_columns(feats), ["db_1", "os_1", "x_9"])
    assert out["lstm_sum"] == {"db_1": 3.0, "os_1": 7.5, "x_9": 0.0}
    assert out["lstm_top3"]["os_1"] == pytest.approx((4 + 2 + 1) / 3)
    assert out["lstm_top3"]["db_1"] == 3.0
