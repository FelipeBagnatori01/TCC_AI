"""ADR 0019: window scores read the last D + 1 minutes, oracle windows never cross days, the
time-of-day reference detector and the per-fold AUC-PR."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
from pydantic import ValidationError

from tcc_rca.config import Config, load_config
from tcc_rca.evaluation.baselines import (
    detection_metrics,
    failures_without_positive_window,
    oracle_end_row,
)
from tcc_rca.evaluation.lstm import Experiment
from tcc_rca.evaluation.metrics import auc_pr
from tcc_rca.explain.functions import ForecastScoreModule, evaluate, observed_target_mask
from tcc_rca.ingestion.cases import Failure
from tcc_rca.models.baselines.detection import time_of_day_prior_scores
from tcc_rca.models.baselines.features import last_steps, window_max_abs
from tcc_rca.models.lstm.data import normalize
from tcc_rca.models.lstm.model import ForecastLSTM
from tcc_rca.models.lstm.train import ForecastScorer

W, F = 8, 3


@pytest.fixture(autouse=True, scope="module")
def _single_torch_thread():  # type: ignore[no-untyped-def]
    """Same reason as in test_lstm.py: LightGBM's OpenMP vs multithreaded torch (ADR 0015)."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# ------------------------------------------------------------------------------ config
def test_base_config_scores_the_last_d_plus_one_minutes(repo_root: Path) -> None:
    t = load_config(repo_root / "configs/base.yaml").tasks
    assert t.score_minutes == t.detection_tolerance_minutes + 1


def test_score_span_cannot_exceed_the_forecaster_steps(repo_root: Path) -> None:
    with pytest.raises(ValidationError, match="score_minutes"):
        load_config(repo_root / "configs/base.yaml", overrides={"tasks": {"score_minutes": 20}})
    cfg = load_config(repo_root / "configs/base.yaml", overrides={"tasks": {"score_minutes": None}})
    assert cfg.tasks.score_minutes is None  # the whole-window control


# ------------------------------------------------------------------------- window scores
def test_last_steps_hand_computed() -> None:
    x = np.arange(12, dtype=np.float64).reshape(2, 6)
    assert last_steps(x, 2).tolist() == [[4.0, 5.0], [10.0, 11.0]]
    assert last_steps(x, None) is x
    with pytest.raises(ValueError, match="last 7"):
        last_steps(x, 7)


def test_zscore_window_score_ignores_an_early_spike() -> None:
    z = np.zeros((6, 2))
    z[1, 0] = 9.0  # spike in the 2nd minute of the window ending at row 5
    z[5, 1] = -2.0  # small deviation in the last minute
    end = np.array([5])
    assert window_max_abs(z, end, 6).tolist() == [9.0]  # whole window: the old spike wins
    assert window_max_abs(z, end, 6, score_minutes=2).tolist() == [2.0]


def _sine(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(W)[None, :, None]
    x = np.sin(0.6 * t + rng.uniform(0, 2 * np.pi, (n, 1, F))) + 0.05 * rng.normal(size=(n, W, F))
    return x.astype(np.float32)


def test_forecaster_score_reads_only_the_last_minutes() -> None:
    torch.manual_seed(0)
    model = ForecastLSTM(F, 8, 1, 0.0).eval()
    normal = _sine(64, 0)
    whole = ForecastScorer.fit(model, normal, top_k=1)
    tail = ForecastScorer.fit(model, normal, top_k=1, score_minutes=3)
    assert np.array_equal(whole.scale, tail.scale)  # the span does not change the fit
    early = _sine(1, 1)
    early[0, 2, 0] += 1000.0  # error spike at target step 2 (error index 1), far from the end
    assert whole.score_windows(early)[0] > 100.0
    assert tail.score_windows(early)[0] < whole.score_windows(early)[0] / 20.0
    late = _sine(1, 1)
    late[0, -1, 0] += 1000.0  # the same spike in the last minute counts in both
    assert tail.score_windows(late)[0] == pytest.approx(whole.score_windows(late)[0])


@pytest.mark.parametrize("span", [None, 3])
def test_explained_module_equals_the_scorer_for_any_span(span: int | None) -> None:
    torch.manual_seed(0)
    model = ForecastLSTM(F, 8, 1, 0.0).eval()
    windows = _sine(5, 2)
    scorer = ForecastScorer.fit(model, windows, top_k=2, score_minutes=span)
    windows[1, -2, 0] = np.nan
    expected = scorer.score_windows(windows)
    module = ForecastScoreModule(model, scorer.scale, top_k=2, score_minutes=span)
    for i in range(windows.shape[0]):
        module.set_mask(observed_target_mask(windows[i]))
        got = evaluate(module, np.nan_to_num(windows[i : i + 1], nan=0.0))[0]
        assert got == pytest.approx(expected[i], rel=1e-5)


# ------------------------------------------------------------------------ oracle windows
def _two_days(minutes: int) -> tuple[dict[float, int], np.ndarray, list[datetime]]:
    """Two days of ``minutes`` rows each, 18 h apart (Telecom-like)."""
    starts = [datetime(2020, 5, 1, 16, tzinfo=UTC), datetime(2020, 5, 2, 16, tzinfo=UTC)]
    times = [s + timedelta(minutes=m) for s in starts for m in range(minutes)]
    days = np.array(["d1"] * minutes + ["d2"] * minutes)
    return {t.timestamp() / 60.0: i for i, t in enumerate(times)}, days, starts


def test_oracle_window_never_crosses_days() -> None:
    minute_to_row, days, starts = _two_days(60)
    window, tol = 20, 5
    late = oracle_end_row(minute_to_row, days, starts[1] + timedelta(minutes=30), tol, window)
    assert late == 60 + 35  # the usual window: ends `tol` minutes after the onset
    for onset_minute in (0, 5, 13, 14):
        end = oracle_end_row(
            minute_to_row, days, starts[1] + timedelta(minutes=onset_minute), tol, window
        )
        rows = np.arange(end - window + 1, end + 1)
        assert set(days[rows].tolist()) == {"d2"}
        assert end == 60 + window - 1  # first complete window of the day, onset included
    first_day = oracle_end_row(minute_to_row, days, starts[0] + timedelta(minutes=2), tol, window)
    assert first_day == window - 1  # also at the very first row of the matrix
    short, short_days, short_starts = _two_days(10)
    with pytest.raises(ValueError, match="shorter than one window"):
        oracle_end_row(short, short_days, short_starts[1], 2, window)


def _failure(start: datetime) -> Failure:
    return Failure("l", "r", "c", start)


def test_failures_without_positive_window_counts_early_onsets() -> None:
    t0 = datetime(2020, 5, 1, 16, tzinfo=UTC)
    windows = pl.DataFrame(
        {
            "end_utc": [t0 + timedelta(minutes=m) for m in (30, 31, 50)],
            "label": [1, 1, 0],
        }
    ).with_columns(pl.col("label").cast(pl.Int8))
    covered = _failure(t0 + timedelta(minutes=27))  # positive window ends 3 min after it
    uncovered = _failure(t0 + timedelta(minutes=45))  # only a negative window nearby
    assert failures_without_positive_window(windows, [covered, uncovered], 5) == 1


# ------------------------------------------------------------------- time-of-day reference
def test_time_of_day_prior_hand_computed_and_circular() -> None:
    train_minute = np.array([100, 100, 101, 1439, 700], dtype=np.int64)
    train_y = np.array([1, 0, 1, 1, 0])
    scores = time_of_day_prior_scores(
        train_minute, train_y, np.array([100, 103, 0, 702, 400], dtype=np.int64), halfwidth=2
    )
    # 100: windows 100,100,101 -> 2/3; 103: only 101 -> 1/1; 0: 1439 wraps -> 1/1;
    # 702: 700 -> 0/1; 400: nothing within 2 min -> 0
    assert scores.tolist() == pytest.approx([2 / 3, 1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="minutes of the day"):
        time_of_day_prior_scores(train_minute, train_y, np.array([1440]), 2)


# --------------------------------------------------------------------- per-fold AUC-PR
def test_detection_metrics_report_the_mean_of_per_fold_auc_pr(repo_root: Path) -> None:
    cfg = load_config(repo_root / "configs/base.yaml", overrides={"baselines": {"bootstrap": 50}})
    t0 = datetime(2020, 5, 1, 16, tzinfo=UTC)
    rng = np.random.default_rng(0)
    n = 120
    fold = np.repeat([0, 1], n // 2)
    label = (rng.uniform(size=n) < 0.3).astype(np.int64)
    label[[0, 60]] = 1
    # fold 1 scores are shifted up: pooling mixes the two scales, the per-fold mean does not
    score = rng.normal(size=n) + 2.0 * label + 5.0 * fold
    frame = pl.DataFrame(
        {
            "fold": fold,
            "day": np.where(fold == 0, "d1", "d2"),
            "end_utc": [t0 + timedelta(days=int(f), minutes=i) for i, f in enumerate(fold)],
            "label": label,
            "score": score,
            "alarm": score > 4.0,
        }
    )
    onset = t0.timestamp() / 60.0 + 10.0
    out = detection_metrics(frame, np.array([onset]), ["r"], cfg)
    expected = np.mean([auc_pr(label[fold == f], score[fold == f]) for f in (0, 1)])
    assert out["auc_pr_fold_mean"]["value"] == pytest.approx(expected, abs=1e-4)
    assert out["auc_pr_fold_mean"]["ci_low"] <= expected <= out["auc_pr_fold_mean"]["ci_high"]


# ---------------------------------------------------------------- cached local inputs
def test_cached_local_inputs_equal_the_per_fold_normalization(config: Config) -> None:
    rng = np.random.default_rng(0)
    rows, feats = 40, ["a::x", "b::y", "c::z"]
    values = pl.DataFrame(
        {"day": ["d1"] * rows + ["d2"] * rows, **{f: rng.normal(size=2 * rows) for f in feats}}
    )
    exp = Experiment.__new__(Experiment)  # only what ``inputs`` reads
    exp.config, exp.t, exp.values, exp.features = config, config.tasks, values, feats
    exp.normal_min = np.ones(2 * rows, dtype=np.bool_)
    exp.day_arr = values["day"].to_numpy()
    exp._column_of = {f: j for j, f in enumerate(feats)}
    exp._local = None
    fd = SimpleNamespace(kept=["c::z", "a::x"], fold=SimpleNamespace(train_days=("d1",)))
    p = config.tasks.precursor
    for _ in range(2):  # first call fills the cache, the second reads it
        got = exp.inputs(fd, "local")  # type: ignore[arg-type]
        ref = normalize(
            values,
            fd.kept,
            exp.normal_min,
            "local",
            config.lstm.global_clip,
            p.local_window_minutes,
            p.local_cap,
        )
        assert got.features == ref.features
        assert np.array_equal(got.matrix, ref.matrix, equal_nan=True)
