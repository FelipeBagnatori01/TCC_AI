"""E3 acceptance tests on a synthetic minute grid (never real data)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from tcc_rca.dataset.split import (
    day_of,
    fold_of_test_day,
    min_gap_to_test_minutes,
    rolling_origin_folds,
    with_roles,
)
from tcc_rca.dataset.windows import WindowSpec, build_window_index, label_windows

# Six 60-minute "days" at 16:00 UTC (00:00 UTC+8), like OpenRCA Telecom days.
DAYS = ["2020_04_11", "2020_04_20", "2020_04_21", "2020_05_22", "2020_05_23", "2020_05_24"]


def _day_start(day: str) -> datetime:
    y, m, d = (int(p) for p in day.split("_"))
    return datetime(y, m, d, tzinfo=UTC) - timedelta(hours=8)


def _index(minutes: int = 60) -> pl.DataFrame:
    rows = [(d, _day_start(d) + timedelta(minutes=i)) for d in DAYS for i in range(minutes)]
    return pl.DataFrame(rows, schema=["day", "time_utc"], orient="row").with_columns(
        pl.col("time_utc").dt.cast_time_unit("us")
    )


def _at(day: str, minute: int) -> datetime:
    return _day_start(day) + timedelta(minutes=minute)


ONSETS = [_at("2020_04_11", 30), _at("2020_05_22", 20), _at("2020_05_23", 40)]
SPEC = WindowSpec(window_minutes=10, margin_before_minutes=5, margin_after_minutes=15)


def _windows(spec: WindowSpec = SPEC) -> pl.DataFrame:
    w = build_window_index(_index(), ONSETS, spec)
    return w.with_columns(label_windows(w, spec.horizon_minutes, 3))


def _row(windows: pl.DataFrame, day: str, minute: int) -> dict:  # type: ignore[type-arg]
    return windows.filter(pl.col("end_utc") == _at(day, minute)).row(0, named=True)


def test_windows_stay_inside_one_day_and_count() -> None:
    w = _windows()
    assert w.height == len(DAYS) * (60 - 10 + 1)
    assert (w["end_utc"] - w["start_utc"]).dt.total_minutes().unique().to_list() == [9]
    starts = w.group_by("day").agg(pl.col("start_utc").min())
    assert all(s == _day_start(d) for d, s in starts.iter_rows())


def test_labels_hand_computed() -> None:
    w = _windows()  # W=10, D=3, H=0, margins 5 before / 15 after; onset 05_22 at minute 20
    assert _row(w, "2020_05_22", 20)["label"] == 1  # onset at t
    assert _row(w, "2020_05_22", 23)["label"] == 1  # 3 min after onset: within D
    assert _row(w, "2020_05_22", 24)["label"] is None  # after D, inside the margin: ignored
    assert _row(w, "2020_05_22", 19)["label"] is None  # 1 min before onset, H=0: ignored
    assert _row(w, "2020_05_22", 15)["label"] is None  # window [6, 15] touches onset-5 = 15
    assert _row(w, "2020_05_22", 14)["label"] == 0  # window [5, 14] ends before onset-5
    # after: window must start after onset+15 = 35 -> ends at >= 45 with W=10
    assert _row(w, "2020_05_22", 44)["label"] is None
    assert _row(w, "2020_05_22", 45)["label"] == 0
    r = _row(w, "2020_05_22", 18)
    assert r["minutes_to_next_onset"] == 2.0
    assert r["minutes_since_last_onset"] > 3  # previous onset is on 04_11, far outside D


def test_horizon_makes_prediction_windows_positive_and_never_normal() -> None:
    spec = WindowSpec(10, margin_before_minutes=2, margin_after_minutes=15, horizon_minutes=6)
    w = _windows(spec)
    r = _row(w, "2020_05_22", 14)  # onset 6 min ahead
    assert r["label"] == 1
    assert not r["normal"]  # effective before-margin = max(2, H) = 6
    assert _row(w, "2020_05_22", 13)["label"] == 0
    assert not w.filter(pl.col("label") == 1)["normal"].any()


def test_no_lstm_training_window_touches_a_failure_margin() -> None:
    """Acceptance: no normal (LSTM-trainable) window contains an instant within a margin."""
    w = _windows().filter(pl.col("normal"))
    before, after = timedelta(minutes=5), timedelta(minutes=15)
    for onset in ONSETS:
        overlap = w.filter(
            (pl.col("start_utc") <= onset + after) & (pl.col("end_utc") >= onset - before)
        )
        assert overlap.is_empty()
    assert w.height > 0


def test_rolling_origin_folds_and_exclusion() -> None:
    folds = rolling_origin_folds(DAYS, ["2020_04_21"], "2020_05_22", val_days=1)
    assert [f.test_day for f in folds] == ["2020_05_22", "2020_05_23", "2020_05_24"]
    assert folds[0].train_days == ("2020_04_11",)
    assert folds[0].val_days == ("2020_04_20",)
    assert folds[2].train_days == ("2020_04_11", "2020_04_20", "2020_05_22")
    assert folds[2].val_days == ("2020_05_23",)
    for f in folds:
        assert "2020_04_21" not in (*f.train_days, *f.val_days, f.test_day)
        assert max((*f.train_days, *f.val_days)) < f.test_day  # chronological
    assert fold_of_test_day(folds) == {"2020_05_22": 0, "2020_05_23": 1, "2020_05_24": 2}


def test_fold_errors() -> None:
    with pytest.raises(ValueError, match="not in the data"):
        rolling_origin_folds(DAYS, ["1999_01_01"], "2020_05_22", 1)
    with pytest.raises(ValueError, match="no training day"):
        rolling_origin_folds(DAYS, [], "2020_04_20", 1)


def test_test_windows_never_overlap_train_or_val_windows() -> None:
    """Acceptance: in every fold, test windows are disjoint from (and guarded against)
    train/val windows."""
    w = _windows()
    folds = rolling_origin_folds(DAYS, ["2020_04_21"], "2020_05_22", 1)
    for fold in folds:
        roled = with_roles(w, fold, SPEC.window_minutes + SPEC.horizon_minutes)
        test = roled.filter(pl.col("role") == "test")
        other = roled.filter(pl.col("role").is_in(["train", "val"]))
        overlap = other.filter(
            (pl.col("start_utc") <= test["end_utc"].max())
            & (pl.col("end_utc") >= test["start_utc"].min())
        )
        assert overlap.is_empty()
        assert (
            min_gap_to_test_minutes(w, fold, SPEC.window_minutes + SPEC.horizon_minutes)
            > SPEC.window_minutes + SPEC.horizon_minutes
        )
        assert roled.filter(pl.col("day") == "2020_04_21")["role"].is_null().all()


def test_min_gap_detects_overlap() -> None:
    w = _windows()
    fold = rolling_origin_folds(DAYS, [], "2020_05_22", 1)[0]
    # Fake a leak: relabel one test-day window as belonging to a training day.
    leaked = w.with_columns(
        pl.when(pl.col("end_utc") == _at("2020_05_22", 30))
        .then(pl.lit(fold.train_days[0]))
        .otherwise(pl.col("day"))
        .alias("day")
    )
    assert (
        min_gap_to_test_minutes(leaked, fold, SPEC.window_minutes + SPEC.horizon_minutes) <= 0
    )  # overlap is never purged


def test_each_failure_is_tested_exactly_once() -> None:
    """Acceptance: across folds, every failure on a test day is tested exactly once."""
    index = _index()
    folds = rolling_origin_folds(DAYS, ["2020_04_21"], "2020_05_22", 1)
    tested_by = {f: [fo.index for fo in folds if fo.test_day == day_of(f, index)] for f in ONSETS}
    assert tested_by[ONSETS[0]] == []  # initial block (04_11): validation only, by design
    assert tested_by[ONSETS[1]] == [0]
    assert tested_by[ONSETS[2]] == [1]
    assert day_of(_at("2020_05_22", 0) - timedelta(hours=1), index) is None


def test_irregular_grid_is_rejected() -> None:
    index = _index().filter(pl.col("time_utc") != _at("2020_05_22", 7))
    with pytest.raises(ValueError, match="regular"):
        build_window_index(index, ONSETS, SPEC)


def test_onset_vector_order_does_not_matter() -> None:
    a = build_window_index(_index(), ONSETS, SPEC)
    b = build_window_index(_index(), list(reversed(ONSETS)), SPEC)
    assert a.equals(b)
    assert np.isfinite(a["minutes_to_next_onset"].drop_nulls().to_numpy()).all()


def test_guard_purges_the_end_of_a_contiguous_validation_day() -> None:
    """Bank days are contiguous: validation ends 23:59, test starts 00:00 of the next day."""
    a, b, c = "2021_03_04", "2021_03_05", "2021_03_06"
    rows = (
        [(a, _day_start(a) + timedelta(minutes=i)) for i in range(60)]
        + [(b, _day_start(b) + timedelta(hours=23, minutes=i)) for i in range(60)]
        + [(c, _day_start(c) + timedelta(minutes=i)) for i in range(60)]
    )
    index = pl.DataFrame(rows, schema=["day", "time_utc"], orient="row").with_columns(
        pl.col("time_utc").dt.cast_time_unit("us")
    )
    w = build_window_index(index, [], SPEC)
    fold = rolling_origin_folds([a, b, c], [], c, 1)[0]
    assert fold.val_days == (b,)
    guard = SPEC.window_minutes + SPEC.horizon_minutes  # 10
    assert min_gap_to_test_minutes(w, fold, 0) == 1.0  # without the guard: 23:59 -> 00:00
    roled = with_roles(w, fold, guard)
    purged = roled.filter((pl.col("day") == b) & pl.col("role").is_null())
    assert purged.height == guard  # validation windows ending 23:50 .. 23:59
    assert min_gap_to_test_minutes(w, fold, guard) == guard + 1
    assert roled.filter(pl.col("day") == a)["role"].to_list() == ["train"] * 51  # untouched
