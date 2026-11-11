"""E4 baselines: hand-computed features, leakage guards, determinism and official scoring."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tcc_rca.config import LightGBMConfig
from tcc_rca.evaluation.openrca import Prediction, score_with_official_script
from tcc_rca.ingestion.cases import Failure, Query
from tcc_rca.models.baselines.constant import constant_predictions, most_frequent
from tcc_rca.models.baselines.detection import (
    choose_threshold,
    isolation_forest_scores,
    lightgbm_scores,
)
from tcc_rca.models.baselines.features import (
    ZScore,
    window_max_abs,
    window_rows,
    window_statistics,
)
from tcc_rca.models.baselines.localization import component_columns, zscore_component_scores

LGBM = LightGBMConfig(
    n_estimators=20, learning_rate=0.1, num_leaves=4, min_child_samples=2, colsample_bytree=1.0
)


def test_zscore_fits_on_training_rows_only() -> None:
    frame = pl.DataFrame({"a::x": [1.0, 3.0, 1000.0, -50.0], "b::c": [5.0, 5.0, 9.0, 9.0]})
    train = np.array([True, True, False, False])
    zs = ZScore.fit(frame, ["a::x", "b::c"], train)
    assert zs.mean.tolist() == [2.0, 5.0]
    assert zs.std.tolist() == [1.0, 1.0]  # std of [1, 3] = 1; constant -> fallback 1.0
    poisoned = frame.with_columns(pl.Series("a::x", [1.0, 3.0, 1e9, 1e9]))
    again = ZScore.fit(poisoned, ["a::x", "b::c"], train)
    assert again.mean.tolist() == zs.mean.tolist()
    assert again.std.tolist() == zs.std.tolist()
    assert zs.transform(frame)[2].tolist() == [998.0, 4.0]


def test_window_helpers_hand_computed() -> None:
    assert window_rows(np.array([2, 4]), 3).tolist() == [[0, 1, 2], [2, 3, 4]]
    z = np.array([[0.5, -1.0], [2.0, np.nan], [0.0, -4.0], [1.0, 1.0], [0.0, 0.0]])
    # per-minute max |z|: 1, 2, 4, 1, 0 -> windows of 2 ending at rows 1, 3, 4: 2, 4, 1
    assert window_max_abs(z, np.array([1, 3, 4]), 2).tolist() == [2.0, 4.0, 1.0]
    x = np.array([[1.0], [3.0], [np.nan], [5.0]])
    stats = window_statistics(x, np.array([1, 3]), 2, chunk=1)
    # columns: mean, std, min, max, last; window rows [0,1] and [2,3] (NaN ignored)
    assert stats.tolist() == [[2.0, 1.0, 1.0, 3.0, 3.0], [5.0, 0.0, 5.0, 5.0, 5.0]]


def test_isolation_forest_is_seeded_and_ranks_outlier_highest() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 3)).astype(np.float32)
    probe = np.vstack([train[:5], np.full((1, 3), 8.0, dtype=np.float32)])
    s1 = isolation_forest_scores(train, probe, 50, seed=1)
    assert s1.tolist() == isolation_forest_scores(train, probe, 50, seed=1).tolist()
    assert int(np.argmax(s1)) == 5


def test_lightgbm_is_seeded_learns_and_needs_both_classes() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 4)).astype(np.float32)
    y = (x[:, 0] > 0.8).astype(np.int64)
    p1 = lightgbm_scores(x, y, x, LGBM, seed=3)
    assert p1.tolist() == lightgbm_scores(x, y, x, LGBM, seed=3).tolist()
    assert p1[y == 1].mean() > p1[y == 0].mean()
    with pytest.raises(ValueError, match="both"):
        lightgbm_scores(x, np.zeros(120, dtype=np.int64), x, LGBM, seed=3)


def test_threshold_uses_validation_f1_or_normal_quantile() -> None:
    thr, rule = choose_threshold(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1]), 0.99)
    assert (thr, rule) == (0.7, "best_f1_val")
    scores = np.arange(101, dtype=np.float64)
    thr, rule = choose_threshold(np.zeros(101, dtype=np.int64), scores, 0.99)
    assert thr == pytest.approx(99.0)
    assert rule.startswith("q0.99")


def test_localization_scores_by_component() -> None:
    feats = ["os_1::cpu", "os_1::mem", "db_1::conn"]
    cols = component_columns(feats)
    assert {k: v.tolist() for k, v in cols.items()} == {"os_1": [0, 1], "db_1": [2]}
    z = np.array([[0.0, 1.0, -2.0], [5.0, np.nan, 0.5], [9.0, 9.0, 9.0]])
    scores = zscore_component_scores(z, np.array([0, 1]), cols, ["db_1", "os_1", "docker_9"])
    assert scores == {"db_1": 2.0, "os_1": 5.0, "docker_9": 0.0}  # row 2 is outside the window


def _failure(component: str, reason: str) -> Failure:
    return Failure("node", reason, component, datetime(2020, 5, 22, tzinfo=UTC))


def test_constant_baseline_uses_training_modes_and_window_start() -> None:
    assert most_frequent(["b", "a", "b", "a", "c"]) == "a"  # tie broken alphabetically
    with pytest.raises(ValueError, match="no items"):
        most_frequent([])
    train = [
        _failure("os_1", "CPU fault"),
        _failure("os_1", "network loss"),
        _failure("db_1", "CPU fault"),
    ]
    start = datetime(2020, 5, 22, 16, 0, tzinfo=UTC)
    q = Query(7, "task_7", "text", start, start)
    [pred] = constant_predictions(train, [q])
    assert (pred.query_row, pred.occurrence_utc, pred.component, pred.reason) == (
        7,
        start,
        "os_1",
        "CPU fault",
    )
    assert pred.as_official_json() == (
        '{"root cause occurrence datetime": "2020-05-23 00:00:00", '
        '"root cause component": "os_1", "root cause reason": "CPU fault"}'
    )


QUERY_CSV = (
    "task_index,instruction,scoring_points\n"
    'task_1,"q0","The only root cause occurrence time is within 1 minutes (i.e., <=1min) of '
    '2020-05-23 00:05:00\n"\n'
    'task_3,"q1","The only predicted root cause component is os_1\n"\n'
    'task_7,"q2","The only root cause occurrence time is within 1 minutes (i.e., <=1min) of '
    "2020-05-23 01:00:00\nThe only predicted root cause component is db_1\n"
    'The only predicted root cause reason is db close\n"\n'
)


def test_official_script_scores_a_query_subset(tmp_path: Path, repo_root: Path) -> None:
    """Runs the real OpenRCA evaluator (submodule) on rows 0 and 2 only."""
    query_csv = tmp_path / "query.csv"
    query_csv.write_text(QUERY_CSV, encoding="utf-8")
    t0 = datetime(2020, 5, 22, 16, 5, 30, tzinfo=UTC)  # 00:05:30 UTC+8: within 1 min
    t2 = datetime(2020, 5, 22, 17, 0, tzinfo=UTC)
    preds = [
        Prediction(2, t2, "db_1", "CPU fault"),  # time + component right, reason wrong: 2/3
        Prediction(0, t0, "os_9", "x"),  # only time is asked: correct
    ]
    out = score_with_official_script(
        preds, query_csv, repo_root / "external/OpenRCA", tmp_path / "eval"
    )
    assert out["query_row"].to_list() == [0, 2]
    assert out["task_index"].to_list() == ["task_1", "task_7"]
    assert out["score"].to_list() == [1.0, 0.67]
    assert out["strict"].to_list() == [True, False]
    subset = (tmp_path / "eval" / "query_subset.csv").read_text(encoding="utf-8")
    assert "os_1" not in subset  # row 1 was not requested


def test_official_wrapper_rejects_duplicates(tmp_path: Path, repo_root: Path) -> None:
    query_csv = tmp_path / "query.csv"
    query_csv.write_text(QUERY_CSV, encoding="utf-8")
    p = Prediction(0, None, "os_1", None)
    with pytest.raises(ValueError, match="duplicate"):
        score_with_official_script([p, p], query_csv, repo_root / "external/OpenRCA", tmp_path)


def test_local_component_scores_average_top3_window_max() -> None:
    from tcc_rca.models.baselines.localization import local_component_scores

    feats = ["os_1::a", "os_1::b", "os_1::c", "os_1::d", "db_1::a"]
    cols = component_columns(feats)
    local = np.array(
        [
            [50.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 9.0, np.nan, 3.0, 50.0],
            [99.0, 99.0, 99.0, 99.0, 99.0],  # outside the window
        ]
    )
    scores = local_component_scores(local, np.array([0, 1]), cols, ["os_1", "db_1", "x_9"])
    # os_1 window maxima: a=50, b=9, c=2, d=3 -> top3 mean (50 + 9 + 3) / 3; db_1: 50
    assert scores == {"os_1": pytest.approx(62 / 3), "db_1": 50.0, "x_9": 0.0}


def test_prior_component_scores_count_training_root_causes() -> None:
    from tcc_rca.evaluation.metrics import pessimistic_rank
    from tcc_rca.models.baselines.localization import prior_component_scores

    train = [_failure("os_1", "r"), _failure("os_1", "r"), _failure("db_1", "r")]
    scores = prior_component_scores(train, ["db_1", "os_1", "x_9"])
    assert scores == {"db_1": 1.0, "os_1": 2.0, "x_9": 0.0}
    assert pessimistic_rank("os_1", scores) == 1
    assert pessimistic_rank("x_9", scores) == 3  # never seen: last
