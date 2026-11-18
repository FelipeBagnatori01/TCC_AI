"""ADR 0020: onset estimators, localizers, reason rules, official-rule hits and the prequential
selection (no label of the test day or later reaches a fold's choice)."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from tcc_rca.config import load_config
from tcc_rca.evaluation.evidence_select import (
    COMPONENT_METHODS,
    Cause,
    FoldInfo,
    Past,
    ScoredQuery,
    best_hits,
    current_package_answer,
    estimated_end_row,
    predict_components,
    prequential_selections,
    select,
)
from tcc_rca.evaluation.openrca import MultiPrediction, Prediction
from tcc_rca.ingestion.cases import Failure, Query
from tcc_rca.models.rca.localize import (
    component_max,
    nsigma_feature_scores,
    personalized_pagerank,
    reciprocal_rank_fusion,
    robust_feature_scores,
    seasonal_feature_scores,
    training_statistics,
)
from tcc_rca.models.rca.onset import (
    bocpd_onsets,
    bocpd_run_length,
    pick_onsets,
    threshold_onsets,
)
from tcc_rca.models.rca.reason import (
    ReasonClassifier,
    component_prior_reason,
    component_type,
    prior_reason,
    reason_features,
    rule_reason,
)

T0 = datetime(2020, 5, 22, 16, 0, tzinfo=UTC)


# ------------------------------------------------------------------------------ config
def test_base_and_experiment_configs_load(repo_root: Path) -> None:
    base = load_config(repo_root / "configs/base.yaml").evidence_select
    assert base.hazard_minutes == 30
    assert base.package_name == "xai"
    ctl = load_config(
        repo_root / "configs/base.yaml",
        repo_root / "configs/experiment/evidence_select_sem_isencao.yaml",
    )
    assert ctl.explain.lstm_name == "lstm_sem_isencao"
    assert ctl.evidence_select.package_name == "xai_sem_isencao"
    assert ctl.series.selection.constant_exempt_categories == ()


# ------------------------------------------------------------------------------ onsets
def test_bocpd_posterior_is_a_distribution_and_finds_a_step() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    x[45:] += 6.0
    r = bocpd_run_length(x, 1 / 30)
    assert r.shape == (60, 61)
    assert np.allclose(r.sum(axis=1), 1.0)
    query = np.zeros(60, dtype=np.bool_)
    query[30:] = True
    assert abs(bocpd_onsets(x, query, 1 / 30)[0] - 45) <= 1
    with pytest.raises(ValueError, match="finite"):
        bocpd_run_length(np.array([0.0, np.nan]), 0.1)


def test_bocpd_without_change_in_the_query_falls_back_to_the_maximum() -> None:
    x = np.zeros(40)
    x[35] = 0.5  # flat signal: no regime change, the maximum inside the query closes the list
    query = np.zeros(40, dtype=np.bool_)
    query[30:] = True
    assert bocpd_onsets(x, query, 1 / 30)[-1] == 35


def test_threshold_onsets_are_run_starts_inside_the_query_then_the_maximum() -> None:
    sig = np.array([9.0, 0.0, 2.0, 3.0, 0.0, 5.0, 4.0, np.nan])
    query = np.array([False, True, True, True, True, True, True, True])
    assert threshold_onsets(sig, query, 2.5) == [3, 5]  # position 0 is outside the query
    assert threshold_onsets(sig, query, 10.0) == [5]  # never crosses: maximum inside


def test_pick_onsets_keeps_the_separation_for_a_second_failure() -> None:
    sig = np.arange(40, dtype=np.float64)
    query = np.ones(40, dtype=np.bool_)
    assert pick_onsets([5, 10, 30], sig, query, 2, 20) == [5, 30]
    assert pick_onsets([5, 10], sig, query, 2, 20) == [5, 39]  # maximum at least 20 away
    small = np.ones(10, dtype=np.bool_)
    assert pick_onsets([3], np.zeros(10), small, 2, 20) == [3, 3]  # nothing far enough


def test_estimated_end_row_clips_to_the_day() -> None:
    day = np.arange(100, 160, dtype=np.int64)
    assert estimated_end_row(130, day, 5, 20) == 135
    assert estimated_end_row(157, day, 5, 20) == 159  # day end
    assert estimated_end_row(103, day, 5, 20) == 119  # first complete window of the day
    with pytest.raises(ValueError, match="shorter"):
        estimated_end_row(100, np.arange(100, 110, dtype=np.int64), 5, 20)


# -------------------------------------------------------------------------- localizers
def test_robust_and_nsigma_scores_hand_computed() -> None:
    ref = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0], [5.0, 5.0]])
    post = np.array([[3.0, 5.0], [9.0, 7.0]])
    fb = np.zeros(2)
    # column 0: median 3, IQR 2 -> max |9 - 3| / 2 = 3; column 1: IQR 0 -> scale 1 -> 2
    assert robust_feature_scores(ref, post, fb, fb).tolist() == [3.0, 2.0]
    # mean 3, std sqrt(2) -> 6 / sqrt(2); column 1: std 0 -> 1 -> 2
    np.testing.assert_allclose(nsigma_feature_scores(ref, post, fb, fb), [6 / np.sqrt(2), 2.0])
    # empty reference -> training fallback statistics
    out = robust_feature_scores(ref[:0], post, np.array([1.0, 5.0]), np.array([4.0, 2.0]))
    assert out.tolist() == [2.0, 1.0]


def test_seasonal_scores_and_missing_values() -> None:
    post = np.array([[10.0, np.nan], [12.0, 1.0]])
    prev = np.array([[10.0, 0.0], [np.nan, np.nan]])
    assert seasonal_feature_scores(post, prev, np.array([2.0, 1.0])).tolist() == [0.0, 0.0]
    stats = training_statistics(np.array([[1.0, np.nan], [3.0, np.nan]]))
    assert stats["median"].tolist() == [2.0, 0.0]
    assert stats["iqr"].tolist() == [1.0, 1.0]


def test_component_max_and_fusion() -> None:
    cols = {"a": np.array([0, 1]), "b": np.array([2])}
    assert component_max(np.array([1.0, 4.0, 2.0]), cols, ["a", "b", "c"]) == {
        "a": 4.0,
        "b": 2.0,
        "c": 0.0,
    }
    fused = reciprocal_rank_fusion([{"a": 2.0, "b": 1.0}, {"a": 0.0, "b": 1.0}], ["a", "b"], 60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_personalized_pagerank_two_nodes_hand_computed() -> None:
    # a -> b (weight 1), b -> a backwards (0.2, normalized to 1); restart to a
    r = personalized_pagerank([("a", "b")], {"a": 1.0}, damping=0.85, backward_weight=0.2)
    assert r["a"] == pytest.approx(1 / 1.85)
    assert r["b"] == pytest.approx(0.85 / 1.85)
    uniform = personalized_pagerank([("a", "b")], {}, 0.85, 0.0)
    assert sum(uniform.values()) == pytest.approx(1.0)


# ------------------------------------------------------------------------------ reasons
MAP = {"CPU fault": ("CPU",), "network delay": ("rede", "latência"), "network loss": ("rede",)}
REASONS = ("CPU fault", "db close", "network delay", "network loss")


def test_reason_rules() -> None:
    past = [("os_1", "network loss"), ("os_1", "network loss"), ("db_1", "db close")]
    assert prior_reason([r for _, r in past], REASONS) == "network loss"
    assert prior_reason([], REASONS) == "CPU fault"  # alphabetical without history
    assert component_prior_reason("db_1", past, REASONS) == "db close"
    assert component_prior_reason("os_9", past, REASONS) == "network loss"
    # "rede" maps to delay and loss; the component's past picks loss
    assert rule_reason("rede", "os_1", past, MAP, REASONS) == "network loss"
    assert rule_reason("rede", "os_9", [], MAP, REASONS) == "network delay"  # alphabetical
    assert rule_reason("CPU", "db_1", past, MAP, REASONS) == "CPU fault"
    assert rule_reason("outro", "db_1", past, MAP, REASONS) == "db close"  # unmapped -> R1
    assert component_type("os_001") == "os"
    assert component_type("Tomcat01") == "Tomcat"


def test_reason_features_and_classifier() -> None:
    feats = reason_features(
        np.array([3.0, 1.0, 7.0]), np.array([0, 2]), ["CPU", "rede", "CPU"], ["CPU", "rede"]
    )
    np.testing.assert_allclose(feats, [np.log1p(7.0), 0.0])
    single = ReasonClassifier(["os"], 1.0).fit([(np.zeros(2), "os", "CPU fault")], "x")
    assert single.predict(np.ones(2), "os") == "CPU fault"  # one class: constant
    empty = ReasonClassifier(["os"], 1.0).fit([], "db close")
    assert empty.predict(np.ones(2), "os") == "db close"
    rows = [(np.array([5.0, 0.0]), "os", "CPU fault")] * 4 + [
        (np.array([0.0, 5.0]), "os", "network loss")
    ] * 4
    model = ReasonClassifier(["os", "db"], 1.0).fit(rows, "x")
    assert model.predict(np.array([6.0, 0.0]), "os") == "CPU fault"
    assert model.predict(np.array([0.0, 6.0]), "os") == "network loss"


# ------------------------------------------------------------- official-rule hits
def test_best_hits_uses_the_best_assignment() -> None:
    gold = (Failure("l", "CPU fault", "a", T0), Failure("l", "network loss", "b", T0))
    preds = [
        {"occurrence_datetime": T0 + timedelta(seconds=90), "component": "b", "reason": "x"},
        {"occurrence_datetime": T0 + timedelta(seconds=30), "component": "a", "reason": "x"},
    ]
    assert best_hits(preds, gold, ["component"]) == 2
    assert best_hits(preds, gold, ["occurrence_datetime"]) == 1  # 90 s is too far
    assert best_hits(preds, gold, ["component", "occurrence_datetime"]) == 3
    with pytest.raises(ValueError, match="one predicted cause"):
        best_hits(preds[:1], gold, ["component"])


def test_multi_prediction_is_scored_by_the_official_evaluator(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "external/OpenRCA"))
    try:
        from main.evaluate import evaluate
    finally:
        sys.path.pop(0)
    pred = MultiPrediction(
        3,
        (
            Prediction(3, T0, "Mysql01", "network latency"),
            Prediction(3, T0 + timedelta(minutes=20), "Tomcat01", "high CPU usage"),
        ),
    )
    scoring = (
        "The 1-th root cause occurrence time is within 1 minutes (i.e., <=1min) of "
        "2020-05-23 00:20:00\n"
        "The 1-th predicted root cause component is Tomcat01\n"
        "The 2-th root cause occurrence time is within 1 minutes (i.e., <=1min) of "
        "2020-05-23 00:00:30\n"
        "The 2-th predicted root cause component is Mysql01\n"
    )
    _, failed, score = evaluate(pred.as_official_json(), scoring)
    assert score == 1.0
    assert not failed
    one = MultiPrediction(3, (Prediction(3, None, "a", None),))
    assert one.as_official_json() == Prediction(3, None, "a", None).as_official_json()
    with pytest.raises(ValueError, match="same query row"):
        MultiPrediction(3, (Prediction(4, None, "a", None),)).as_official_json()


# ------------------------------------------------------------------ prequential selection
INFO = FoldInfo(0, {"a": np.array([0]), "b": np.array([1])}, ["CPU", "rede"])


def _query(row: int, day: str, gold_comp: str, gold_reason: str, good: str) -> ScoredQuery:
    """One-failure query whose onset is hit only by ``good`` onset method, and whose gold
    component is ranked first only by C3."""
    onset = T0 + timedelta(days=row, minutes=10)
    other = "b" if gold_comp == "a" else "a"
    scores = {c: {other: 2.0, gold_comp: 1.0} for c in COMPONENT_METHODS if c != "C0"}
    scores["C3"] = {gold_comp: 2.0, other: 1.0}
    c4 = np.array([1.0, 0.0]) if gold_comp == "a" else np.array([0.0, 1.0])
    onsets = {
        t: [
            Cause(
                0,
                onset if t == good else onset + timedelta(minutes=5),
                scores,
                c4,
            )
        ]
        for t in ("T0", "T1", "T2", "T3", "T4")
    }
    q = Query(row, "task_7", "x", onset, onset + timedelta(minutes=30))
    gold = (Failure("l", gold_reason, gold_comp, onset),)
    return ScoredQuery(
        q, day, INFO, 1, ("occurrence_datetime",), {"a": 0.0, "b": 0.0}, onsets, gold
    )


def _past(labels: list[tuple[str, str, str]]) -> Past:
    return Past(
        lambda day: [(c, r) for d, c, r in labels if d < day],
        REASONS,
        MAP,
        ("CPU", "rede"),
        1.0,
    )


def test_select_picks_the_best_method_of_each_stage() -> None:
    qs = [_query(i, f"d{i}", "a", "CPU fault", "T2") for i in range(3)]
    sel = select(qs, _past([(f"d{i}", "a", "CPU fault") for i in range(3)]))
    assert (sel.onset, sel.component) == ("T2", "C3")
    assert sel.accuracy["T2"] == 1.0
    assert sel.accuracy["T1"] == 0.0
    assert predict_components(qs[0], "T2", "C3") == ["a"]
    # R2 reads the CPU category of "a"; R0 has no history for d0 -> alphabetical "CPU fault"
    assert sel.reason == "R0"
    assert not sel.reason_beats_prior


def test_selection_never_reads_labels_of_the_test_day_or_later() -> None:
    days = ["d0", "d1", "d2", "d3"]
    base = [_query(i, d, "a", "CPU fault", "T1") for i, d in enumerate(days)]
    labels = [(d, "a", "CPU fault") for d in days]
    test_days = {0: "d2", 1: "d3"}
    ref = prequential_selections(base, test_days, _past(labels))
    # rewrite the test day and later: another onset method, component and reason are "right"
    changed = base[:2] + [
        _query(i, d, "b", "network loss", "T4") for i, d in [(2, "d2"), (3, "d3")]
    ]
    changed_labels = labels[:2] + [(d, "b", "network loss") for d in ("d2", "d3")]
    got = prequential_selections(changed, test_days, _past(changed_labels))
    assert got[0] == ref[0]  # fold testing d2 only sees d0 and d1
    assert got[1] != ref[1]  # the fold testing d3 does see the changed d2 (sanity)


def test_current_package_answer_follows_the_fixed_rule() -> None:
    sq = _query(0, "d5", "a", "CPU fault", "T1")
    package = {
        "kpis": [{"peak_time_local": "2020-05-23 00:12:00"}],
        "components": [{"component": "zz"}, {"component": "b"}],
        "suggested_category": "rede",
    }
    past = _past(
        [("d1", "a", "network loss"), ("d2", "a", "network delay"), ("d3", "a", "network loss")]
    )
    out = current_package_answer(package, sq, ["a", "b"], past)
    assert out == [
        {
            "occurrence_datetime": datetime(2020, 5, 22, 16, 12, tzinfo=UTC),
            "component": "b",  # "zz" is not a candidate
            "reason": "network loss",  # most frequent past reason among those of "rede"
        }
    ]
