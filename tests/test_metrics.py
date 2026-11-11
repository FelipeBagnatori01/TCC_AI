"""Hand-computed values for every evaluation metric (R8, E8 acceptance)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tcc_rca.evaluation.bootstrap import cluster_bootstrap, clusters_from_labels
from tcc_rca.evaluation.metrics import (
    auc_pr,
    auc_roc,
    best_f1_threshold,
    event_detection,
    false_alarms_per_hour,
    hits_at_k,
    mcnemar_exact,
    mean_reciprocal_rank,
    pessimistic_rank,
    precision_recall_f1,
)

Y = np.array([1, 0, 1, 0, 0])
S = np.array([0.9, 0.8, 0.7, 0.3, 0.1])


def test_auc_roc_hand_computed() -> None:
    # positives 0.9, 0.7 vs negatives 0.8, 0.3, 0.1: 0.9 wins 3, 0.7 wins 2 -> 5/6
    assert auc_roc(Y, S) == pytest.approx(5 / 6)


def test_auc_pr_hand_computed() -> None:
    # ranked: 1(P) 0(N) 1(P) ...: precision at each recall step: 1/1 at R=0.5, 2/3 at R=1.0
    # AP = 0.5 * 1 + 0.5 * 2/3 = 5/6
    assert auc_pr(Y, S) == pytest.approx(5 / 6)
    with pytest.raises(ValueError, match="positive"):
        auc_pr(np.zeros(3, dtype=int), np.ones(3))


def test_precision_recall_f1_hand_computed() -> None:
    r = precision_recall_f1(Y, S >= 0.75)  # predicted: idx 0, 1 -> TP 1, FP 1, FN 1
    assert (r.tp, r.fp, r.fn) == (1, 1, 1)
    assert (r.precision, r.recall, r.f1) == (0.5, 0.5, 0.5)
    none = precision_recall_f1(Y, np.zeros(5, dtype=bool))
    assert (none.precision, none.recall, none.f1) == (0.0, 0.0, 0.0)


def test_best_f1_threshold_hand_computed() -> None:
    # t=0.9: P1 R.5 F.667 | t=0.8: P.5 R.5 F.5 | t=0.7: P.667 R1 F.8 | t=0.3: F.667 | t=0.1: F.571
    assert best_f1_threshold(Y, S) == 0.7
    # tie between thresholds -> the highest one is kept
    assert best_f1_threshold(np.array([1, 1]), np.array([0.5, 0.5])) == 0.5
    with pytest.raises(ValueError, match="positive"):
        best_f1_threshold(np.zeros(2, dtype=int), np.array([0.1, 0.2]))


def test_event_detection_and_delay() -> None:
    alarms = [95.0, 103.0, 140.0]
    assert event_detection(100.0, alarms, 5, 30) == event_detection(100.0, [95.0], 5, 30)
    first = event_detection(100.0, alarms, 5, 30)
    assert first.detected
    assert first.delay_minutes == -5.0  # anticipation is negative
    late = event_detection(100.0, alarms, 2, 30)
    assert late.delay_minutes == 3.0
    missed = event_detection(100.0, [131.0, 50.0], 5, 30)
    assert not missed.detected
    assert missed.delay_minutes is None


def test_false_alarms_per_hour() -> None:
    assert false_alarms_per_hour(np.array([True] * 3 + [False] * 117)) == 1.5  # 3 in 2 hours
    with pytest.raises(ValueError, match="no normal windows"):
        false_alarms_per_hour(np.array([], dtype=bool))


def test_pessimistic_rank_and_ranking_metrics() -> None:
    scores = {"a": 3.0, "b": 2.0, "c": 2.0, "d": 1.0}
    assert pessimistic_rank("a", scores) == 1
    assert pessimistic_rank("b", scores) == 3  # tied with c: c counts as ahead
    assert pessimistic_rank("d", scores) == 4
    assert pessimistic_rank("zzz", scores) == 5  # absent candidate ranks last
    assert pessimistic_rank("a", {"a": math.nan, "b": 1.0}) == 2
    ranks = [1, 3, 2, 10]
    assert hits_at_k(ranks, 1) == 0.25
    assert hits_at_k(ranks, 3) == 0.75
    assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 1 / 3 + 1 / 2 + 1 / 10) / 4)


def test_mcnemar_exact_hand_computed() -> None:
    a = np.array([1, 1, 1, 1, 0, 1, 0, 0], dtype=bool)
    b = np.array([0, 0, 0, 0, 0, 1, 1, 0], dtype=bool)
    # discordant: a-only 4, b-only 1 -> n=5, k=1: 2 * (C(5,0)+C(5,1))/32 = 12/32
    assert mcnemar_exact(a, b) == pytest.approx(12 / 32)
    assert mcnemar_exact(a, a) == 1.0
    with pytest.raises(ValueError, match="same shape"):
        mcnemar_exact(a, b[:3])


def test_cluster_bootstrap_resamples_whole_clusters_and_is_seeded() -> None:
    values = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    clusters = clusters_from_labels(["x", "x", "y", "y", "y"])
    assert [c.tolist() for c in clusters] == [[0, 1], [2, 3, 4]]
    est = cluster_bootstrap(lambda i: float(values[i].mean()), clusters, 500, seed=3)
    assert est.value == pytest.approx(0.4)
    # any resample is {x,x}: 1.0, {y,y}: 0.0 or {x,y}: 0.4 -> CI within [0, 1]
    assert 0.0 <= est.ci_low <= est.value <= est.ci_high <= 1.0
    assert est == cluster_bootstrap(lambda i: float(values[i].mean()), clusters, 500, seed=3)


def test_cluster_bootstrap_skips_undefined_resamples() -> None:
    y = np.array([1, 0])
    s = np.array([0.9, 0.1])
    clusters = clusters_from_labels(["p", "n"])
    est = cluster_bootstrap(lambda i: auc_roc(y[i], s[i]), clusters, 200, seed=0)
    assert est.value == 1.0
    assert 0 < est.n_valid_resamples < 200  # resamples with a single class are skipped


def test_chance_event_recall_hand_computed() -> None:
    from tcc_rca.evaluation.metrics import chance_event_recall

    assert chance_event_recall(0.5, 2) == 0.75  # 1 - 0.5^2
    assert chance_event_recall(0.0, 36) == 0.0
    assert chance_event_recall(1.0, 1) == 1.0
    with pytest.raises(ValueError, match="probability"):
        chance_event_recall(1.5, 3)
