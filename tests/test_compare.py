"""Paired comparisons (E5/E8): hand-computed differences and the paired design."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from tcc_rca.evaluation.bootstrap import Estimate
from tcc_rca.evaluation.compare import (
    paired_rank_difference,
    paired_window_difference,
    verdict,
)


def _frame(scores: list[float], labels: list[int]) -> pl.DataFrame:
    t0 = datetime(2021, 3, 6, tzinfo=UTC)
    return pl.DataFrame(
        {
            "row": list(range(len(scores))),
            "day": ["d"] * len(scores),
            "end_utc": [t0 + timedelta(minutes=37 * i) for i in range(len(scores))],
            "label": labels,
            "score": scores,
        }
    )


def test_identical_methods_differ_by_exactly_zero() -> None:
    a = _frame([0.9, 0.1, 0.8, 0.3, 0.2, 0.7], [1, 0, 1, 0, 0, 1])
    e = paired_window_difference(a, a, "auc_pr", 30, 200, seed=0)
    assert (e.value, e.ci_low, e.ci_high) == (0.0, 0.0, 0.0)
    assert verdict(e) == "empate"


def test_window_difference_hand_computed() -> None:
    labels = [1, 0, 1, 0]
    perfect = _frame([0.9, 0.1, 0.8, 0.2], labels)  # AUC-ROC 1
    inverted = _frame([0.1, 0.9, 0.2, 0.8], labels)  # AUC-ROC 0
    e = paired_window_difference(perfect, inverted, "auc_roc", 30, 300, seed=1)
    assert e.value == 1.0
    assert verdict(e) == "superior"


def test_windows_must_match() -> None:
    a = _frame([0.9, 0.1], [1, 0])
    with pytest.raises(ValueError, match="same test windows"):
        paired_window_difference(a, a.head(1), "auc_pr", 30, 10, seed=0)


def test_rank_difference_hand_computed() -> None:
    a = np.array([1, 1, 2, 5])
    b = np.array([3, 1, 4, 5])
    e = paired_rank_difference(a, b, "hr@1", 300, seed=2)
    assert e.value == pytest.approx(0.5 - 0.25)
    mrr = paired_rank_difference(a, b, "mrr", 50, seed=2)
    assert mrr.value == pytest.approx((1 + 1 + 0.5 + 0.2) / 4 - (1 / 3 + 1 + 0.25 + 0.2) / 4)
    with pytest.raises(ValueError, match="same failures"):
        paired_rank_difference(a, b[:3], "hr@1", 10, seed=0)


def test_verdict() -> None:
    assert verdict(Estimate(0.1, 0.01, 0.2, 10)) == "superior"
    assert verdict(Estimate(-0.1, -0.2, -0.01, 10)) == "inferior"
    assert verdict(Estimate(0.1, -0.01, 0.2, 10)) == "empate"
