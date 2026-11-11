from __future__ import annotations

import numpy as np
import pytest

from tcc_rca.dataset.precursor import auroc, bootstrap_auroc_ci, lead_samples, minute_scores


def test_auroc_hand_computed() -> None:
    # pairs (p, n): 3>1, 3>2, 3>2.5 | 2==2 (0.5), 2>1, 2<2.5 -> (3 + 1.5) / 6 = 0.75
    assert auroc([3.0, 2.0], [1.0, 2.0, 2.5]) == pytest.approx(4.5 / 6)
    assert auroc([5.0], [1.0, 2.0]) == 1.0
    assert auroc([0.0], [1.0, 2.0]) == 0.0
    assert auroc([1.0, 1.0], [1.0]) == 0.5


def test_auroc_rejects_empty_and_nan() -> None:
    with pytest.raises(ValueError, match="at least one"):
        auroc([], [1.0])
    with pytest.raises(ValueError, match="NaN"):
        auroc([np.nan], [1.0])


def test_bootstrap_ci_brackets_estimate_and_is_seeded() -> None:
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1.0, 1.0, 40), rng.normal(0.0, 1.0, 300)
    lo, hi = bootstrap_auroc_ci(pos, neg, 500, seed=7)
    assert lo < auroc(pos, neg) < hi
    assert (lo, hi) == bootstrap_auroc_ci(pos, neg, 500, seed=7)


def test_minute_scores_top_k_and_missing() -> None:
    z = np.array([[1.0, -4.0, 2.0, np.nan], [0.0, 0.0, 0.0, 0.0]])
    assert minute_scores(z, 2).tolist() == [3.0, 0.0]  # mean(|-4|, 2)
    assert minute_scores(z, 10).tolist() == [7.0 / 4, 0.0]  # k capped at n features, NaN = 0


def test_lead_samples_picks_minutes_before_onset() -> None:
    minutes = np.arange("2020-05-22T00:00", "2020-05-22T00:10", dtype="datetime64[m]")
    scores = np.arange(10, dtype=np.float64)
    onsets = np.array(["2020-05-22T00:05", "2020-05-22T00:01"], dtype="datetime64[m]")
    assert lead_samples(scores, minutes, onsets, 2).tolist() == [3.0]  # 00:-01 not in grid
    assert lead_samples(scores, minutes, onsets, -1).tolist() == [6.0, 2.0]
