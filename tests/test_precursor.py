from __future__ import annotations

import numpy as np
import pytest

from tcc_rca.dataset.precursor import bootstrap_auroc_ci, lead_samples, minute_scores
from tcc_rca.evaluation.metrics import auroc


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


def test_local_abs_z_is_causal_hand_computed_and_floored() -> None:
    import polars as pl

    from tcc_rca.dataset.precursor import local_abs_z

    base = [10.0, 12.0] * 10  # past median 11, past std ~1.026
    frame = pl.DataFrame({"day": ["d"] * 22, "a::x": [*base, 20.0, 11.0], "b::c": [5.0] * 22})
    z = local_abs_z(frame, ["a::x", "b::c"], window=20, cap=50.0)
    past = np.array(base)
    assert np.isnan(z[:10, 0]).all()  # fewer than 10 past samples: undefined
    assert z[20, 0] == pytest.approx((20.0 - 11.0) / past.std(ddof=1))
    # constant series: zero spread is floored at 0.1% of (|median| + 1), never a division by 0
    assert z[21, 1] == 0.0
    # causality: changing the last value never changes any earlier |z|
    later = frame.with_columns(pl.Series("a::x", [*base, 20.0, 999.0]))
    z2 = local_abs_z(later, ["a::x", "b::c"], window=20, cap=50.0)
    np.testing.assert_array_equal(z[:21], z2[:21])
    assert z2[21, 0] == 50.0  # capped
