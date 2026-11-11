from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from tcc_rca.series.transforms import FeatureSelector, RobustScaler

FEATURES = ["a::x", "b::const", "c::gappy", "d::near", "e::state"]


def _frame() -> pl.DataFrame:
    n = 10
    return pl.DataFrame(
        {
            "a::x": [float(i) for i in range(n)],
            "b::const": [1.0] * n,
            "c::gappy": [1.0, None, None, None, None, None, 2.0, 3.0, 4.0, 5.0],
            "d::near": [0.0] * 9 + [1.0],
            "e::state": [1.0] * n,
        }
    )


TRAIN = np.array([True] * 6 + [False] * 4)


def test_selector_reasons() -> None:
    sel = FeatureSelector(max_missing_frac=0.5, near_constant_mode_share=0.99).fit(
        _frame(), FEATURES, TRAIN
    )
    assert sel.kept == ["a::x"]
    assert sel.dropped["b::const"] == "constante no treino"
    assert sel.dropped["c::gappy"].startswith("ausente em 83.3%")
    assert sel.dropped["d::near"] == "constante no treino"  # the 1.0 is outside training rows
    assert list(sel.transform(_frame()).columns) == ["a::x"]


def test_near_constant_threshold() -> None:
    frame = pl.DataFrame({"n::k": [0.0] * 999 + [1.0]})
    train = np.ones(1000, dtype=np.bool_)
    assert FeatureSelector(0.5, 0.999).fit(frame, ["n::k"], train).dropped
    assert not FeatureSelector(0.5, 0.9995).fit(frame, ["n::k"], train).dropped


def test_exempt_features_survive_constancy_but_not_missingness() -> None:
    sel = FeatureSelector(0.5, 0.99, exempt=frozenset({"e::state", "c::gappy"})).fit(
        _frame(), FEATURES, TRAIN
    )
    assert "e::state" in sel.kept
    assert "c::gappy" in sel.dropped
    assert sel.rescued == {"e::state": "constante no treino"}


def test_rescued_features_reproduce_the_control_selection() -> None:
    """Sensitivity analysis: kept minus rescued must equal the selection without exemption."""
    frame = _frame()
    control = FeatureSelector(0.5, 0.99).fit(frame, FEATURES, TRAIN)
    exempted = FeatureSelector(0.5, 0.99, exempt=frozenset({"a::x", "b::const", "e::state"})).fit(
        frame, FEATURES, TRAIN
    )
    assert "a::x" not in exempted.rescued  # exempt but varying: kept on its own merits
    assert [f for f in exempted.kept if f not in exempted.rescued] == control.kept
    assert set(exempted.rescued) | set(exempted.dropped) == set(control.dropped)


@pytest.mark.parametrize("cls", [FeatureSelector, RobustScaler])
def test_fit_ignores_validation_and_test_rows(cls) -> None:  # type: ignore[no-untyped-def]
    """Leakage detector: rewriting every non-training row must not change the fitted state."""
    frame = _frame()
    poisoned = frame.with_columns(
        [
            pl.when(pl.Series(~TRAIN)).then(pl.lit(1e9)).otherwise(pl.col(f)).alias(f)
            for f in FEATURES
        ]
    )
    feats = ["a::x"]
    make = (lambda: FeatureSelector(0.5, 0.99)) if cls is FeatureSelector else RobustScaler
    clean, dirty = make().fit(frame, feats, TRAIN), make().fit(poisoned, feats, TRAIN)
    assert clean == dirty
    # sanity: the same poisoning inside the training rows DOES change a scaler fit
    if cls is RobustScaler:
        everything = np.ones(frame.height, dtype=np.bool_)
        assert RobustScaler().fit(poisoned, feats, everything) != clean


def test_robust_scaler_values_and_inverse() -> None:
    frame = pl.DataFrame({"a::x": [1.0, 2.0, 3.0, 4.0, 5.0], "b::c": [7.0] * 5})
    train = np.ones(5, dtype=np.bool_)
    scaler = RobustScaler().fit(frame, ["a::x", "b::c"], train)
    assert scaler.center["a::x"] == 3.0
    assert scaler.scale["a::x"] == 2.0  # IQR of 1..5 = 4 - 2
    assert scaler.fallback == {"b::c": "unit"}
    out = scaler.transform(frame)
    assert out["a::x"].to_list() == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert scaler.inverse_transform(out)["a::x"].to_list() == frame["a::x"].to_list()


def test_std_fallback_when_iqr_is_zero() -> None:
    frame = pl.DataFrame({"a::x": [0.0] * 8 + [10.0, 10.0]})
    scaler = RobustScaler().fit(frame, ["a::x"], np.ones(10, dtype=np.bool_))
    assert scaler.fallback == {"a::x": "std"}
    assert scaler.scale["a::x"] == pytest.approx(4.0)


def test_mask_must_be_valid() -> None:
    with pytest.raises(ValueError, match="boolean mask"):
        RobustScaler().fit(_frame(), ["a::x"], np.ones(3, dtype=np.bool_))
    with pytest.raises(ValueError, match="no rows"):
        RobustScaler().fit(_frame(), ["a::x"], np.zeros(10, dtype=np.bool_))
