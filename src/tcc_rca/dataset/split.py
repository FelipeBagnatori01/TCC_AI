"""Rolling-origin (expanding window) split by whole days, in chronological order (R2, ADR 0008).

Every usable day from ``first_test_day`` on is the test set of exactly one fold. The fold trains
on all earlier usable days except the last ``val_days`` of them, which form its validation set.
Excluded days never appear in any role.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import polars as pl


@dataclass(frozen=True)
class Fold:
    index: int
    train_days: tuple[str, ...]
    val_days: tuple[str, ...]
    test_day: str

    def role(self, day: str) -> str | None:
        if day == self.test_day:
            return "test"
        if day in self.val_days:
            return "val"
        if day in self.train_days:
            return "train"
        return None


def rolling_origin_folds(
    days: Sequence[str], excluded: Sequence[str], first_test_day: str, val_days: int
) -> list[Fold]:
    unknown = set(excluded) - set(days)
    if unknown:
        raise ValueError(f"excluded days not in the data: {sorted(unknown)}")
    if first_test_day in excluded or first_test_day not in days:
        raise ValueError(f"first test day {first_test_day!r} is excluded or not in the data")
    usable = sorted(d for d in days if d not in set(excluded))
    folds: list[Fold] = []
    for test_day in (d for d in usable if d >= first_test_day):
        prior = [d for d in usable if d < test_day]
        if len(prior) <= val_days:
            raise ValueError(f"fold testing {test_day} has no training day after validation")
        folds.append(Fold(len(folds), tuple(prior[:-val_days]), tuple(prior[-val_days:]), test_day))
    return folds


def with_roles(windows: pl.DataFrame, fold: Fold) -> pl.DataFrame:
    """``windows`` with a ``role`` column (train / val / test / null) for ``fold``."""
    mapping = {d: fold.role(d) for d in windows["day"].unique().to_list()}
    return windows.with_columns(
        pl.col("day").replace_strict(mapping, return_dtype=pl.String).alias("role")
    )


def min_gap_to_test_minutes(windows: pl.DataFrame, fold: Fold) -> float:
    """Smallest time gap between any train/val window span and any test window span (minutes).

    Negative or zero means some train/val window overlaps or touches the test period.
    """
    roled = with_roles(windows, fold)
    test = roled.filter(pl.col("role") == "test")
    other = roled.filter(pl.col("role").is_in(["train", "val"]))
    if test.is_empty() or other.is_empty():
        raise ValueError(f"fold {fold.index} has no test or no train/val windows")
    t_start, t_end = test["start_utc"].min(), test["end_utc"].max()
    gaps = other.select(
        pl.max_horizontal(
            (pl.lit(t_start) - pl.col("end_utc")).dt.total_seconds(),
            (pl.col("start_utc") - pl.lit(t_end)).dt.total_seconds(),
        ).alias("gap")
    )["gap"]
    return float(gaps.min()) / 60.0  # type: ignore[arg-type]


def day_of(instant: datetime, index: pl.DataFrame) -> str | None:
    """The matrix day whose grid ``[first, last]`` minute contains ``instant``."""
    bounds = index.group_by("day").agg(
        pl.col("time_utc").min().alias("lo"), pl.col("time_utc").max().alias("hi")
    )
    hit = bounds.filter((pl.col("lo") <= instant) & (pl.col("hi") >= instant))["day"]
    return None if hit.is_empty() else str(hit[0])


def fold_of_test_day(folds: Sequence[Fold]) -> dict[str, int]:
    """Test day -> fold index; raises if a day were tested twice."""
    out: dict[str, int] = {}
    for fold in folds:
        if fold.test_day in out:
            raise ValueError(f"day {fold.test_day} is tested by two folds")
        out[fold.test_day] = fold.index
    return out
