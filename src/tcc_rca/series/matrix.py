"""Assemble the wide minute matrix (time x component::kpi), segmented by day, with fill + mask."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from tcc_rca.ingestion.timeutil import OPENRCA_TZ
from tcc_rca.series.bank import bank_app_long, bank_metric_long, bank_trace_long
from tcc_rca.series.catalog import KpiCatalog, split_feature
from tcc_rca.series.logs import LogTemplateModel, is_count_kpi, log_long
from tcc_rca.series.metrics import metric_long
from tcc_rca.series.traces import trace_long

INDEX_COLUMNS = ("day", "time_utc")


@dataclass(frozen=True)
class SeriesMatrix:
    values: pl.DataFrame  # day, time_utc, features... (raw scale, forward-filled within day)
    observed: pl.DataFrame  # day, time_utc, features... (True where a sample existed)
    sources: dict[str, str]  # feature -> source table
    # feature -> forward-fill limit in steps (only when adaptive filling is on; ADR 0012)
    fill_limits: dict[str, int] = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return [c for c in self.values.columns if c not in INDEX_COLUMNS]


def day_long(
    interim_dir: Path,
    system: str,
    day: str,
    catalog: KpiCatalog,
    step_seconds: int,
    log_model: LogTemplateModel | None = None,
) -> pl.DataFrame:
    """Long ``(time_utc, feature, source, value)`` of one day, clipped to the folder's day."""
    step = f"{step_seconds}s"
    if system == "Telecom":
        parts = [
            metric_long(interim_dir, system, day, catalog, step),
            trace_long(interim_dir, system, day, step),
        ]
    elif system == "Bank":
        if log_model is None:
            raise ValueError("Bank series need a fitted log template model")
        parts = [
            bank_metric_long(interim_dir, system, day, catalog, step),
            bank_app_long(interim_dir, system, day, step),
            bank_trace_long(interim_dir, system, day, step),
            log_long(interim_dir, system, day, step, log_model),
        ]
    else:
        raise KeyError(f"no series builder for system {system!r}")
    start = datetime.strptime(day, "%Y_%m_%d").replace(tzinfo=OPENRCA_TZ)
    # Files spill into neighbouring days (Bank traces start up to 4 h early); keep the folder's
    # UTC+8 calendar day only, so that no minute belongs to two days.
    return pl.concat(parts).filter(
        pl.col("time_utc").is_between(start, start + timedelta(days=1), closed="left")
    )


def _grid(long: pl.DataFrame, step_seconds: int) -> pl.DataFrame:
    """Regular minute grid spanning the day's observed telemetry (first to last minute)."""
    step = timedelta(seconds=step_seconds)
    return long.select(
        pl.datetime_range(pl.col("time_utc").min(), pl.col("time_utc").max(), step)
        .explode()
        .alias("time_utc")
    )


def build_matrix(
    interim_dir: Path,
    system: str,
    days: list[str],
    catalog: KpiCatalog,
    step_seconds: int,
    ffill_limit: int,
    log_model: LogTemplateModel | None = None,
    adaptive_fill_days: Sequence[str] | None = None,
    max_fill_limit: int = 60,
) -> SeriesMatrix:
    """Minute matrix of ``days``. Forward fill is limited to ``ffill_limit`` steps, or, when
    ``adaptive_fill_days`` is given, per feature to its native sampling step measured on those
    days only (at least ``ffill_limit``, at most ``max_fill_limit``)."""
    values_parts: list[pl.DataFrame] = []
    sources: dict[str, str] = {}
    for day in days:
        long = day_long(interim_dir, system, day, catalog, step_seconds, log_model)
        for feature, source in long.select("feature", "source").unique().iter_rows():
            if sources.setdefault(feature, source) != source:
                raise ValueError(f"feature {feature!r} produced by two sources")
        wide = long.pivot(on="feature", index="time_utc", values="value", aggregate_function=None)
        day_frame = _grid(long, step_seconds).join(wide, on="time_utc", how="left")
        values_parts.append(day_frame.with_columns(pl.lit(day).alias("day")))
    raw = pl.concat(values_parts, how="diagonal_relaxed")
    features = sorted(sources)
    for f in features:  # a feature absent on a whole day still needs its column
        if f not in raw.columns:
            raw = raw.with_columns(pl.lit(None, dtype=pl.Float64).alias(f))
    raw = raw.select(*INDEX_COLUMNS, *features).sort("time_utc")

    observed = raw.select(*INDEX_COLUMNS, *[pl.col(f).is_not_null() for f in features])
    # Zero events in a minute (spans, requests, log lines) is an observation, not a gap.
    calls = [f for f in features if is_count_kpi(split_feature(f)[1])]
    limits: dict[str, int] = {}
    if adaptive_fill_days is not None:
        steps = native_steps(observed, features, adaptive_fill_days)
        limits = {
            f: min(max_fill_limit, max(ffill_limit, steps.get(f, ffill_limit)))
            for f in features
            if f not in calls
        }
    filled = raw.with_columns([pl.col(f).fill_null(0.0) for f in calls]).with_columns(
        [
            pl.col(f).forward_fill(limit=limits.get(f, ffill_limit)).over("day")
            for f in features
            if f not in calls
        ]
    )
    return SeriesMatrix(filled, observed, sources, limits)


def native_steps(
    observed: pl.DataFrame, features: Sequence[str], days: Sequence[str]
) -> dict[str, int]:
    """Median spacing (in steps) between real samples of each feature, within days of ``days``.

    Only the given days are read, so the fill policy never depends on validation or test data.
    """
    sub = observed.filter(pl.col("day").is_in(list(days)))
    if sub.is_empty():
        raise ValueError(f"no rows for adaptive-fill days {list(days)}")
    day_arr = sub["day"].to_numpy()
    out: dict[str, int] = {}
    for f in features:
        idx = np.flatnonzero(sub[f].to_numpy())
        if idx.size < 2:
            continue
        same_day = day_arr[idx[1:]] == day_arr[idx[:-1]]
        gaps = np.diff(idx)[same_day]
        if gaps.size:
            out[f] = int(np.median(gaps))
    return out


def save_matrix(matrix: SeriesMatrix, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    values_path, observed_path = out_dir / "series.parquet", out_dir / "observed.parquet"
    matrix.values.write_parquet(values_path, compression="zstd")
    matrix.observed.write_parquet(observed_path, compression="zstd")
    return values_path, observed_path
