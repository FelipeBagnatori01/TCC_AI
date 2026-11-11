"""Assemble the wide minute matrix (time x component::kpi), segmented by day, with fill + mask."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import polars as pl

from tcc_rca.series.catalog import KpiCatalog
from tcc_rca.series.metrics import metric_long
from tcc_rca.series.traces import trace_long

INDEX_COLUMNS = ("day", "time_utc")
_CALLS_SUFFIX = "::trace_calls"


@dataclass(frozen=True)
class SeriesMatrix:
    values: pl.DataFrame  # day, time_utc, features... (raw scale, forward-filled within day)
    observed: pl.DataFrame  # day, time_utc, features... (True where a sample existed)
    sources: dict[str, str]  # feature -> source table

    @property
    def features(self) -> list[str]:
        return [c for c in self.values.columns if c not in INDEX_COLUMNS]


def day_long(
    interim_dir: Path, system: str, day: str, catalog: KpiCatalog, step_seconds: int
) -> pl.DataFrame:
    step = f"{step_seconds}s"
    return pl.concat(
        [
            metric_long(interim_dir, system, day, catalog, step),
            trace_long(interim_dir, system, day, step),
        ]
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
) -> SeriesMatrix:
    values_parts: list[pl.DataFrame] = []
    sources: dict[str, str] = {}
    for day in days:
        long = day_long(interim_dir, system, day, catalog, step_seconds)
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
    # Zero spans in a minute of a day that has traces is an observation (no calls), not a gap.
    calls = [f for f in features if f.endswith(_CALLS_SUFFIX)]
    filled = raw.with_columns([pl.col(f).fill_null(0.0) for f in calls]).with_columns(
        [pl.col(f).forward_fill(limit=ffill_limit).over("day") for f in features if f not in calls]
    )
    return SeriesMatrix(filled, observed, sources)


def save_matrix(matrix: SeriesMatrix, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    values_path, observed_path = out_dir / "series.parquet", out_dir / "observed.parquet"
    matrix.values.write_parquet(values_path, compression="zstd")
    matrix.observed.write_parquet(observed_path, compression="zstd")
    return values_path, observed_path
