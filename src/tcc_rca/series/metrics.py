"""Metric tables -> long minute series ``(time_utc, feature, value)`` for one day."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from tcc_rca.series.catalog import SEPARATOR, KpiCatalog

SERIES_METRIC_FILES = ("metric_container", "metric_middleware", "metric_node", "metric_service")
APP_KPIS = ("avg_time", "num", "succee_num", "succee_rate")
_ITEM_AGG = {
    "mean": pl.col("value").mean(),
    "max": pl.col("value").max(),
    "min": pl.col("value").min(),
    "sum": pl.col("value").sum(),
}


def _day_file(interim_dir: Path, system: str, day: str, name: str) -> Path:
    return interim_dir / system / day / f"{name}.parquet"


def metric_long(
    interim_dir: Path, system: str, day: str, catalog: KpiCatalog, step: str
) -> pl.DataFrame:
    """Per minute: last sample of each item, then items combined with the KPI's ``item_agg``."""
    frames: list[pl.DataFrame] = []
    for name in SERIES_METRIC_FILES:
        per_item = (
            pl.scan_parquet(_day_file(interim_dir, system, day, name))
            .with_columns(pl.col("time_utc").dt.truncate(step).alias("minute"))
            .sort("timestamp")
            .group_by("cmdb_id", "name", "itemid", "minute")
            .agg(pl.col("value").last())
            .collect()
        )
        for agg_name, agg_expr in _ITEM_AGG.items():
            kpis = [k for k, info in catalog.get(name, {}).items() if info.item_agg == agg_name]
            part = per_item.filter(pl.col("name").is_in(kpis))
            if part.is_empty():
                continue
            frames.append(
                part.group_by("cmdb_id", "name", "minute")
                .agg(agg_expr)
                .select(
                    pl.col("minute").alias("time_utc"),
                    (pl.col("cmdb_id") + SEPARATOR + pl.col("name")).alias("feature"),
                    pl.lit(name).alias("source"),
                    pl.col("value"),
                )
            )
        unknown = per_item.filter(~pl.col("name").is_in(list(catalog.get(name, {}))))
        if not unknown.is_empty():
            raise KeyError(
                f"{name}: KPIs without catalog entry: {unknown['name'].unique().to_list()}"
            )
    frames.append(app_long(interim_dir, system, day, step))
    return pl.concat(frames)


def app_long(interim_dir: Path, system: str, day: str, step: str) -> pl.DataFrame:
    """Service-level aggregates; several rows per minute are merged by call-weighted sums."""
    df = (
        pl.scan_parquet(_day_file(interim_dir, system, day, "metric_app"))
        .with_columns(pl.col("time_utc").dt.truncate(step).alias("minute"))
        .group_by("serviceName", "minute")
        .agg(
            pl.col("num").sum(),
            pl.col("succee_num").sum(),
            ((pl.col("avg_time") * pl.col("num")).sum() / pl.col("num").sum()).alias("avg_time"),
        )
        .with_columns((pl.col("succee_num") / pl.col("num")).alias("succee_rate"))
        .collect()
    )
    return df.unpivot(
        index=["serviceName", "minute"], on=list(APP_KPIS), variable_name="kpi"
    ).select(
        pl.col("minute").alias("time_utc"),
        (pl.col("serviceName") + SEPARATOR + pl.col("kpi")).alias("feature"),
        pl.lit("metric_app").alias("source"),
        pl.col("value").cast(pl.Float64),
    )
