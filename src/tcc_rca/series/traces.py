"""Trace spans -> per-component minute aggregates and caller -> callee edges for one day."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from tcc_rca.series.catalog import SEPARATOR

TRACE_KPIS = (
    "trace_calls",
    "trace_latency_mean",
    "trace_latency_p95",
    "trace_latency_p99",
    "trace_error_rate",
)


def _spans(interim_dir: Path, system: str, day: str) -> pl.LazyFrame:
    return pl.scan_parquet(interim_dir / system / day / "trace_span.parquet")


def _aggregate(lf: pl.LazyFrame, entity: str, step: str) -> pl.LazyFrame:
    elapsed = pl.col("elapsedTime")
    return (
        lf.with_columns(pl.col("time_utc").dt.truncate(step).alias("minute"))
        .group_by(entity, "minute")
        .agg(
            pl.len().cast(pl.Float64).alias("trace_calls"),
            elapsed.mean().alias("trace_latency_mean"),
            elapsed.quantile(0.95, "linear").alias("trace_latency_p95"),
            elapsed.quantile(0.99, "linear").alias("trace_latency_p99"),
            (1.0 - pl.col("success").cast(pl.Float64).mean()).alias("trace_error_rate"),
        )
        .rename({entity: "component"})
    )


def trace_long(interim_dir: Path, system: str, day: str, step: str) -> pl.DataFrame:
    """Aggregates per executing component (``cmdb_id``) and per called database (``dsName``)."""
    spans = _spans(interim_dir, system, day)
    by_executor = _aggregate(spans.filter(pl.col("cmdb_id").is_not_null()), "cmdb_id", step)
    by_database = _aggregate(spans.filter(pl.col("dsName").is_not_null()), "dsName", step)
    wide = pl.concat([by_executor, by_database]).collect()
    overlap = set(wide.group_by("component", "minute").len().filter(pl.col("len") > 1)["component"])
    if overlap:
        raise ValueError(f"components seen both as executor and database: {sorted(overlap)}")
    return wide.unpivot(
        index=["component", "minute"], on=list(TRACE_KPIS), variable_name="kpi"
    ).select(
        pl.col("minute").alias("time_utc"),
        (pl.col("component") + SEPARATOR + pl.col("kpi")).alias("feature"),
        pl.lit("trace").alias("source"),
        pl.col("value"),
    )


def trace_edges(interim_dir: Path, system: str, day: str) -> pl.DataFrame:
    """Observed calls ``src -> dst`` with counts: parent span component -> child span component
    (via ``pid`` -> ``id`` in the same trace) and span component -> ``dsName`` (JDBC)."""
    spans = _spans(interim_dir, system, day).select("traceId", "id", "pid", "cmdb_id", "dsName")
    parents = spans.select(
        "traceId", pl.col("id").alias("pid"), pl.col("cmdb_id").alias("parent_cmdb")
    )
    span_calls = (
        spans.join(parents, on=["traceId", "pid"], how="inner")
        .filter(
            pl.col("parent_cmdb").is_not_null()
            & pl.col("cmdb_id").is_not_null()
            & (pl.col("parent_cmdb") != pl.col("cmdb_id"))
        )
        .group_by(pl.col("parent_cmdb").alias("src"), pl.col("cmdb_id").alias("dst"))
        .agg(pl.len().alias("weight"))
    )
    db_calls = (
        spans.filter(pl.col("cmdb_id").is_not_null() & pl.col("dsName").is_not_null())
        .group_by(pl.col("cmdb_id").alias("src"), pl.col("dsName").alias("dst"))
        .agg(pl.len().alias("weight"))
    )
    return pl.concat([span_calls, db_calls]).collect()
