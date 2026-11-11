"""Bank metrics, service metrics and traces -> long minute series; trace call edges (ADR 0012)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from tcc_rca.series.catalog import SEPARATOR, KpiCatalog

BANK_TRACE_KPIS = ("trace_calls", "trace_latency_mean", "trace_latency_p95", "trace_latency_p99")
BANK_APP_KPIS = ("app_calls", "app_mrt", "app_sr", "app_rr")
_CONTAINER_RE = r"^Container-DOCKER_CONTAINER_[0-9a-f]+--.*_([A-Za-z]+)$"
_AGG = {
    "mean": pl.col("value").mean(),
    "max": pl.col("value").max(),
    "min": pl.col("value").min(),
    "sum": pl.col("value").sum(),
}


def _day(interim_dir: Path, system: str, day: str, name: str) -> pl.LazyFrame:
    return pl.scan_parquet(interim_dir / system / day / f"{name}.parquet")


def normalize_kpi(kpi_name: pl.Expr) -> pl.Expr:
    """Container KPIs embed the container hash, which changes on every restart; keep the metric."""
    metric = kpi_name.str.extract(_CONTAINER_RE, 1)
    return pl.when(metric.is_not_null()).then(pl.lit("Container_") + metric).otherwise(kpi_name)


def bank_metric_long(
    interim_dir: Path, system: str, day: str, catalog: KpiCatalog, step: str
) -> pl.DataFrame:
    """Per minute: last sample of each raw series, then series sharing a normalized KPI (old and
    new container during a restart) combined with the KPI's ``item_agg``."""
    per_series = (
        _day(interim_dir, system, day, "metric_container")
        .with_columns(
            pl.col("time_utc").dt.truncate(step).alias("minute"),
            normalize_kpi(pl.col("kpi_name")).alias("kpi"),
        )
        .sort("timestamp")
        .group_by("cmdb_id", "kpi_name", "kpi", "minute")
        .agg(pl.col("value").last())
        .collect()
    )
    kpis = per_series["kpi"].unique().to_list()
    infos = {k: catalog.lookup("metric_container", k) for k in kpis}
    missing = sorted(k for k, info in infos.items() if info is None)
    if missing:
        raise KeyError(f"metric_container: KPIs without catalog entry: {missing}")
    frames = []
    for agg_name, agg_expr in _AGG.items():
        selected = [
            k for k, info in infos.items() if info is not None and info.item_agg == agg_name
        ]
        part = per_series.filter(pl.col("kpi").is_in(selected))
        if part.is_empty():
            continue
        frames.append(
            part.group_by("cmdb_id", "kpi", "minute")
            .agg(agg_expr)
            .select(
                pl.col("minute").alias("time_utc"),
                (pl.col("cmdb_id") + SEPARATOR + pl.col("kpi")).alias("feature"),
                pl.lit("metric_container").alias("source"),
                pl.col("value"),
            )
        )
    return pl.concat(frames)


def bank_app_long(interim_dir: Path, system: str, day: str, step: str) -> pl.DataFrame:
    """Service rows of a minute merged: calls summed, mrt/sr/rr weighted by calls."""
    calls = pl.col("cnt").cast(pl.Float64)

    def weighted(col: str) -> pl.Expr:
        return ((pl.col(col) * calls).sum() / calls.sum()).alias(f"app_{col}")

    df = (
        _day(interim_dir, system, day, "metric_app")
        .with_columns(pl.col("time_utc").dt.truncate(step).alias("minute"))
        .group_by("tc", "minute")
        .agg(calls.sum().alias("app_calls"), weighted("mrt"), weighted("sr"), weighted("rr"))
        .collect()
    )
    return df.unpivot(index=["tc", "minute"], on=list(BANK_APP_KPIS), variable_name="kpi").select(
        pl.col("minute").alias("time_utc"),
        (pl.col("tc") + SEPARATOR + pl.col("kpi")).alias("feature"),
        pl.lit("metric_app").alias("source"),
        pl.col("value").cast(pl.Float64),
    )


def bank_trace_long(interim_dir: Path, system: str, day: str, step: str) -> pl.DataFrame:
    """Per emitting component and minute: span count and duration mean/p95/p99.

    Bank spans have no success flag, so there is no error-rate feature (unlike Telecom).
    """
    duration = pl.col("duration")
    wide = (
        _day(interim_dir, system, day, "trace_span")
        .filter(pl.col("cmdb_id").is_not_null())
        .with_columns(pl.col("time_utc").dt.truncate(step).alias("minute"))
        .group_by("cmdb_id", "minute")
        .agg(
            pl.len().cast(pl.Float64).alias("trace_calls"),
            duration.mean().alias("trace_latency_mean"),
            duration.quantile(0.95, "linear").alias("trace_latency_p95"),
            duration.quantile(0.99, "linear").alias("trace_latency_p99"),
        )
        .collect()
    )
    return wide.unpivot(
        index=["cmdb_id", "minute"], on=list(BANK_TRACE_KPIS), variable_name="kpi"
    ).select(
        pl.col("minute").alias("time_utc"),
        (pl.col("cmdb_id") + SEPARATOR + pl.col("kpi")).alias("feature"),
        pl.lit("trace").alias("source"),
        pl.col("value"),
    )


def bank_trace_edges(interim_dir: Path, system: str, day: str) -> pl.DataFrame:
    """Observed calls ``src -> dst``: component of the parent span -> component of the child span
    (``parent_id`` -> ``span_id`` within the same ``trace_id``)."""
    spans = _day(interim_dir, system, day, "trace_span").select(
        "trace_id", "span_id", "parent_id", "cmdb_id"
    )
    parents = spans.select(
        "trace_id", pl.col("span_id").alias("parent_id"), pl.col("cmdb_id").alias("parent_cmdb")
    )
    return (
        spans.join(parents, on=["trace_id", "parent_id"], how="inner")
        .filter(
            pl.col("parent_cmdb").is_not_null()
            & pl.col("cmdb_id").is_not_null()
            & (pl.col("parent_cmdb") != pl.col("cmdb_id"))
        )
        .group_by(pl.col("parent_cmdb").alias("src"), pl.col("cmdb_id").alias("dst"))
        .agg(pl.len().alias("weight"))
        .collect()
    )
