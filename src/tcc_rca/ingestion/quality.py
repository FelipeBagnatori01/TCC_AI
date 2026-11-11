"""Data-quality report: gaps, nulls, constant KPIs, components without data, window coverage."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from tcc_rca.ingestion.cases import Case
from tcc_rca.ingestion.telemetry import TelemetryFile, files_for, files_of_kind

_MARGIN = timedelta(minutes=30)
_GAP_THRESHOLD_S = 300  # a silent stretch longer than 5 minutes inside one series


def _glob(interim_dir: Path, system: str, name: str) -> str:
    return str(interim_dir / system / "*" / f"{name}.parquet")


def _scan(interim_dir: Path, system: str, file: TelemetryFile) -> pl.LazyFrame:
    return pl.scan_parquet(_glob(interim_dir, system, file.name))


def null_and_row_counts(interim_dir: Path, system: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for file in files_for(system):
        lf = _scan(interim_dir, system, file)
        cols = list(file.schema)
        row = lf.select(
            [pl.len().alias("__rows")] + [pl.col(c).null_count().alias(c) for c in cols]
        ).collect()
        out[file.name] = {"rows": int(row["__rows"][0]), "nulls": {c: int(row[c][0]) for c in cols}}
    return out


def series_summary(interim_dir: Path, system: str, file: TelemetryFile) -> pl.DataFrame:
    """One row per (component, KPI) series: samples, distinct values, largest silent gap (ms)."""
    if file.kpi_column is None:
        raise ValueError(f"{file.name} is not a long-format metric file")
    comp, kpi = file.component_columns[0], file.kpi_column
    lf = _scan(interim_dir, system, file)
    return (
        lf.group_by([comp, kpi])
        .agg(
            pl.len().alias("n"),
            pl.col("value").n_unique().alias("n_unique_values"),
            pl.col(file.utc_column).sort().diff().max().dt.total_milliseconds().alias("max_gap_ms"),
        )
        .sort([comp, kpi])
        .rename({comp: "component", kpi: "kpi"})
        .collect()
    )


def constant_and_gappy(interim_dir: Path, system: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for file in files_of_kind(system, "metric"):
        s = series_summary(interim_dir, system, file)
        out[file.name] = {
            "series": s.height,
            "constant_series": s.filter(pl.col("n_unique_values") <= 1)
            .select("component", "kpi")
            .to_dicts(),
            # max_gap_ms includes silences between days; per-day gaps are analysed in E2.
            "components": sorted(s["component"].unique().to_list()),
        }
    return out


def components_in_telemetry(interim_dir: Path, system: str) -> set[str]:
    """Components emitting metrics, traces or logs (first component column of each file)."""
    found: set[str] = set()
    for file in files_for(system):
        if file.kind == "app":  # service names, not deployed components
            continue
        col = file.component_columns[0]
        found |= set(
            _scan(interim_dir, system, file).select(col).drop_nulls().unique().collect()[col]
        )
    return found


def _day_minute_values(interim_dir: Path, system: str, day: str) -> pl.DataFrame:
    """Last value per (source, component, KPI, minute of the UTC+8 day) of every metric file."""
    parts = []
    for file in files_of_kind(system, "metric"):
        if file.kpi_column is None:
            continue
        local = pl.col(file.utc_column).dt.convert_time_zone("Asia/Shanghai")
        parts.append(
            pl.scan_parquet(interim_dir / system / day / f"{file.name}.parquet")
            .select(
                pl.lit(file.name).alias("source"),
                pl.col(file.component_columns[0]).alias("component"),
                pl.col(file.kpi_column).alias("kpi"),
                (local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute()).alias("minute"),
                pl.col("value"),
            )
            .group_by("source", "component", "kpi", "minute")
            .agg(pl.col("value").last())
            .collect()
        )
    return pl.concat(parts)


def day_similarity(interim_dir: Path, system: str, days: list[str]) -> pl.DataFrame:
    """For every pair of days: share of (series, minute-of-day) cells holding the same value.

    Near-copies of a day show up as pairs close to 1 (Telecom: three pairs above 99.99%, ADR
    0008). Cells are matched on source, component, KPI and minute of the UTC+8 day.
    """
    frames = {d: _day_minute_values(interim_dir, system, d) for d in days}
    keys = ["source", "component", "kpi", "minute"]
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(days):
        for b in days[i + 1 :]:
            joined = (
                frames[a].join(frames[b], on=keys, suffix="_b").drop_nulls(["value", "value_b"])
            )
            equal = joined.select(
                ((pl.col("value") - pl.col("value_b")).abs() <= 1e-9 * pl.col("value_b").abs())
                .sum()
                .alias("eq")
            ).item()
            rows.append(
                {
                    "day_a": a,
                    "day_b": b,
                    "common_cells": joined.height,
                    "equal_share": round(equal / joined.height, 6) if joined.height else None,
                }
            )
    return pl.DataFrame(rows).sort("equal_share", descending=True)


def _count_rows(lf: pl.LazyFrame, predicate: pl.Expr, lo: datetime, hi: datetime) -> int:
    window = pl.col("time_utc").is_between(lo, hi, closed="left")
    return int(lf.filter(predicate & window).select(pl.len()).collect().item())


def window_coverage(interim_dir: Path, system: str, cases: list[Case]) -> list[dict[str, Any]]:
    """For each labeled failure: telemetry rows of the labeled component around its start."""
    metric_lfs = [
        (_scan(interim_dir, system, f), f.component_columns[0])
        for f in files_of_kind(system, "metric")
    ]
    [trace_file] = files_of_kind(system, "trace")
    trace_lf = _scan(interim_dir, system, trace_file)
    rows: list[dict[str, Any]] = []
    for case in cases:
        for failure in case.failures:
            comp = failure.component
            trace_pred = pl.any_horizontal(
                [pl.col(c) == comp for c in trace_file.component_columns]
            )
            windows = {
                "before": (failure.start_utc - _MARGIN, failure.start_utc),
                "after": (failure.start_utc, failure.start_utc + _MARGIN),
            }
            row: dict[str, Any] = {
                "query_row": case.query.row,
                "task": case.query.task_index,
                "component": comp,
                "reason": failure.reason,
                "start_utc": failure.start_utc.isoformat(),
            }
            for label, (lo, hi) in windows.items():
                row[f"metric_rows_{label}_30min"] = sum(
                    _count_rows(lf, pl.col(col) == comp, lo, hi) for lf, col in metric_lfs
                )
                row[f"trace_rows_{label}_30min"] = _count_rows(trace_lf, trace_pred, lo, hi)
            row["has_metric_data"] = bool(
                row["metric_rows_before_30min"] and row["metric_rows_after_30min"]
            )
            rows.append(row)
    return rows


def build_quality_report(
    interim_dir: Path, system: str, cases: list[Case], orphans_count: int
) -> dict[str, Any]:
    counts = null_and_row_counts(interim_dir, system)
    series = constant_and_gappy(interim_dir, system)
    in_telemetry = components_in_telemetry(interim_dir, system)
    coverage = window_coverage(interim_dir, system, cases)
    labeled = sorted({f.component for c in cases for f in c.failures})
    return {
        "system": system,
        "row_and_null_counts": counts,
        "series": series,
        "components_in_any_telemetry": sorted(in_telemetry),
        "labeled_components_without_any_data": [c for c in labeled if c not in in_telemetry],
        "queries_without_failures": [c.query.row for c in cases if not c.failures],
        "failures_outside_any_query_window": orphans_count,
        "window_coverage": coverage,
        "failures_without_metric_coverage": [r for r in coverage if not r["has_metric_data"]],
    }


def write_quality_report(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"# Relatório de qualidade — {report['system']}", ""]
    lines += ["## Linhas e nulos", "", "| Tabela | Linhas | Colunas com nulos |", "|---|---|---|"]
    for name, d in report["row_and_null_counts"].items():
        nulls = ", ".join(f"{c}={n}" for c, n in d["nulls"].items() if n) or "nenhuma"
        lines.append(f"| {name} | {d['rows']} | {nulls} |")
    lines += ["", "## Séries constantes", "", "| Tabela | Séries | Constantes |", "|---|---|---|"]
    for name, d in report["series"].items():
        lines.append(f"| {name} | {d['series']} | {len(d['constant_series'])} |")
    lines += ["", "## Cobertura nas janelas de falha", ""]
    lines.append(f"- Falhas rotuladas: {len(report['window_coverage'])}")
    lines.append(f"- Sem cobertura de métricas: {len(report['failures_without_metric_coverage'])}")
    no_data = report["labeled_components_without_any_data"] or "nenhum"
    lines.append(f"- Componentes rotulados sem nenhum dado: {no_data}")
    lines.append(
        f"- Consultas sem falha associada: {report['queries_without_failures'] or 'nenhuma'}"
    )
    outside = report["failures_outside_any_query_window"]
    lines.append(f"- Falhas fora de qualquer janela de consulta: {outside}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
