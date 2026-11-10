"""Data-quality report: gaps, nulls, constant KPIs, components without data, window coverage."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from tcc_rca.ingestion.cases import Case
from tcc_rca.ingestion.telemetry import TELEMETRY_FILES, TelemetryFile

SERIES_METRIC_FILES = ("metric_container", "metric_middleware", "metric_node", "metric_service")
_MARGIN = timedelta(minutes=30)
_GAP_THRESHOLD_S = 300  # a silent stretch longer than 5 minutes inside one series


def _glob(interim_dir: Path, system: str, name: str) -> str:
    return str(interim_dir / system / "*" / f"{name}.parquet")


def _scan(interim_dir: Path, system: str, file: TelemetryFile) -> pl.LazyFrame:
    return pl.scan_parquet(_glob(interim_dir, system, file.name))


def null_and_row_counts(interim_dir: Path, system: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for file in TELEMETRY_FILES:
        lf = _scan(interim_dir, system, file)
        cols = list(file.schema)
        row = lf.select(
            [pl.len().alias("__rows")] + [pl.col(c).null_count().alias(c) for c in cols]
        ).collect()
        out[file.name] = {"rows": int(row["__rows"][0]), "nulls": {c: int(row[c][0]) for c in cols}}
    return out


def series_summary(interim_dir: Path, system: str, file: TelemetryFile) -> pl.DataFrame:
    """One row per (cmdb_id, name) series: samples, distinct values, largest silent gap."""
    lf = _scan(interim_dir, system, file)
    return (
        lf.group_by(["cmdb_id", "name"])
        .agg(
            pl.len().alias("n"),
            pl.col("value").n_unique().alias("n_unique_values"),
            pl.col("timestamp").sort().diff().max().alias("max_gap_ms"),
        )
        .sort(["cmdb_id", "name"])
        .collect()
    )


def constant_and_gappy(interim_dir: Path, system: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in SERIES_METRIC_FILES:
        file = next(f for f in TELEMETRY_FILES if f.name == name)
        s = series_summary(interim_dir, system, file)
        out[name] = {
            "series": s.height,
            "constant_series": s.filter(pl.col("n_unique_values") <= 1)
            .select("cmdb_id", "name")
            .to_dicts(),
            # Gaps inside a day are what matter; the cross-day silence (06:00 -> 00:00) is not a
            # gap in the data, so only the per-day telemetry window is considered in the series.
            "components": sorted(s["cmdb_id"].unique().to_list()),
        }
    return out


def components_in_telemetry(interim_dir: Path, system: str) -> set[str]:
    found: set[str] = set()
    for name in SERIES_METRIC_FILES:
        file = next(f for f in TELEMETRY_FILES if f.name == name)
        found |= set(
            _scan(interim_dir, system, file)
            .select("cmdb_id")
            .drop_nulls()
            .unique()
            .collect()["cmdb_id"]
        )
    trace = next(f for f in TELEMETRY_FILES if f.name == "trace_span")
    found |= set(
        _scan(interim_dir, system, trace)
        .select("cmdb_id")
        .drop_nulls()
        .unique()
        .collect()["cmdb_id"]
    )
    return found


def _count_rows(lf: pl.LazyFrame, predicate: pl.Expr, lo: datetime, hi: datetime) -> int:
    window = pl.col("time_utc").is_between(lo, hi, closed="left")
    return int(lf.filter(predicate & window).select(pl.len()).collect().item())


def window_coverage(interim_dir: Path, system: str, cases: list[Case]) -> list[dict[str, Any]]:
    """For each labeled failure: telemetry rows of the labeled component around its start."""
    metric_lfs = [
        _scan(interim_dir, system, f) for f in TELEMETRY_FILES if f.name in SERIES_METRIC_FILES
    ]
    trace_lf = _scan(
        interim_dir, system, next(f for f in TELEMETRY_FILES if f.name == "trace_span")
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        for failure in case.failures:
            comp = failure.component
            metric_pred = pl.col("cmdb_id") == comp
            trace_pred = (pl.col("cmdb_id") == comp) | (pl.col("dsName") == comp)
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
                    _count_rows(lf, metric_pred, lo, hi) for lf in metric_lfs
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
