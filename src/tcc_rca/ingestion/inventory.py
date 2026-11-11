"""Structural inventory of the raw telemetry: files, schemas, granularity, components, KPIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from tcc_rca.ingestion.telemetry import (
    TelemetryFile,
    detect_epoch_unit,
    files_for,
    list_days,
    raw_path,
)


def _infer_schema(path: Path) -> dict[str, str]:
    """Schema as inferred from the file itself (independent of the enforced reader schema)."""
    schema = pl.scan_csv(path, infer_schema_length=10_000).collect_schema()
    return {name: str(dtype) for name, dtype in schema.items()}


def _file_stats(path: Path, file: TelemetryFile) -> dict[str, Any]:
    lf = pl.scan_csv(path, infer_schema_length=10_000)
    t = pl.col(file.time_column)
    aggs: list[pl.Expr] = [
        pl.len().alias("rows"),
        t.min().alias("t_min_raw"),
        t.max().alias("t_max_raw"),
        pl.col(file.component_columns[0]).n_unique().alias("n_components"),
    ]
    if file.kpi_column is not None:
        aggs.append(pl.col(file.kpi_column).n_unique().alias("n_kpis"))
    if file.kind == "log":
        aggs.append(pl.col("log_name").n_unique().alias("n_log_names"))
    stats: dict[str, Any] = lf.select(aggs).collect().to_dicts()[0]
    stats["time_unit"] = (
        detect_epoch_unit(lf, file.time_column, str(path))
        if file.time_unit == "auto"
        else file.time_unit
    )
    # Dominant spacing between consecutive samples of one series -> granularity (in ms).
    if file.kind == "metric" and file.kpi_column is not None:
        to_ms = 1000 if stats["time_unit"] == "s" else 1
        step = (
            lf.group_by([file.component_columns[0], file.kpi_column])
            .agg(t.sort().diff().alias("d"))
            .explode("d")
            .drop_nulls("d")
            .select((pl.col("d").mode().first() * to_ms).alias("modal_step_ms"))
            .collect()
        )
        stats["modal_step_ms"] = step.item() if step.height else None
    return stats


def build_inventory(system_dir: Path) -> dict[str, Any]:
    """Inventory of every day and file of one system. Reads one file at a time, lazily.

    The system's layout is chosen by the directory name (``Telecom``, ``Bank``).
    """
    days: dict[str, Any] = {}
    for day in list_days(system_dir):
        entry: dict[str, Any] = {}
        for file in files_for(system_dir.name):
            path = raw_path(system_dir, day, file)
            if not path.is_file():
                entry[file.name] = {"missing": True}
                continue
            entry[file.name] = {
                "size_bytes": path.stat().st_size,
                "schema": _infer_schema(path),
                **_file_stats(path, file),
            }
        days[day] = entry
    return {"system": system_dir.name, "n_days": len(days), "days": days}


def write_inventory(inventory: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
