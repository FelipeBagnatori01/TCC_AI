"""Lazy, per-day readers for OpenRCA telemetry (metrics and traces) and parquet conversion.

Schemas below were taken from the real Telecom files during the E1 inventory (see
``docs/DICIONARIO_DADOS.md``) and are enforced on read: a file whose columns differ fails loudly
instead of being silently coerced. Epoch-millisecond columns are converted to UTC datetimes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

_DAY_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")

_METRIC_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "itemid": pl.Int64,
    "name": pl.String,
    "bomc_id": pl.String,
    "timestamp": pl.Int64,
    "value": pl.Float64,
    "cmdb_id": pl.String,
}
_APP_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "serviceName": pl.String,
    "startTime": pl.Int64,
    "avg_time": pl.Float64,
    "num": pl.Int64,
    "succee_num": pl.Int64,
    "succee_rate": pl.Float64,
}
_TRACE_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "callType": pl.String,
    "startTime": pl.Int64,
    "elapsedTime": pl.Float64,
    "success": pl.Boolean,
    "traceId": pl.String,
    "id": pl.String,
    "pid": pl.String,
    "cmdb_id": pl.String,
    "dsName": pl.String,
    "serviceName": pl.String,
}


@dataclass(frozen=True)
class TelemetryFile:
    """A telemetry table: its location under a day directory, schema and time column."""

    name: str  # stem, e.g. "metric_node"
    subdir: str  # "metric" or "trace"
    schema: dict[str, pl.DataType | type[pl.DataType]]
    time_column: str  # epoch-millisecond column
    utc_column: str  # name of the UTC datetime column added on read


TELEMETRY_FILES: tuple[TelemetryFile, ...] = (
    TelemetryFile("metric_app", "metric", _APP_SCHEMA, "startTime", "time_utc"),
    TelemetryFile("metric_container", "metric", _METRIC_SCHEMA, "timestamp", "time_utc"),
    TelemetryFile("metric_middleware", "metric", _METRIC_SCHEMA, "timestamp", "time_utc"),
    TelemetryFile("metric_node", "metric", _METRIC_SCHEMA, "timestamp", "time_utc"),
    TelemetryFile("metric_service", "metric", _METRIC_SCHEMA, "timestamp", "time_utc"),
    TelemetryFile("trace_span", "trace", _TRACE_SCHEMA, "startTime", "time_utc"),
)
FILES_BY_NAME = {f.name: f for f in TELEMETRY_FILES}


def list_days(system_dir: Path) -> list[str]:
    """Day folders (``YYYY_MM_DD``, UTC+8 calendar day), chronologically sorted."""
    telemetry = system_dir / "telemetry"
    return sorted(p.name for p in telemetry.iterdir() if p.is_dir() and _DAY_RE.match(p.name))


def raw_path(system_dir: Path, day: str, file: TelemetryFile) -> Path:
    return system_dir / "telemetry" / day / file.subdir / f"{file.name}.csv"


def scan_raw(system_dir: Path, day: str, file: TelemetryFile) -> pl.LazyFrame:
    """Lazily scan a raw CSV with its enforced schema and add the UTC time column."""
    path = raw_path(system_dir, day, file)
    header = pl.read_csv(path, n_rows=0).columns
    if header != list(file.schema):
        raise ValueError(f"{path}: columns {header} differ from expected {list(file.schema)}")
    lf = pl.scan_csv(path, schema=file.schema)
    return lf.with_columns(
        pl.from_epoch(pl.col(file.time_column), time_unit="ms")
        .dt.replace_time_zone("UTC")
        .alias(file.utc_column)
    )


def interim_path(interim_dir: Path, system: str, day: str, file: TelemetryFile) -> Path:
    return interim_dir / system / day / f"{file.name}.parquet"


def convert_day(
    system_dir: Path, system: str, day: str, interim_dir: Path, overwrite: bool = False
) -> list[Path]:
    """Convert one day to parquet with streaming writes (never materializes a whole file)."""
    written: list[Path] = []
    for file in TELEMETRY_FILES:
        out = interim_path(interim_dir, system, day, file)
        if out.exists() and not overwrite:
            written.append(out)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".parquet.tmp")
        scan_raw(system_dir, day, file).sink_parquet(tmp, compression="zstd")
        tmp.replace(out)
        written.append(out)
    return written


def scan_interim(interim_dir: Path, system: str, day: str, name: str) -> pl.LazyFrame:
    return pl.scan_parquet(interim_path(interim_dir, system, day, FILES_BY_NAME[name]))
