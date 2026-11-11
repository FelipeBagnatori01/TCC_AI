"""Lazy, per-day readers for OpenRCA telemetry and parquet conversion, one layout per system.

Each system has its own files and schemas, taken from the real files during the E1 inventory of
that system (see ``docs/DICIONARIO_DADOS.md``); nothing is shared by assumption. Schemas are
enforced on read: a file whose columns differ fails loudly instead of being silently coerced.
Epoch columns (milliseconds or seconds, per file) are converted to a UTC ``time_utc`` column; the
raw columns are kept unchanged in the interim parquet.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import polars as pl

from tcc_rca.ingestion.timeutil import OPENRCA_TZ

_DAY_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")

Schema = dict[str, pl.DataType | type[pl.DataType]]

# ---------------------------------------------------------------- Telecom (inventoried in E1)
_TELECOM_METRIC: Schema = {
    "itemid": pl.Int64,
    "name": pl.String,
    "bomc_id": pl.String,
    "timestamp": pl.Int64,
    "value": pl.Float64,
    "cmdb_id": pl.String,
}
_TELECOM_APP: Schema = {
    "serviceName": pl.String,
    "startTime": pl.Int64,
    "avg_time": pl.Float64,
    "num": pl.Int64,
    "succee_num": pl.Int64,
    "succee_rate": pl.Float64,
}
_TELECOM_TRACE: Schema = {
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

# ------------------------------------------------------------ Bank (inventoried in Plan B E1)
_BANK_METRIC: Schema = {
    "timestamp": pl.Int64,
    "cmdb_id": pl.String,
    "kpi_name": pl.String,
    "value": pl.Float64,
}
_BANK_APP: Schema = {
    "timestamp": pl.Int64,
    "rr": pl.Float64,
    "sr": pl.Float64,
    "cnt": pl.Int64,
    "mrt": pl.Float64,
    "tc": pl.String,
}
_BANK_TRACE: Schema = {
    "timestamp": pl.Int64,
    "cmdb_id": pl.String,
    "parent_id": pl.String,
    "span_id": pl.String,
    "trace_id": pl.String,
    "duration": pl.Float64,
}
_BANK_LOG: Schema = {
    "log_id": pl.String,
    "timestamp": pl.Int64,
    "cmdb_id": pl.String,
    "log_name": pl.String,
    "value": pl.String,
}

Kind = Literal["metric", "app", "trace", "log"]


@dataclass(frozen=True)
class TelemetryFile:
    """A telemetry table: location under a day directory, schema, time and identity columns."""

    name: str  # stem, e.g. "metric_node"
    subdir: str  # "metric", "trace" or "log"
    kind: Kind
    schema: Schema
    time_column: str  # epoch column
    # "auto": decided per file from the epoch magnitude (Bank traces switch from milliseconds to
    # seconds on 2021-03-23; see detect_epoch_unit).
    time_unit: Literal["ms", "s", "auto"]
    component_columns: tuple[str, ...]  # columns naming a component (first: the emitting one)
    kpi_column: str | None = None  # metric files: the KPI name column
    utc_column: str = "time_utc"  # UTC datetime column added on read


TELECOM_FILES: tuple[TelemetryFile, ...] = (
    TelemetryFile("metric_app", "metric", "app", _TELECOM_APP, "startTime", "ms", ("serviceName",)),
    *(
        TelemetryFile(
            n, "metric", "metric", _TELECOM_METRIC, "timestamp", "ms", ("cmdb_id",), "name"
        )
        for n in ("metric_container", "metric_middleware", "metric_node", "metric_service")
    ),
    TelemetryFile(
        "trace_span", "trace", "trace", _TELECOM_TRACE, "startTime", "ms", ("cmdb_id", "dsName")
    ),
)
BANK_FILES: tuple[TelemetryFile, ...] = (
    TelemetryFile("metric_app", "metric", "app", _BANK_APP, "timestamp", "s", ("tc",)),
    TelemetryFile(
        "metric_container",
        "metric",
        "metric",
        _BANK_METRIC,
        "timestamp",
        "s",
        ("cmdb_id",),
        "kpi_name",
    ),
    TelemetryFile("trace_span", "trace", "trace", _BANK_TRACE, "timestamp", "auto", ("cmdb_id",)),
    TelemetryFile("log_service", "log", "log", _BANK_LOG, "timestamp", "s", ("cmdb_id",)),
)
LAYOUTS: Mapping[str, tuple[TelemetryFile, ...]] = {"Telecom": TELECOM_FILES, "Bank": BANK_FILES}


def files_for(system: str) -> tuple[TelemetryFile, ...]:
    """The telemetry layout of ``system``; an uninventoried system is an error."""
    if system not in LAYOUTS:
        raise KeyError(f"no inventoried telemetry layout for system {system!r}")
    return LAYOUTS[system]


def file_for(system: str, name: str) -> TelemetryFile:
    for file in files_for(system):
        if file.name == name:
            return file
    raise KeyError(f"{system} has no telemetry file {name!r}")


def files_of_kind(system: str, kind: Kind) -> tuple[TelemetryFile, ...]:
    return tuple(f for f in files_for(system) if f.kind == kind)


def list_days(system_dir: Path) -> list[str]:
    """Day folders (``YYYY_MM_DD``, UTC+8 calendar day), chronologically sorted."""
    telemetry = system_dir / "telemetry"
    return sorted(p.name for p in telemetry.iterdir() if p.is_dir() and _DAY_RE.match(p.name))


def raw_path(system_dir: Path, day: str, file: TelemetryFile) -> Path:
    return system_dir / "telemetry" / day / file.subdir / f"{file.name}.csv"


_MS_FLOOR = 10**11  # epochs >= 1e11 are milliseconds (1973+); smaller ones are seconds


def detect_epoch_unit(lf: pl.LazyFrame, column: str, where: str) -> Literal["ms", "s"]:
    """Epoch unit of a whole file; a file mixing seconds and milliseconds is an error."""
    lo, hi = (
        lf.select(pl.col(column).min().alias("lo"), pl.col(column).max().alias("hi"))
        .collect()
        .row(0)
    )
    if lo is None:
        raise ValueError(f"{where}: no timestamps to detect the epoch unit")
    if lo >= _MS_FLOOR:
        return "ms"
    if hi < _MS_FLOOR:
        return "s"
    raise ValueError(f"{where}: mixes second and millisecond epochs ({lo} .. {hi})")


def scan_raw(system_dir: Path, day: str, file: TelemetryFile) -> pl.LazyFrame:
    """Lazily scan a raw CSV with its enforced schema and add the UTC time column."""
    path = raw_path(system_dir, day, file)
    header = pl.read_csv(path, n_rows=0).columns
    if header != list(file.schema):
        raise ValueError(f"{path}: columns {header} differ from expected {list(file.schema)}")
    lf = pl.scan_csv(path, schema=file.schema)
    unit = file.time_unit
    if unit == "auto":
        unit = detect_epoch_unit(lf, file.time_column, str(path))
    return lf.with_columns(
        pl.from_epoch(pl.col(file.time_column), time_unit=unit)
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
    for file in files_for(system):
        out = interim_path(interim_dir, system, day, file)
        if out.exists() and not overwrite:
            written.append(out)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".parquet.tmp")
        scan_raw(system_dir, day, file).sink_parquet(tmp, compression="zstd")
        try:
            check_day_range(tmp, day, file.utc_column)
        except ValueError:
            tmp.unlink()
            raise
        tmp.replace(out)
        written.append(out)
    return written


def check_day_range(parquet: Path, day: str, utc_column: str = "time_utc") -> None:
    """Every timestamp must fall within the folder's UTC+8 day, give or take one day.

    Catches wrong epoch units or corrupt clocks (e.g. a millisecond reading of second epochs
    lands in January 1970) instead of letting them flow into the series.
    """
    # Compare integer epoch seconds inside polars: an absurd unit may not fit a Python datetime.
    epoch = pl.col(utc_column).dt.epoch("s")
    lo, hi = (
        pl.scan_parquet(parquet)
        .select(epoch.min().alias("lo"), epoch.max().alias("hi"))
        .collect()
        .row(0)
    )
    if lo is None:
        return
    start = datetime.strptime(day, "%Y_%m_%d").replace(tzinfo=OPENRCA_TZ)
    allowed_lo, allowed_hi = start - timedelta(days=1), start + timedelta(days=2)
    if lo < allowed_lo.timestamp() or hi >= allowed_hi.timestamp():
        raise ValueError(
            f"{parquet.name} of day {day}: epoch seconds {lo} .. {hi} fall outside "
            f"[{allowed_lo}, {allowed_hi}); check the epoch unit"
        )


def scan_interim(interim_dir: Path, system: str, day: str, name: str) -> pl.LazyFrame:
    return pl.scan_parquet(interim_path(interim_dir, system, day, file_for(system, name)))
