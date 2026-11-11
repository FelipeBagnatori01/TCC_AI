from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from synthetic import DAY, START_MS, make_system
from tcc_rca.ingestion.inventory import build_inventory
from tcc_rca.ingestion.telemetry import (
    convert_day,
    file_for,
    list_days,
    scan_interim,
    scan_raw,
)


def test_list_days_ignores_non_day_folders(tmp_path: Path) -> None:
    system = make_system(tmp_path)
    (system / "telemetry" / "not_a_day").mkdir()
    assert list_days(system) == [DAY]


def test_raw_timestamps_are_converted_to_utc(tmp_path: Path) -> None:
    system = make_system(tmp_path)
    df = scan_raw(system, DAY, file_for("Telecom", "metric_container")).collect()
    first = df.sort("time_utc")["time_utc"][0]
    assert first == datetime(2020, 4, 10, 16, 0, tzinfo=UTC)  # 00:00 UTC+8


def test_round_trip_raw_value_equals_interim_value(tmp_path: Path) -> None:
    """The value of a KPI at an instant read from parquet equals the raw CSV value."""
    system = make_system(tmp_path)
    interim = tmp_path / "interim"
    convert_day(system, "Telecom", DAY, interim)
    raw = scan_raw(system, DAY, file_for("Telecom", "metric_container")).collect()
    out = scan_interim(interim, "Telecom", DAY, "metric_container").collect()
    key = ["cmdb_id", "name", "timestamp"]
    assert raw.sort(key).select("value").equals(out.sort(key).select("value"))
    cpu_at_7min = out.filter(
        (pl.col("name") == "container_cpu_used") & (pl.col("timestamp") == START_MS + 7 * 60_000)
    )["value"][0]
    assert cpu_at_7min == 98.0


def test_convert_is_idempotent_and_overwrite_works(tmp_path: Path) -> None:
    system = make_system(tmp_path)
    interim = tmp_path / "interim"
    first = convert_day(system, "Telecom", DAY, interim)
    mtimes = [p.stat().st_mtime_ns for p in first]
    second = convert_day(system, "Telecom", DAY, interim)
    assert [p.stat().st_mtime_ns for p in second] == mtimes
    convert_day(system, "Telecom", DAY, interim, overwrite=True)
    assert all(p.exists() for p in first)


def test_unexpected_columns_fail_loudly(tmp_path: Path) -> None:
    system = make_system(tmp_path)
    path = system / "telemetry" / DAY / "metric" / "metric_node.csv"
    path.write_text("itemid,name,timestamp,value,cmdb_id\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differ from expected"):
        scan_raw(system, DAY, file_for("Telecom", "metric_node"))


def test_inventory_reports_structure(tmp_path: Path) -> None:
    system = make_system(tmp_path)
    inv = build_inventory(system)
    day = inv["days"][DAY]
    assert inv["n_days"] == 1
    assert day["metric_container"]["n_kpis"] == 2
    assert day["metric_container"]["n_components"] == 1
    assert day["metric_container"]["modal_step_ms"] == 60_000
    assert day["trace_span"]["rows"] == 40
