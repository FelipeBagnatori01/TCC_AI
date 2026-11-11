"""Bank telemetry layout (Plan B): second-resolution epochs, its own schemas, multi-line logs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from synthetic import BANK_DAY, make_bank_system
from tcc_rca.ingestion.cases import build_cases, read_failures, read_queries
from tcc_rca.ingestion.inventory import build_inventory
from tcc_rca.ingestion.quality import build_quality_report
from tcc_rca.ingestion.telemetry import convert_day, file_for, files_for, scan_interim, scan_raw


def test_unknown_system_has_no_layout() -> None:
    with pytest.raises(KeyError, match="no inventoried telemetry layout"):
        files_for("Market")


def test_second_epochs_become_the_right_utc_instant(tmp_path: Path) -> None:
    system = make_bank_system(tmp_path)
    metric = scan_raw(system, BANK_DAY, file_for("Bank", "metric_container")).collect()
    assert metric["time_utc"][0] == datetime(2021, 3, 3, 16, 0, tzinfo=UTC)  # 00:00 UTC+8
    trace = scan_raw(system, BANK_DAY, file_for("Bank", "trace_span")).collect()
    assert trace["time_utc"][0] == datetime(2021, 3, 3, 16, 0, 0, 5000, tzinfo=UTC)  # ms epoch


def test_multiline_log_values_survive_conversion(tmp_path: Path) -> None:
    system = make_bank_system(tmp_path)
    convert_day(system, "Bank", BANK_DAY, tmp_path / "interim")
    logs = scan_interim(tmp_path / "interim", "Bank", BANK_DAY, "log_service").collect()
    assert logs.height == 2
    assert logs["value"][0] == "[GC (Allocation Failure)\nline two, with comma]"


def test_bank_record_with_float_timestamps_and_query_window(tmp_path: Path) -> None:
    system = make_bank_system(tmp_path)
    [failure] = read_failures(system / "record.csv")
    assert failure.start_utc == datetime(2021, 3, 3, 16, 5, tzinfo=UTC)
    cases, orphans = build_cases(read_queries(system / "query.csv"), [failure])
    assert orphans == []
    assert cases[0].failures == (failure,)


def test_non_integral_record_timestamp_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "record.csv"
    bad.write_text(
        "level,component,timestamp,datetime,reason\npod,X,1614787500.5,2021-03-04 00:05:00,r\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-integral"):
        read_failures(bad)


def test_bank_inventory_and_quality(tmp_path: Path) -> None:
    system = make_bank_system(tmp_path)
    inv = build_inventory(system)
    day = inv["days"][BANK_DAY]
    assert set(day) == {"metric_app", "metric_container", "trace_span", "log_service"}
    assert day["metric_container"]["modal_step_ms"] == 60_000
    assert day["metric_container"]["n_kpis"] == 2
    assert day["log_service"]["n_log_names"] == 1
    interim = tmp_path / "interim"
    convert_day(system, "Bank", BANK_DAY, interim)
    cases, orphans = build_cases(
        read_queries(system / "query.csv"), read_failures(system / "record.csv")
    )
    report = build_quality_report(interim, "Bank", cases, len(orphans))
    assert report["series"]["metric_container"]["constant_series"] == [
        {"component": "Mysql01", "kpi": "Mysql-MySQL_3306_Uptime"}
    ]
    assert report["failures_without_metric_coverage"] == []
    assert report["window_coverage"][0]["trace_rows_after_30min"] > 0


def _rewrite_trace(system: Path, body: str) -> None:
    path = system / "telemetry" / BANK_DAY / "trace" / "trace_span.csv"
    path.write_text("timestamp,cmdb_id,parent_id,span_id,trace_id,duration\n" + body, "utf-8")


def test_trace_epoch_unit_is_detected_per_file(tmp_path: Path) -> None:
    """Bank traces switch from millisecond to second epochs on 2021-03-23 (real inventory)."""
    system = make_bank_system(tmp_path)
    _rewrite_trace(system, "1614787205,Tomcat01,p,s,t,3\n")  # 10 digits: seconds
    trace = scan_raw(system, BANK_DAY, file_for("Bank", "trace_span")).collect()
    assert trace["time_utc"][0] == datetime(2021, 3, 3, 16, 0, 5, tzinfo=UTC)


def test_file_mixing_epoch_units_is_rejected(tmp_path: Path) -> None:
    system = make_bank_system(tmp_path)
    _rewrite_trace(system, "1614787205,Tomcat01,p,s,t,3\n1614787205000,Tomcat01,p,s,t,3\n")
    with pytest.raises(ValueError, match="mixes second and millisecond"):
        scan_raw(system, BANK_DAY, file_for("Bank", "trace_span"))


def test_conversion_rejects_timestamps_outside_the_day(tmp_path: Path) -> None:
    """A wrong epoch unit lands in 1970; the day-range guard must stop the conversion."""
    system = make_bank_system(tmp_path)
    _rewrite_trace(system, "1614787205,Tomcat01,p,s,t,3\n")
    path = system / "telemetry" / BANK_DAY / "metric" / "metric_app.csv"
    path.write_text("timestamp,rr,sr,cnt,mrt,tc\n1614787205000,1.0,1.0,1,1.0,S\n", "utf-8")
    interim = tmp_path / "interim"
    with pytest.raises(ValueError, match="fall outside"):
        convert_day(system, "Bank", BANK_DAY, interim)
    assert not list(interim.rglob("*.tmp"))  # no half-written file left behind
