from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from synthetic import (
    DAY,
    DAY2,
    MIN_MS,
    START_MS,
    TRACE_HEADER,
    add_shifted_day,
    make_system,
    write_metric,
)
from tcc_rca.ingestion.telemetry import convert_day
from tcc_rca.series.catalog import build_feature_catalog, load_kpi_catalog
from tcc_rca.series.matrix import SeriesMatrix, build_matrix
from tcc_rca.series.traces import trace_edges


@pytest.fixture
def kpi_catalog(repo_root: Path):  # type: ignore[no-untyped-def]
    return load_kpi_catalog(repo_root / "configs/data/kpi_catalog_telecom.yaml")


def _build(tmp_path: Path, kpi_catalog, days=(DAY,), limit: int = 5) -> SeriesMatrix:  # type: ignore[no-untyped-def]
    system = tmp_path / "Sys"
    interim = tmp_path / "interim"
    for day in days:
        convert_day(system, "Sys", day, interim, overwrite=True)
    return build_matrix(interim, "Sys", list(days), kpi_catalog, 60, limit)


def _at(matrix: SeriesMatrix, feature: str, minute: int, day_offset_ms: int = 0) -> float | None:
    when = datetime.fromtimestamp((START_MS + day_offset_ms + minute * MIN_MS) / 1000, tz=UTC)
    value = matrix.values.filter(pl.col("time_utc") == when)[feature][0]
    return None if value is None else float(value)


def test_round_trip_raw_value_equals_matrix_value(tmp_path: Path, kpi_catalog) -> None:  # type: ignore[no-untyped-def]
    make_system(tmp_path)
    matrix = _build(tmp_path, kpi_catalog)
    # raw CSV: docker_001 container_cpu_used is 98.0 at minutes 6..9 and 1.0 at minute 0
    assert _at(matrix, "docker_001::container_cpu_used", 7) == 98.0
    assert _at(matrix, "docker_001::container_cpu_used", 0) == 1.0
    assert matrix.values.height == 40


def test_every_feature_is_catalogued_with_readable_name(tmp_path: Path, kpi_catalog) -> None:  # type: ignore[no-untyped-def]
    make_system(tmp_path)
    matrix = _build(tmp_path, kpi_catalog)
    entries = build_feature_catalog(matrix.sources, kpi_catalog)
    assert {e.feature for e in entries} == set(matrix.features)
    assert all(re.fullmatch(r"[a-z_]+_\d{3}::\w+", f) for f in matrix.features)
    assert "osb_001::avg_time" in matrix.features
    assert "db_001::trace_calls" in matrix.features  # database seen through dsName


def test_missing_catalog_entry_fails(kpi_catalog) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(KeyError, match="without catalog entry"):
        build_feature_catalog({"docker_001::made_up_kpi": "metric_container"}, kpi_catalog)


def test_samples_in_same_minute_keep_last_and_items_use_kpi_rule(
    tmp_path: Path, kpi_catalog
) -> None:  # type: ignore[no-untyped-def]
    system = make_system(tmp_path)
    t = START_MS
    write_metric(
        system,
        "metric_node",
        [
            (1, "CPU_util_pct", t + 1_000, 10.0, "os_001"),  # same item twice: last wins
            (1, "CPU_util_pct", t + 30_000, 20.0, "os_001"),
            (1, "Disk_io_util", t, 5.0, "os_001"),  # two disks: max
            (2, "Disk_io_util", t, 70.0, "os_001"),
            (1, "Sent_packets", t, 3.0, "os_001"),  # two interfaces: sum
            (2, "Sent_packets", t, 4.0, "os_001"),
        ],
    )
    matrix = _build(tmp_path, kpi_catalog)
    assert _at(matrix, "os_001::CPU_util_pct", 0) == 20.0
    assert _at(matrix, "os_001::Disk_io_util", 0) == 70.0
    assert _at(matrix, "os_001::Sent_packets", 0) == 7.0


def test_forward_fill_is_limited_never_backward_and_marked(tmp_path: Path, kpi_catalog) -> None:  # type: ignore[no-untyped-def]
    system = make_system(tmp_path)
    # os_001 sampled at minutes 2 and 20 only; other files keep the grid at 40 minutes
    write_metric(
        system,
        "metric_node",
        [
            (1, "CPU_util_pct", START_MS + 2 * MIN_MS, 50.0, "os_001"),
            (1, "CPU_util_pct", START_MS + 20 * MIN_MS, 60.0, "os_001"),
        ],
    )
    matrix = _build(tmp_path, kpi_catalog, limit=3)
    f = "os_001::CPU_util_pct"
    assert _at(matrix, f, 0) is None  # no backward fill (would use the future)
    assert _at(matrix, f, 5) == 50.0  # filled up to 3 steps
    assert _at(matrix, f, 6) is None  # beyond the limit stays missing
    assert _at(matrix, f, 19) is None  # never interpolated toward the future value
    observed = matrix.observed.filter(pl.col("time_utc") == matrix.values["time_utc"][3])[f][0]
    assert observed is False


def test_forward_fill_does_not_cross_days(tmp_path: Path, kpi_catalog) -> None:  # type: ignore[no-untyped-def]
    system = make_system(tmp_path)
    write_metric(
        system,
        "metric_node",
        [(1, "CPU_util_pct", START_MS + 39 * MIN_MS, 50.0, "os_001")],
    )
    add_shifted_day(system)
    write_metric(
        system, "metric_node", [(1, "CPU_util_pct", START_MS + 39 * MIN_MS, 50.0, "os_001")]
    )
    # day 2 starts with no os_001 sample: its first minute must stay empty
    path = system / "telemetry" / DAY2 / "metric" / "metric_node.csv"
    path.write_text(path.read_text().splitlines()[0] + "\n", encoding="utf-8")
    matrix = _build(tmp_path, kpi_catalog, days=(DAY, DAY2), limit=5)
    assert _at(matrix, "os_001::CPU_util_pct", 39) == 50.0
    assert _at(matrix, "os_001::CPU_util_pct", 0, day_offset_ms=86_400_000) is None
    assert set(matrix.values["day"].unique()) == {DAY, DAY2}


def test_trace_minutes_without_spans_count_zero_calls(tmp_path: Path, kpi_catalog) -> None:  # type: ignore[no-untyped-def]
    system = make_system(tmp_path)
    trace = system / "telemetry" / DAY / "trace" / "trace_span.csv"
    trace.write_text(
        TRACE_HEADER
        + f"\nJDBC,{START_MS},2.0,True,t0,s0,p0,docker_001,db_001,\n"
        + f"JDBC,{START_MS + 10},4.0,False,t0,s1,p0,docker_001,db_001,\n",
        encoding="utf-8",
    )
    matrix = _build(tmp_path, kpi_catalog)
    assert _at(matrix, "docker_001::trace_calls", 0) == 2.0
    assert _at(matrix, "docker_001::trace_error_rate", 0) == 0.5
    assert _at(matrix, "docker_001::trace_latency_mean", 0) == 3.0
    assert _at(matrix, "docker_001::trace_calls", 10) == 0.0


def test_trace_edges_follow_parent_span_and_database(tmp_path: Path) -> None:
    system = make_system(tmp_path)
    trace = system / "telemetry" / DAY / "trace" / "trace_span.csv"
    trace.write_text(
        TRACE_HEADER
        + f"\nCSF,{START_MS},5.0,True,t1,root,,os_021,,csf_001\n"
        + f"LOCAL,{START_MS + 1},2.0,True,t1,child,root,docker_001,,local_method_001\n"
        + f"JDBC,{START_MS + 2},1.0,True,t1,db,child,docker_001,db_001,\n",
        encoding="utf-8",
    )
    convert_day(system, "Sys", DAY, tmp_path / "interim")
    edges = {(s, d): w for s, d, w in trace_edges(tmp_path / "interim", "Sys", DAY).iter_rows()}
    assert edges == {("os_021", "docker_001"): 1, ("docker_001", "db_001"): 1}
