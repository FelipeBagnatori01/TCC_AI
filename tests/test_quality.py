from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from synthetic import DAY, make_system
from tcc_rca.ingestion.cases import Failure, build_cases, read_failures, read_queries
from tcc_rca.ingestion.eda import plot_failure
from tcc_rca.ingestion.quality import build_quality_report, write_quality_report
from tcc_rca.ingestion.telemetry import convert_day


def _prepare(tmp_path: Path):  # type: ignore[no-untyped-def]
    system = make_system(tmp_path)
    interim = tmp_path / "interim"
    convert_day(system, "Sys", DAY, interim)
    cases, orphans = build_cases(
        read_queries(system / "query.csv"), read_failures(system / "record.csv")
    )
    return interim, cases, orphans


def test_quality_report_finds_constant_kpi_and_covered_failure(tmp_path: Path) -> None:
    interim, cases, orphans = _prepare(tmp_path)
    report = build_quality_report(interim, "Sys", cases, len(orphans))
    constants = report["series"]["metric_container"]["constant_series"]
    assert constants == [{"cmdb_id": "docker_001", "name": "container_fgc"}]
    assert report["labeled_components_without_any_data"] == []
    assert report["failures_without_metric_coverage"] == []
    row = report["window_coverage"][0]
    assert row["metric_rows_before_30min"] > 0
    assert row["metric_rows_after_30min"] > 0
    assert row["trace_rows_after_30min"] > 0


def test_component_without_data_is_listed_explicitly(tmp_path: Path) -> None:
    interim, cases, _ = _prepare(tmp_path)
    ghost = Failure("pod", "CPU fault", "docker_999", cases[0].failures[0].start_utc)
    from tcc_rca.ingestion.cases import Case

    report = build_quality_report(interim, "Sys", [Case(cases[0].query, (ghost,))], 0)
    assert report["labeled_components_without_any_data"] == ["docker_999"]
    assert len(report["failures_without_metric_coverage"]) == 1


def test_report_is_written_as_json_and_markdown(tmp_path: Path) -> None:
    interim, cases, orphans = _prepare(tmp_path)
    report = build_quality_report(interim, "Sys", cases, len(orphans))
    write_quality_report(report, tmp_path / "q.json", tmp_path / "q.md")
    assert "Relatório de qualidade" in (tmp_path / "q.md").read_text(encoding="utf-8")
    assert (tmp_path / "q.json").stat().st_size > 0


def test_eda_plot_written_only_when_component_has_data(tmp_path: Path) -> None:
    interim, cases, _ = _prepare(tmp_path)
    failure = cases[0].failures[0]
    out = tmp_path / "fig.png"
    assert plot_failure(interim, "Sys", failure, out)
    assert out.stat().st_size > 0
    ghost = Failure("pod", "x", "docker_999", datetime(2020, 4, 10, 16, 5, tzinfo=UTC))
    assert not plot_failure(interim, "Sys", ghost, tmp_path / "none.png")
