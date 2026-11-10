from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from synthetic import INSTRUCTION, make_system
from tcc_rca.ingestion.cases import (
    Failure,
    Query,
    build_cases,
    parse_window,
    read_failures,
    read_queries,
)
from tcc_rca.ingestion.labels import read_scoring_points


def test_window_is_30_minutes_and_converted_to_utc() -> None:
    start, end = parse_window(INSTRUCTION)
    assert start == datetime(2020, 4, 10, 16, 0, tzinfo=UTC)  # 00:00 UTC+8
    assert end - start == timedelta(minutes=30)


def test_window_crossing_midnight() -> None:
    start, end = parse_window("On May 3, 2020, between 23:30 and 00:00, a failure occurred.")
    assert end - start == timedelta(minutes=30)
    assert start == datetime(2020, 5, 3, 15, 30, tzinfo=UTC)


def test_unparseable_instruction_fails_loudly() -> None:
    with pytest.raises(ValueError, match="cannot find"):
        parse_window("no date here")


def test_record_label_falls_inside_query_window(tmp_path: Path) -> None:
    system = make_system(tmp_path)
    queries = read_queries(system / "query.csv")
    failures = read_failures(system / "record.csv")
    cases, orphans = build_cases(queries, failures)
    assert orphans == []
    assert len(cases) == 1
    assert len(cases[0].failures) == 1
    failure = cases[0].failures[0]
    assert cases[0].query.window_start_utc <= failure.start_utc < cases[0].query.window_end_utc
    assert failure.component == "docker_001"


def test_failure_on_window_end_belongs_to_next_window_and_orphans_are_reported() -> None:
    t0 = datetime(2020, 4, 10, 16, 0, tzinfo=UTC)
    query = Query(0, "task_1", "x", t0, t0 + timedelta(minutes=30))
    on_end = Failure("pod", "CPU fault", "docker_001", t0 + timedelta(minutes=30))
    inside = Failure("pod", "CPU fault", "docker_002", t0)
    cases, orphans = build_cases([query], [inside, on_end])
    assert [f.component for f in cases[0].failures] == ["docker_002"]
    assert orphans == [on_end]


def test_inconsistent_epoch_and_datetime_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "record.csv"
    # datetime says 00:05 UTC+8 but epoch is 8 hours off (the classic missing-timezone bug)
    bad.write_text(
        "level,reason,component,timestamp,datetime\n"
        "pod,CPU fault,docker_001,1586563500,2020-04-11 00:05:00\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disagrees"):
        read_failures(bad)


def test_agent_visible_query_does_not_carry_ground_truth(tmp_path: Path) -> None:
    system = make_system(tmp_path)
    query = read_queries(system / "query.csv")[0]
    assert not hasattr(query, "scoring_points")
    assert "CPU fault" not in query.instruction  # the instruction never names the answer
    assert read_scoring_points(system / "query.csv")[0].startswith("The only predicted")


def test_agent_package_never_imports_label_module(repo_root: Path) -> None:
    offenders: list[str] = []
    for path in (repo_root / "src/tcc_rca/agent").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if any("labels" in n or "scoring_points" in n for n in names):
                offenders.append(str(path))
    assert offenders == []
