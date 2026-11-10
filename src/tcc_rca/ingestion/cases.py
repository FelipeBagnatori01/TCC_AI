"""Readers for OpenRCA ``record.csv`` and ``query.csv`` (agent-visible parts only).

``query.csv`` also carries the ground truth (``scoring_points``). That column is deliberately
NOT exposed here; it lives in :mod:`tcc_rca.ingestion.labels` so the LLM agent code can never
import it by accident (R3).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tcc_rca.ingestion.timeutil import epoch_s_to_utc, parse_local

_MONTHS = {
    name: number
    for number, name in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}
_DATE_RE = re.compile(r"(" + "|".join(_MONTHS) + r")\s+(\d{1,2}),\s+(\d{4})")
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_TASK_RE = re.compile(r"^task_([1-7])$")


@dataclass(frozen=True)
class Failure:
    """One labeled failure from ``record.csv`` (start instant in UTC)."""

    level: str
    reason: str
    component: str
    start_utc: datetime


@dataclass(frozen=True)
class Query:
    """One OpenRCA query: instruction text and its 30-minute window, no ground truth."""

    row: int  # zero-based row in query.csv (the official evaluator aligns by row)
    task_index: str
    instruction: str
    window_start_utc: datetime
    window_end_utc: datetime

    @property
    def task_number(self) -> int:
        match = _TASK_RE.match(self.task_index)
        if match is None:
            raise ValueError(f"unexpected task_index {self.task_index!r}")
        return int(match.group(1))


@dataclass(frozen=True)
class Case:
    query: Query
    failures: tuple[Failure, ...]


def parse_window(instruction: str) -> tuple[datetime, datetime]:
    """Extract the (UTC) window from an instruction written in UTC+8 natural language."""
    date_match = _DATE_RE.search(instruction)
    times = _TIME_RE.findall(instruction)
    if date_match is None or len(times) < 2:
        raise ValueError(f"cannot find date and time range in instruction: {instruction[:120]!r}")
    month, day, year = (
        _MONTHS[date_match.group(1)],
        int(date_match.group(2)),
        int(date_match.group(3)),
    )
    (h1, m1), (h2, m2) = (int(times[0][0]), int(times[0][1])), (int(times[1][0]), int(times[1][1]))
    local_start = datetime(year, month, day, h1, m1)
    local_end = datetime(year, month, day, h2, m2)
    if local_end <= local_start:  # range crossing midnight, e.g. 23:30 to 00:00
        local_end += timedelta(days=1)
    return parse_local(local_start.strftime("%Y-%m-%d %H:%M:%S")), parse_local(
        local_end.strftime("%Y-%m-%d %H:%M:%S")
    )


def read_failures(record_csv: Path) -> list[Failure]:
    """Read ``record.csv``. The epoch column and the UTC+8 text column must agree."""
    failures: list[Failure] = []
    with record_csv.open(encoding="utf-8", newline="") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            from_epoch = epoch_s_to_utc(int(row["timestamp"]))
            from_text = parse_local(row["datetime"])
            if from_epoch != from_text.astimezone(UTC):
                raise ValueError(
                    f"{record_csv}:{line_no}: timestamp {row['timestamp']} disagrees with "
                    f"datetime {row['datetime']!r} read as UTC+8"
                )
            failures.append(Failure(row["level"], row["reason"], row["component"], from_epoch))
    return failures


def read_queries(query_csv: Path) -> list[Query]:
    """Read ``query.csv`` keeping only fields the agent may see (no ``scoring_points``)."""
    queries: list[Query] = []
    with query_csv.open(encoding="utf-8", newline="") as handle:
        for row_no, row in enumerate(csv.DictReader(handle)):
            start, end = parse_window(row["instruction"])
            queries.append(Query(row_no, row["task_index"], row["instruction"], start, end))
    return queries


def build_cases(queries: list[Query], failures: list[Failure]) -> tuple[list[Case], list[Failure]]:
    """Attach each failure to the query whose half-open window ``[start, end)`` contains it.

    Returns the cases (same order as ``queries``) and the failures that fall in no window.
    """
    assigned: set[int] = set()
    cases: list[Case] = []
    for query in queries:
        inside = tuple(
            f for f in failures if query.window_start_utc <= f.start_utc < query.window_end_utc
        )
        assigned.update(id(f) for f in inside)
        cases.append(Case(query, inside))
    orphans = [f for f in failures if id(f) not in assigned]
    return cases, orphans
