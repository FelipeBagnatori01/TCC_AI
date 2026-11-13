"""What each OpenRCA query asks for, read from agent-visible text only (ADR 0017).

* fields per ``task_index`` (task_1 = occurrence time ... task_7 = all three), as defined by the
  OpenRCA benchmark;
* number of failures, stated in the instruction ("two failures", "two system failures"...);
  one otherwise. Both were checked against every Telecom and Bank query (ADR 0017), but the
  check itself lives outside this package: the agent never reads the scoring points.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from tcc_rca.ingestion.timeutil import utc_to_local_str

FIELDS = ("occurrence_datetime", "component", "reason")
_TASK_FIELDS: dict[str, tuple[str, ...]] = {
    "task_1": ("occurrence_datetime",),
    "task_2": ("reason",),
    "task_3": ("component",),
    "task_4": ("occurrence_datetime", "reason"),
    "task_5": ("occurrence_datetime", "component"),
    "task_6": ("component", "reason"),
    "task_7": ("occurrence_datetime", "component", "reason"),
}
_COUNT_RE = re.compile(r"\b(two|three|four|2|3|4)\s+(?:[a-z-]+\s+){0,2}failures\b", re.IGNORECASE)
_COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "2": 2, "3": 3, "4": 4}


def failure_count(instruction: str) -> int:
    match = _COUNT_RE.search(instruction)
    return _COUNT_WORDS[match.group(1).lower()] if match else 1


def required_fields(task_index: str) -> tuple[str, ...]:
    try:
        return _TASK_FIELDS[task_index]
    except KeyError:
        raise ValueError(f"unknown OpenRCA task_index {task_index!r}") from None


@dataclass(frozen=True)
class TaskSpec:
    system: str
    query_row: int
    task_index: str
    instruction: str
    window_start_utc: datetime
    window_end_utc: datetime
    n_failures: int
    fields: tuple[str, ...]

    @property
    def window_local(self) -> tuple[str, str]:
        return utc_to_local_str(self.window_start_utc), utc_to_local_str(self.window_end_utc)

    @property
    def day(self) -> str:
        """Matrix day id (``YYYY_MM_DD``) of the query: its UTC+8 calendar date."""
        return self.window_local[0][:10].replace("-", "_")


def task_spec(
    system: str,
    query_row: int,
    task_index: str,
    instruction: str,
    window_start_utc: datetime,
    window_end_utc: datetime,
) -> TaskSpec:
    return TaskSpec(
        system=system,
        query_row=query_row,
        task_index=task_index,
        instruction=instruction,
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
        n_failures=failure_count(instruction),
        fields=required_fields(task_index),
    )


@dataclass(frozen=True)
class Candidates:
    """Possible root-cause components and reasons, as the OpenRCA agent prompts list them."""

    components: tuple[str, ...]
    reasons: tuple[str, ...]


def load_candidates(prompts_dir: Path, system: str) -> Candidates:
    data = yaml.safe_load((prompts_dir / f"candidates_{system}.yaml").read_text(encoding="utf-8"))
    return Candidates(tuple(data["components"]), tuple(data["reasons"]))
