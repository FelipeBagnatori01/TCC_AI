"""Deployment relations from the OpenRCA Telecom ``AppTelecomDeploymentList.xlsx``.

Layout found in the file (E2 inventory): columns ``Type, Number, Host, Name, From (containers)``.
``Type`` and ``Number`` appear only on the first row of each group (``Number`` = group size).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openpyxl import load_workbook

EXPECTED_HEADER = ("Type", "Number", "Host", "Name", "From (containers)")


@dataclass(frozen=True)
class DeployedComponent:
    name: str
    group: str  # value of the "Type" column for the component's group
    host: str | None


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    relation: str  # calls | hosted_on | member_of | runs_in
    source: str  # traces | deployment_list
    weight: int | None = None  # number of observed calls for trace edges


def _cell(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def read_deployment(path: Path) -> tuple[list[DeployedComponent], list[Edge]]:
    """Parse components and deployment edges, validating group sizes against ``Number``."""
    sheet = load_workbook(path, read_only=True).active
    if sheet is None:
        raise ValueError(f"{path}: workbook has no active sheet")
    rows = list(sheet.iter_rows(values_only=True))
    header = tuple(_cell(c) for c in rows[0][: len(EXPECTED_HEADER)])
    if header != EXPECTED_HEADER:
        raise ValueError(f"{path}: header {header} differs from expected {EXPECTED_HEADER}")

    components: list[DeployedComponent] = []
    runs_in: list[tuple[str, str]] = []
    group_sizes: dict[str, int] = {}
    group: str | None = None
    for row in rows[1:]:
        type_, number, host, name, from_ = (_cell(c) for c in row[: len(EXPECTED_HEADER)])
        if name is None:
            continue
        if type_ is not None:
            group = type_
            if number is None:
                raise ValueError(f"{path}: group {type_!r} without Number")
            group_sizes[group] = int(float(number))
        if group is None:
            raise ValueError(f"{path}: component {name!r} before any group Type")
        components.append(DeployedComponent(name, group, host))
        if from_ is not None:
            runs_in.append((name, from_))

    counts: dict[str, int] = {}
    for c in components:
        counts[c.group] = counts.get(c.group, 0) + 1
    if counts != group_sizes:
        raise ValueError(f"{path}: group sizes {counts} disagree with Number {group_sizes}")

    groups = set(group_sizes)
    edges: list[Edge] = []
    for c in components:
        if c.host is not None:
            edges.append(Edge(c.name, c.host, "hosted_on", "deployment_list"))
    for c in components:  # members of a container group, e.g. docker_001 -> container_001
        if c.group in {dst for _, dst in runs_in}:
            edges.append(Edge(c.name, c.group, "member_of", "deployment_list"))
    for name, dst in runs_in:  # e.g. csf_001 runs in container_001
        if dst not in groups:
            raise ValueError(f"{path}: {name!r} runs in unknown group {dst!r}")
        edges.append(Edge(name, dst, "runs_in", "deployment_list"))
    return components, edges
