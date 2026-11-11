from __future__ import annotations

from pathlib import Path

import pytest
from openpyxl import Workbook

from tcc_rca.series.deployment import Edge, read_deployment

HEADER = ["Type", "Number", "Host", "Name", "From (containers)"]


def _xlsx(path: Path, rows: list[list[object]]) -> Path:
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.append(HEADER)
    for row in rows:
        ws.append(row)
    wb.save(path)
    return path


ROWS: list[list[object]] = [
    ["CSF", 2, None, "csf_001", "container_001"],
    [None, None, None, "csf_002"],
    ["container_001", 2, "os_017", "docker_001"],
    [None, None, "os_018", "docker_002"],
    ["Virtual Machine", 2, None, "os_017"],
    [None, None, None, "os_018"],
]


def test_components_and_edges(tmp_path: Path) -> None:
    components, edges = read_deployment(_xlsx(tmp_path / "d.xlsx", ROWS))
    assert [(c.name, c.group, c.host) for c in components][2] == (
        "docker_001",
        "container_001",
        "os_017",
    )
    assert set(edges) == {
        Edge("docker_001", "os_017", "hosted_on", "deployment_list"),
        Edge("docker_002", "os_018", "hosted_on", "deployment_list"),
        Edge("docker_001", "container_001", "member_of", "deployment_list"),
        Edge("docker_002", "container_001", "member_of", "deployment_list"),
        Edge("csf_001", "container_001", "runs_in", "deployment_list"),
    }


def test_group_size_must_match_number(tmp_path: Path) -> None:
    bad = [list(r) for r in ROWS]
    bad[0][1] = 3
    with pytest.raises(ValueError, match="disagree with Number"):
        read_deployment(_xlsx(tmp_path / "d.xlsx", bad))


def test_unexpected_header_fails(tmp_path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.append(["Kind", "Count", "Host", "Name", "From"])
    wb.save(tmp_path / "d.xlsx")
    with pytest.raises(ValueError, match="header"):
        read_deployment(tmp_path / "d.xlsx")


def test_real_catalog_covers_categories(repo_root: Path) -> None:
    from tcc_rca.series.catalog import CATEGORIES, load_kpi_catalog

    catalog = load_kpi_catalog(repo_root / "configs/data/kpi_catalog_telecom.yaml")
    used = {info.category for kpis in catalog.values() for info in kpis.values()}
    assert used <= set(CATEGORIES)
    assert sum(len(k) for k in catalog.values()) == 138
