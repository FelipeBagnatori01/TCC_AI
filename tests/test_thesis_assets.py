"""E9: monograph tables/figures are generated from the versioned result tables only."""

from __future__ import annotations

import csv
from pathlib import Path

from docx import Document

from tcc_rca.thesis_assets import NUMBERS, build, ci, num


def test_number_formatting_uses_decimal_comma_and_minus() -> None:
    assert num(0.5163) == "0,52"
    assert num(-0.0182, 3) == chr(0x2212) + "0,018"
    assert ci({"value": 0.6, "ci_low": 0.46, "ci_high": 0.72}) == "0,60 [0,46; 0,72]"


def test_build_traces_every_item_to_versioned_tables(repo_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "monografia"
    summary = build(repo_root / "reports/tables", repo_root / "reports/figures", out)
    with (out / "MANIFEST.csv").open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter=";"))
    assert len(manifest) == summary["tables"] + summary["figures"]
    for row in manifest:
        assert (out / row["arquivo"]).is_file()
        assert row["origem"], row["titulo"]
        for src in row["origem"].split(" "):
            if src.startswith("reports/"):
                assert (repo_root / src).is_file(), src
    with (out / "numeros.csv").open(encoding="utf-8-sig", newline="") as handle:
        numbers = list(csv.DictReader(handle, delimiter=";"))
    assert len(numbers) == len(NUMBERS)
    assert all((repo_root / n["arquivo"]).is_file() for n in numbers)
    doc = Document(str(out / "tabelas.docx"))
    captions = [p.text for p in doc.paragraphs if p.text.startswith("Tabela ")]
    assert len(captions) == summary["tables"]
    assert all(f" {chr(0x2013)} " in c for c in captions)  # caption: number, dash, title
    assert sum(p.text.startswith("Fonte:") for p in doc.paragraphs) == summary["tables"]
