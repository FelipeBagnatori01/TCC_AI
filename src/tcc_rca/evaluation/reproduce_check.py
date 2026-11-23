"""Compares regenerated result tables with the committed ones (E9, ADR 0024).

A table counts as regenerated when it was written after the reproduction run started. Run
directory names (``<timestamp>-<name>``) and wall-clock fields differ by construction and are
ignored; every other value must match: strings, integers and booleans exactly, floats within
``rtol``/``atol`` (0 by default: same platform; ADR 0024 fixes the tolerance for other hardware).
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

RUN_DIR = re.compile(r"\d{8}T\d{6}-[A-Za-z0-9_.-]+")
IGNORED_KEYS = frozenset({"elapsed_s", "peak_rss_mb", "started_at", "latency_s", "latency_mean_s"})
MAX_LISTED = 20


def _norm(text: str) -> str:
    return RUN_DIR.sub("<run>", text)


def _as_float(x: Any) -> float | None:
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, int | float):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def _close(a: float, b: float, rtol: float, atol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def diff_values(ref: Any, new: Any, rtol: float, atol: float, path: str = "") -> Iterator[str]:
    """Paths where ``new`` departs from ``ref`` (JSON-like values)."""
    if isinstance(ref, dict) and isinstance(new, dict):
        for key in sorted(set(ref) | set(new)):
            if key in IGNORED_KEYS:
                continue
            if key not in ref or key not in new:
                yield f"{path}/{key}: present in only one side"
                continue
            yield from diff_values(ref[key], new[key], rtol, atol, f"{path}/{key}")
    elif isinstance(ref, list) and isinstance(new, list):
        if len(ref) != len(new):
            yield f"{path}: length {len(ref)} != {len(new)}"
            return
        for i, (a, b) in enumerate(zip(ref, new, strict=True)):
            yield from diff_values(a, b, rtol, atol, f"{path}[{i}]")
    else:
        fa, fb = _as_float(ref), _as_float(new)
        strings = isinstance(ref, str) and isinstance(new, str)
        if isinstance(ref, int) and isinstance(new, int):  # counts: exact (bools included)
            if ref != new:
                yield f"{path}: {ref} != {new}"
        elif fa is not None and fb is not None and not strings:
            if not _close(fa, fb, rtol, atol):
                yield f"{path}: {ref} != {new}"
        elif strings:
            if _norm(ref) != _norm(new):
                if fa is not None and fb is not None and _close(fa, fb, rtol, atol):
                    return
                yield f"{path}: {ref!r} != {new!r}"
        elif ref != new:
            yield f"{path}: {ref!r} != {new!r}"


def _load(text: str, suffix: str) -> Any:
    if suffix == ".json":
        return json.loads(text)
    return list(csv.reader(io.StringIO(text)))


def committed(path: Path, ref: str) -> str | None:
    """File content at a git revision (None when the file is not tracked there)."""
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{path.as_posix()}"], text=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return None


def check(
    tables_dir: Path, since: float, ref: str = "HEAD", rtol: float = 0.0, atol: float = 0.0
) -> dict[str, Any]:
    """Status of every JSON/CSV table: regenerated after ``since`` or not, and equal or not."""
    files = sorted(p for p in tables_dir.iterdir() if p.suffix in (".json", ".csv"))
    out: dict[str, Any] = {"ref": ref, "rtol": rtol, "atol": atol, "files": {}}
    for p in files:
        if p.stat().st_mtime < since:
            out["files"][p.name] = {"status": "nao_regenerado"}
            continue
        old = committed(p, ref)
        if old is None:
            out["files"][p.name] = {"status": "novo"}
            continue
        diffs = list(
            diff_values(
                _load(old, p.suffix), _load(p.read_text(encoding="utf-8"), p.suffix), rtol, atol
            )
        )
        out["files"][p.name] = (
            {"status": "igual"}
            if not diffs
            else {"status": "diferente", "n_diffs": len(diffs), "diffs": diffs[:MAX_LISTED]}
        )
    counts: dict[str, int] = {}
    for item in out["files"].values():
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    out["counts"] = counts
    out["ok"] = counts.get("diferente", 0) == 0 and counts.get("igual", 0) > 0
    return out


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Verificação da reprodução (E9)",
        "",
        f"Referência: `{result['ref']}`; tolerância rtol = {result['rtol']}, "
        f"atol = {result['atol']}.",
        "",
        "| Situação | Arquivos |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(result["counts"].items())],
        "",
        f"**Resultado:** {'aprovado' if result['ok'] else 'reprovado'}.",
    ]
    differing = {k: v for k, v in result["files"].items() if v["status"] == "diferente"}
    if differing:
        lines += ["", "## Diferenças", ""]
        for name, item in differing.items():
            lines.append(f"- `{name}` ({item['n_diffs']}):")
            lines += [f"  - `{d}`" for d in item["diffs"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
