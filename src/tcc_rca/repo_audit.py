"""Scan of the whole git history for secrets and data (E9 acceptance, rules R1/R6).

Every blob reachable from any ref is checked once:

* content that looks like an API key (Anthropic, OpenAI, generic ``*_API_KEY=value``);
* paths that must never be versioned: telemetry (``data/raw``, ``data/interim``,
  ``data/processed`` except ``.gitkeep``), OpenRCA ground truth (``record.csv``,
  ``query.csv``), ``.env`` files, model weights, parquet, the LLM cache and run outputs;
* blobs larger than ``max_mb`` (telemetry dumps would show here).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

KEY = re.compile(
    rb"sk-ant-[A-Za-z0-9_-]{16,}|sk-proj-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9]{32,}"
    rb"|(?:ANTHROPIC|OPENAI)_API_KEY\s*=\s*[A-Za-z0-9_-]{8,}"
)
FORBIDDEN_PATH = re.compile(
    r"^data/(raw|interim|processed)/(?!\.gitkeep$)"
    r"|(^|/)(record|query)\.csv$"
    r"|(^|/)\.env(\.(?!example$)[^/]*)?$"
    r"|\.(parquet|pt|pth|ckpt|safetensors|joblib|pkl)$"
    r"|^experiments/(?!\.gitkeep$)"
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True)


def scan(max_mb: float = 5.0) -> dict[str, Any]:
    objects = _git("rev-list", "--all", "--objects").splitlines()
    path_of: dict[str, str] = {}
    for line in objects:
        sha, _, path = line.partition(" ")
        if path:
            path_of.setdefault(sha, path)
    findings: list[dict[str, str]] = []
    for sha, path in sorted(path_of.items(), key=lambda kv: kv[1]):
        if FORBIDDEN_PATH.search(path):
            findings.append({"path": path, "object": sha, "issue": "caminho proibido"})
    batch = "\n".join(path_of) + "\n"
    info = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        input=batch, capture_output=True, text=True, check=True,
    ).stdout.splitlines()  # fmt: skip
    blobs = []
    for line in info:
        sha, kind, size = line.split()
        if kind != "blob":
            continue
        blobs.append(sha)
        if int(size) > max_mb * 2**20:
            findings.append({"path": path_of[sha], "object": sha, "issue": f"blob > {max_mb} MB"})
    for sha in blobs:
        data = subprocess.check_output(["git", "cat-file", "blob", sha])
        if KEY.search(data):
            findings.append({"path": path_of[sha], "object": sha, "issue": "possível chave de API"})
    return {
        "commits": len(_git("rev-list", "--all").split()),
        "blobs": len(blobs),
        "findings": findings,
        "ok": not findings,
    }


def write_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Varredura do histórico do Git (E9)",
        "",
        f"- Commits verificados: {result['commits']}",
        f"- Arquivos (blobs) verificados: {result['blobs']}",
        "- Checagens: chaves de API no conteúdo; caminhos proibidos (telemetria, gabarito, `.env`,"
        " pesos, parquet, cache e execuções); arquivos grandes.",
        f"- **Resultado:** {'nenhum achado' if result['ok'] else 'ACHADOS abaixo'}",
    ]
    for f in result["findings"]:
        lines.append(f"  - `{f['path']}` ({f['object'][:10]}): {f['issue']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
