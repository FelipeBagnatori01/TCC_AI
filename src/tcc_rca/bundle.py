"""Bundle of the LLM cache for third-party reproduction (E9, ADR 0024).

The bundle is published as a GitHub Release asset, never committed (it holds excerpts of OpenRCA
telemetry inside the tool outputs, CC BY-NC 4.0). It contains:

* ``experiments/llm_cache/``: every cached model call;
* ``experiments/<run>/``: the original agent test runs (Opus E7, Haiku and Sonnet ADR 0023), which
  ``evaluate-agent``/``compare-models`` score and compare the replays with;
* ``referencia/evidence/<system>/<name>/``: the evidence packages the agent saw, to tell an
  evidence difference apart from any other cache miss;
* ``MANIFEST.json`` (SHA-256 of every file) and ``LEIA-ME.txt`` (content and license).

Building refuses any file that looks like an API key.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any

from tcc_rca.config import Config

AGENT_TABLE = re.compile(r"agent_(Telecom|Bank)_agent_a[123]_(claude-[a-z0-9-]+)\.json")
SECRET = re.compile(rb"sk-ant-[A-Za-z0-9_-]{16,}|sk-proj-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9]{32,}")
REFERENCE = "referencia/evidence"
README = """Pacote de reprodução do TCC (cache do LLM e execuções do agente)

Conteúdo:
- experiments/llm_cache/: todas as chamadas ao modelo de LLM (pedido e resposta), em cache;
- experiments/<execução>/: as execuções de teste do agente (Claude Opus 5.5, Haiku 5.5 e
  Sonnet 5.5), com previsões, transcrições e registro de chamadas;
- referencia/evidence/: os pacotes de evidência LSTM/XAI que o agente recebeu.

Uso: tcc-rca cache-restore --config configs/base.yaml <este arquivo>; depois
tcc-rca reproduce --level agente-cache (ver README.md do repositório). Nenhuma chave de API é
necessária: as chamadas vêm do cache, com teto de custo zero.

Licença: as saídas das ferramentas contêm trechos da telemetria do OpenRCA (XU, J. et al.
OpenRCA: Can Large Language Models Locate the Root Cause of Software Failures? ICLR, 2025),
distribuída sob CC BY-NC 4.0: uso apenas não comercial, com citação. As respostas foram geradas
por modelos Claude (Anthropic).
"""


def agent_runs(tables_dir: Path) -> list[str]:
    """Run directories of the agent test runs recorded in the result tables."""
    runs = set()
    for p in sorted(tables_dir.glob("agent_*_agent_a*_claude-*.json")):
        if AGENT_TABLE.fullmatch(p.name):
            runs.add(json.loads(p.read_text(encoding="utf-8"))["run_dir"])
    return sorted(runs)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _members(config: Config, tables_dir: Path) -> list[tuple[Path, str]]:
    exp = config.paths.experiments_dir
    pairs: list[tuple[Path, str]] = []
    roots = [(config.llm.cache_dir, "experiments/llm_cache")]
    roots += [(exp / run, f"experiments/{run}") for run in agent_runs(tables_dir)]
    name = config.agent.evidence_name
    for system in ("Telecom", "Bank"):
        roots.append(
            (
                config.paths.processed_dir / system / "evidence" / name,
                f"{REFERENCE}/{system}/{name}",
            )
        )
    for src, arc in roots:
        if not src.is_dir():
            raise FileNotFoundError(f"missing {src}: the bundle needs every agent run")
        pairs += [
            (f, f"{arc}/{f.relative_to(src).as_posix()}")
            for f in sorted(src.rglob("*"))
            if f.is_file()
        ]
    return pairs


def make_bundle(config: Config, tables_dir: Path, out: Path) -> dict[str, Any]:
    pairs = _members(config, tables_dir)
    manifest: dict[str, str] = {}
    for src, arc in pairs:
        data = src.read_bytes()
        if SECRET.search(data):
            raise ValueError(f"{src} looks like it holds an API key; not bundling it")
        manifest[arc] = hashlib.sha256(data).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for name, text in (
            ("LEIA-ME.txt", README),
            ("MANIFEST.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n"),
        ):
            blob = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
        for src, arc in pairs:
            tar.add(src, arcname=arc, recursive=False)
    return {
        "bundle": str(out),
        "files": len(pairs),
        "runs": len(agent_runs(tables_dir)),
        "size_mb": round(out.stat().st_size / 2**20, 1),
        "sha256": _sha256(out),
    }


def restore_bundle(config: Config, bundle: Path, use_reference_evidence: bool) -> dict[str, Any]:
    """Extracts into the experiments dir after checking every checksum; never overwrites a
    different existing file. ``use_reference_evidence`` copies the bundled evidence packages over
    the regenerated ones (the regenerated ones are kept under ``<name>_regenerado``)."""
    exp = config.paths.experiments_dir
    staging = exp / "reproduce" / "pacote"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with tarfile.open(bundle, "r:gz") as tar:
        tar.extractall(staging, filter="data")
    manifest = json.loads((staging / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [arc for arc, h in manifest.items() if _sha256(staging / arc) != h]
    if bad:
        raise ValueError(f"{len(bad)} files fail the checksum, e.g. {bad[:3]}")
    copied = skipped = 0
    for arc in manifest:
        if not arc.startswith("experiments/"):
            continue
        dest = exp / arc.removeprefix("experiments/")
        if dest.exists():
            if _sha256(dest) != manifest[arc]:
                raise FileExistsError(f"{dest} exists with different content; not overwriting")
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staging / arc, dest)
        copied += 1
    out: dict[str, Any] = {"files": len(manifest), "copied": copied, "already_present": skipped}
    out["evidence"] = compare_evidence(config, staging / REFERENCE)
    if use_reference_evidence:
        for system, status in out["evidence"].items():
            if status["equal"]:
                continue
            target = config.paths.processed_dir / system / "evidence" / config.agent.evidence_name
            backup = target.with_name(target.name + "_regenerado")
            if target.exists() and not backup.exists():
                target.rename(backup)
            shutil.copytree(staging / REFERENCE / system / config.agent.evidence_name, target)
            status["replaced_by_reference"] = True
    return out


def compare_evidence(config: Config, reference_root: Path) -> dict[str, Any]:
    """Regenerated evidence packages vs. the ones the agent saw (byte equality per file)."""
    name = config.agent.evidence_name
    out: dict[str, Any] = {}
    for system in ("Telecom", "Bank"):
        ref = reference_root / system / name
        cur = config.paths.processed_dir / system / "evidence" / name
        ref_files = {p.name: _sha256(p) for p in sorted(ref.glob("*.json"))}
        cur_files = {p.name: _sha256(p) for p in sorted(cur.glob("*.json"))} if cur.is_dir() else {}
        differing = sorted(k for k in ref_files if cur_files.get(k) != ref_files[k])
        out[system] = {
            "files": len(ref_files),
            "equal": not differing and set(cur_files) == set(ref_files),
            "differing": differing[:10],
        }
    return out
