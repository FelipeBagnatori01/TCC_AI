"""End-to-end reproduction (E9, ADR 0024): one command runs the thesis pipeline in order.

Levels:

* ``principal``: the chain behind the thesis results, per system (E1 conversion, E2 series,
  E3 tasks, E4 baselines, E5 LSTM + paired comparison, E6 XAI and evidence packages, ADR 0020
  evidence selection) and the RQ1 figure.
* ``agente-cache``: replays every agent test run (Opus E7, Haiku and Sonnet ADR 0023) from the
  shared LLM cache with a zero budget (no API key needed), then the E8 evaluation and the model
  comparisons. Needs ``principal`` first (the agent tools read the regenerated series and
  evidence packages) and the cache bundle restored (``tcc-rca cache-restore``).
* ``completo``: ``principal`` plus the E1 reports, every sensitivity control and the ADR 0018
  alternative models.

Every step is a separate process (LightGBM and multithreaded torch must not share one, ADR 0010)
and the first failure stops the run. Steps, timings and exit codes go to
``experiments/reproduce/<timestamp>-<level>/steps.jsonl``; ``tcc-rca reproduce-check`` then
compares the regenerated tables with the committed ones.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from tcc_rca.config import Config, load_config

LEVELS = ("principal", "agente-cache", "completo")
SYSTEMS = ("Telecom", "Bank")
SYSTEM_CONFIG = {"Telecom": None, "Bank": "configs/system/bank.yaml"}
EXP = "configs/experiment"
AGENT_MODELS = (  # (llm config, extra caps merged before the replay config)
    ("configs/llm/claude_opus_5_5.yaml", ()),
    ("configs/llm/claude_haiku_5_5.yaml", (f"{EXP}/agent_haiku_cap.yaml",)),
    ("configs/llm/claude_sonnet_5_5.yaml", (f"{EXP}/agent_sonnet_cap.yaml",)),
)


@dataclass(frozen=True)
class Step:
    command: str
    configs: tuple[str, ...] = ()  # merged after base.yaml and the system config
    system: str | None = None  # None: system-independent step (base.yaml only)
    args: tuple[str, ...] = field(default=())

    def argv(self, base: Sequence[str], extra: Sequence[str]) -> list[str]:
        """Config order: base, system, step configs, then the user's extra files (paths)."""
        sys_cfg = SYSTEM_CONFIG[self.system] if self.system else None
        files = [*base, *([sys_cfg] if sys_cfg else []), *self.configs, *extra]
        return [self.command, *self.args, "--config", *files]

    def label(self) -> str:
        names = " ".join(Path(c).stem for c in self.configs)
        return f"{self.command} [{self.system or 'all'}] {names}".strip()


def _principal(system: str) -> list[Step]:
    s = system
    return [
        Step("convert", system=s),
        Step("series", system=s),
        Step("tasks", system=s),
        Step("baselines", (f"{EXP}/baselines.yaml",), s),
        Step("lstm", (f"{EXP}/lstm.yaml",), s),
        Step("compare-e5", (f"{EXP}/lstm.yaml",), s),
        Step("xai", (f"{EXP}/xai.yaml",), s),
        Step("evidence-select", (f"{EXP}/evidence_select.yaml",), s),
    ]


def _controls(system: str) -> list[Step]:
    s = system
    return [
        Step("inventory", system=s),
        Step("cases", system=s),
        Step("quality", system=s),
        Step("similarity", system=s),
        Step("eda", system=s, args=("-n", "10")),
        Step("baselines", (f"{EXP}/baselines_sem_isencao.yaml",), s),
        Step("baselines", (f"{EXP}/baselines_janela_inteira.yaml",), s),
        Step("lstm", (f"{EXP}/lstm_sem_isencao.yaml",), s),
        Step("compare-e5", (f"{EXP}/lstm_sem_isencao.yaml",), s),
        Step("lstm", (f"{EXP}/lstm_janela_inteira.yaml",), s),
        Step("compare-e5", (f"{EXP}/lstm_janela_inteira.yaml",), s),
        Step("xai", (f"{EXP}/xai_sem_isencao.yaml",), s),
        Step("evidence-select", (f"{EXP}/evidence_select_sem_isencao.yaml",), s),
        Step("alt-models", (f"{EXP}/alt_tabular.yaml",), s),
        Step("alt-models", (f"{EXP}/alt_tcn.yaml",), s),
        Step("alt-models", (f"{EXP}/alt_tabular_sem_isencao.yaml",), s),
        Step("alt-models", (f"{EXP}/alt_tcn_sem_isencao.yaml",), s),
    ]


def _agent_cache(system: str) -> list[Step]:
    steps = [
        Step(
            "agent",
            (
                f"{EXP}/agent_a{v}.yaml",
                f"{EXP}/agent_1rep.yaml",
                llm,
                *caps,
                f"{EXP}/agent_a{v}_replay.yaml",  # zero budget: a cache miss stops the run
            ),
            system,
        )
        for llm, caps in AGENT_MODELS
        for v in (1, 2, 3)
    ]
    return [
        *steps,
        Step("evaluate-agent", (f"{EXP}/agent_eval.yaml",), system),
        Step("compare-models", (f"{EXP}/model_compare.yaml",), system),
        Step("compare-models", (f"{EXP}/model_compare_sonnet.yaml",), system),
    ]


def plan(level: str, systems: Sequence[str] = SYSTEMS) -> list[Step]:
    """Ordered steps of a level; the data check comes first, the RQ1 figure last."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; use one of {LEVELS}")
    bad = sorted(set(systems) - set(SYSTEMS))
    if bad:
        raise ValueError(f"unknown systems {bad}; use {SYSTEMS}")
    if level == "agente-cache":
        return [s for sys_ in systems for s in _agent_cache(sys_)]
    steps = [Step("verify", args=("--quick",))]
    for sys_ in systems:
        steps += _principal(sys_)
        if level == "completo":
            steps += _controls(sys_)
    return [*steps, Step("rq1-curves")]


RELEASE = "cache-llm-2026-11-23"


def preflight(
    config: Config,
    systems: Sequence[str],
    level: str = "principal",
    tables_dir: Path = Path("reports/tables"),
) -> list[str]:
    """What is missing before any step runs, each with the fix (messages in Portuguese, like the
    README they point to). Empty when the data and the OpenRCA scorer are in place and, for
    ``agente-cache``, when the cache bundle has been restored: without it every replay would
    stop at its first call and write empty tables over the committed ones."""
    problems = []
    root = config.paths.data_root
    for system in systems:
        if not (root / system / "query.csv").is_file():
            problems.append(
                f"dados do {system} não encontrados em {root / system}: crie a pasta e os links "
                f"(mkdir -p {root}; ln -s /caminho/{system} {root / system}) ou aponte "
                "paths.data_root em configs/local.yaml (README, seção Dados)"
            )
    if "Telecom" in systems and config.series.deployment_list is not None:
        deployment = root / config.series.deployment_list
        if not deployment.is_file():
            problems.append(
                f"{deployment} não encontrado: ligue AppTelecomDeploymentList.xlsx em {root}"
            )
    scorer = config.paths.openrca_repo / "main" / "evaluate.py"
    if not scorer.is_file():
        problems.append(
            f"submódulo do OpenRCA vazio ({config.paths.openrca_repo}): rode "
            "git submodule update --init (o clone precisa de --recurse-submodules)"
        )
    if level == "agente-cache":
        from tcc_rca.bundle import agent_runs

        exp = config.paths.experiments_dir
        cache = config.llm.cache_dir
        missing = [
            r for r in agent_runs(tables_dir) if not (exp / Path(r).name / "artifacts").is_dir()
        ]
        if not cache.is_dir() or not any(cache.iterdir()) or missing:
            problems.append(
                f"pacote do cache não restaurado (cache em {cache}"
                + (
                    f"; faltam {len(missing)} execuções originais do agente em {exp}"
                    if missing
                    else ""
                )
                + f"): baixe tcc_rca_cache_llm.tar.gz da Release {RELEASE} e rode "
                "uv run tcc-rca cache-restore --config <as mesmas configs> "
                "--bundle /caminho/para/tcc_rca_cache_llm.tar.gz (README, seção Reprodução)"
            )
    return problems


def run(
    level: str,
    base: Sequence[str],
    extra: Sequence[str],
    systems: Sequence[str],
    out_root: Path,
    dry_run: bool = False,
) -> Path:
    """Runs the steps in order (separate processes); stops at the first failure. Checks the data
    and the OpenRCA submodule first, so a missing link fails at once with the fix."""
    steps = plan(level, systems)
    if not dry_run:
        problems = preflight(load_config(*base, *extra), systems, level)
        if problems:
            raise SystemExit("Antes de reproduzir:\n- " + "\n- ".join(problems))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out = out_root / f"{stamp}-{level}"
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "level": level,
        "systems": list(systems),
        "base": list(base),
        "extra": list(extra),
        "started_at": time.time(),
        "commit": _commit(),
        "steps": [s.label() for s in steps],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log = out / "steps.jsonl"
    for i, step in enumerate(steps, 1):
        argv = step.argv(base, extra)
        print(f"[{i}/{len(steps)}] {step.label()}", flush=True)
        record: dict[str, object] = {"step": i, "label": step.label(), "argv": argv}
        if dry_run:
            print("  tcc-rca " + " ".join(argv))
            continue
        start = time.perf_counter()
        with (out / f"step_{i:02d}.log").open("w", encoding="utf-8") as handle:
            rc = subprocess.call(
                [sys.executable, "-m", "tcc_rca.cli", *argv],
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        record |= {"returncode": rc, "elapsed_s": round(time.perf_counter() - start, 1)}
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if rc != 0:
            raise RuntimeError(f"step {i} failed ({step.label()}); see {out}/step_{i:02d}.log")
    return out


def latest_runs(root: Path) -> list[Path]:
    """The latest reproduce dir of each level (``principal`` and ``agente-cache`` are separate
    runs, so checking only the newest would count the other level's tables as not regenerated)."""
    latest: dict[str, Path] = {}
    for d in sorted(p for p in root.iterdir() if (p / "manifest.json").is_file()):
        latest[json.loads((d / "manifest.json").read_text(encoding="utf-8"))["level"]] = d
    if not latest:
        raise SystemExit(f"nenhuma reprodução em {root}: rode make reproduce antes")
    return sorted(latest.values())


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
