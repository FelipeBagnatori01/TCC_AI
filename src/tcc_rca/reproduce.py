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


def run(
    level: str,
    base: Sequence[str],
    extra: Sequence[str],
    systems: Sequence[str],
    out_root: Path,
    dry_run: bool = False,
) -> Path:
    """Runs the steps in order (separate processes); stops at the first failure."""
    steps = plan(level, systems)
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


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
