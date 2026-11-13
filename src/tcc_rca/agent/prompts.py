"""Versioned prompt templates (``configs/prompts/``), rendered with ``string.Template`` (R6).

No few-shot examples, and nothing from the test set: the only case-specific content is the
agent-visible query (instruction text, window, number of failures, requested fields) and the
public candidate lists of the benchmark (R3).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml

from tcc_rca.agent.tasks import Candidates, TaskSpec

_FIELD_PT = {"occurrence_datetime": "instante", "component": "componente", "reason": "motivo"}
TEMPLATE_FILES = (
    "agent_system.md",
    "agent_user.md",
    "agent_retry.md",
    "agent_variant_A1.md",
    "agent_variant_A2.md",
    "agent_variant_A3.md",
    "systems.yaml",
)


@dataclass(frozen=True)
class Prompts:
    directory: Path

    def _read(self, name: str) -> str:
        return (self.directory / name).read_text(encoding="utf-8")

    def system(self, system: str, variant: str, max_tool_calls: int) -> str:
        descriptions = yaml.safe_load(self._read("systems.yaml"))
        block = self._read(f"agent_variant_{variant}.md").rstrip("\n")
        hint = "" if variant == "A1" else "; N#, C# e K# do pacote de evidências"
        return Template(self._read("agent_system.md")).substitute(
            system_description=descriptions[system],
            max_tool_calls=max_tool_calls,
            variant_block=block,
            evidence_ids_hint=hint,
        )

    def user(self, task: TaskSpec, candidates: Candidates) -> str:
        start, end = task.window_local
        return Template(self._read("agent_user.md")).substitute(
            instruction=task.instruction.strip(),
            window_start=start,
            window_end=end,
            n_failures=task.n_failures,
            fields=", ".join(_FIELD_PT[f] for f in task.fields),
            reasons="\n".join(f"- {r}" for r in candidates.reasons),
            components="\n".join(f"- {c}" for c in candidates.components),
        )

    def retry(self, errors: list[str]) -> str:
        return Template(self._read("agent_retry.md")).substitute(
            errors="\n".join(f"- {e}" for e in errors)
        )

    def fingerprint(self) -> dict[str, str]:
        """SHA-256 of every template, recorded with each run (prompt version)."""
        return {
            name: hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
            for name in TEMPLATE_FILES
        }
