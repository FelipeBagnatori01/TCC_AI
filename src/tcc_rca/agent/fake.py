"""``fake_agent``: an offline, deterministic stand-in for an LLM (tests and smoke runs only).

It follows a fixed script: find_anomalous_kpis over the query window -> get_evidence when the
tool exists -> an answer built from what it saw (top evidence/anomalous component when it is a
candidate, first candidate reason, first deviation time). It reads only its own conversation,
so it exercises the real tools, schema validation, cache and cost accounting end to end. Its
answers have no scientific meaning.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tcc_rca.llm.types import LLMRequest, ProviderResult, ToolCall

_WINDOW_RE = re.compile(r"Janela da consulta: de (\S+ \S+) a (\S+ \S+) \(UTC\+8\)")
_COUNT_RE = re.compile(r"Número de falhas a explicar: (\d+)")


def _section(text: str, title: str) -> list[str]:
    part = text.split(title, 1)[1] if title in text else ""
    items: list[str] = []
    for line in part.splitlines()[1:]:
        if not line.startswith("- "):
            if items:
                break
            continue
        items.append(line[2:].strip())
    return items


class FakeAgentProvider:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: LLMRequest) -> ProviderResult:
        self.calls += 1
        user = next(m.content for m in request.messages if m.role == "user")
        results = [json.loads(m.content) for m in request.messages if m.role == "tool"]
        names = {t.name for t in request.tools}
        start, end = _WINDOW_RE.search(user).groups()  # type: ignore[union-attr]
        n_calls = len(results)
        if n_calls == 0:
            return self._call(
                "find_anomalous_kpis", {"start": start, "end": end, "component": None}
            )
        if n_calls == 1 and "get_evidence" in names:
            return self._call("get_evidence", {})
        return self._answer(user, results, start)

    def _call(self, name: str, arguments: dict[str, Any]) -> ProviderResult:
        call = ToolCall(id=f"fake_{self.calls}", name=name, arguments=arguments)
        return ProviderResult(
            text="", input_tokens=100, output_tokens=20, tool_calls=(call,), stop_reason="tool_use"
        )

    def _answer(self, user: str, results: list[dict[str, Any]], start: str) -> ProviderResult:
        components = _section(user, "Componentes candidatos:")
        reasons = _section(user, "Motivos candidatos:")
        count_match = _COUNT_RE.search(user)
        n = int(count_match.group(1)) if count_match else 1
        cited = [r["id"] for r in results if "erro" not in r]
        component, instant = components[0], start
        anomalies = next((r for r in results if "kpis" in r and "intervalo" in r), None)
        if anomalies:
            for kpi in anomalies["kpis"]:
                comp = kpi["feature"].split("::", 1)[0]
                if comp in components:
                    component = comp
                    instant = kpi["primeiro_desvio_acima_de_3"] or kpi["instante_pico"] or start
                    break
        package = next((r["pacote"] for r in results if "pacote" in r), None)
        if package:
            ranked = package.get("components") or package.get("native_ranking") or []
            for item in ranked:
                if item["component"] in components:
                    component = item["component"]
                    cited.append(item["id"])
                    break
        answer = {
            "causas": [
                {"instante": instant, "componente": component, "motivo": reasons[0]}
                for _ in range(n)
            ],
            "explicacao": f"Resposta do agente falso (sem valor científico): {component}.",
            "evidencias_citadas": cited,
        }
        text = json.dumps(answer, ensure_ascii=False)
        return ProviderResult(text=text, input_tokens=200, output_tokens=60)
