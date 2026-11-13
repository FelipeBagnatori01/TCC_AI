"""Agent answer schema (pydantic) and its validation against the task (ADR 0017).

JSON field names are Portuguese (the answer is read by engineers and cited in the thesis); the
Python attributes are English (R11). ``ANSWER_JSON_SCHEMA`` is the strict JSON Schema sent as the
structured-output format: every property required, no extra properties, nullable fields typed
``["string", "null"]``.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tcc_rca.agent.tasks import Candidates, TaskSpec
from tcc_rca.ingestion.timeutil import parse_local

LOCAL_FORMAT = "%Y-%m-%d %H:%M:%S"


class RootCause(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    occurrence_datetime: str | None = Field(alias="instante")
    component: str | None = Field(alias="componente")
    reason: str | None = Field(alias="motivo")


class AgentAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    root_causes: list[RootCause] = Field(alias="causas")
    explanation: str = Field(alias="explicacao")
    cited_evidence: list[str] = Field(alias="evidencias_citadas")


_NULLABLE_STRING = {"type": ["string", "null"]}
ANSWER_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "causas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "instante": {
                        **_NULLABLE_STRING,
                        "description": "YYYY-MM-DD HH:MM:SS (UTC+8) ou null se não for pedido",
                    },
                    "componente": {**_NULLABLE_STRING, "description": "componente candidato"},
                    "motivo": {**_NULLABLE_STRING, "description": "motivo candidato"},
                },
                "required": ["instante", "componente", "motivo"],
                "additionalProperties": False,
            },
        },
        "explicacao": {"type": "string", "description": "explicação em português"},
        "evidencias_citadas": {
            "type": "array",
            "items": {"type": "string"},
            "description": "identificadores das evidências usadas (C#, K#, N#, T#)",
        },
    },
    "required": ["causas", "explicacao", "evidencias_citadas"],
    "additionalProperties": False,
}

_FIELD_PT = {"occurrence_datetime": "instante", "component": "componente", "reason": "motivo"}


def parse_answer(text: str) -> tuple[AgentAnswer | None, list[str]]:
    """Parse the model's final text. Tolerates a fenced block around the JSON object."""
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`")
        body = body.split("\n", 1)[1] if "\n" in body else body
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None, ["a resposta final não contém um objeto JSON"]
    try:
        data = json.loads(body[start : end + 1])
    except json.JSONDecodeError as exc:
        return None, [f"JSON inválido: {exc.msg} (posição {exc.pos})"]
    try:
        return AgentAnswer.model_validate(data), []
    except ValidationError as exc:
        return None, [
            f"campo {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        ]


def validate_answer(
    answer: AgentAnswer,
    task: TaskSpec,
    candidates: Candidates,
    known_ids: Collection[str],
) -> list[str]:
    """Task-level checks; an empty list means valid. Messages go back to the model on retry."""
    errors: list[str] = []
    if len(answer.root_causes) != task.n_failures:
        errors.append(
            f"a consulta tem {task.n_failures} falha(s); a resposta traz "
            f"{len(answer.root_causes)} causa(s)"
        )
    start, end = task.window_start_utc, task.window_end_utc
    for i, cause in enumerate(answer.root_causes, 1):
        for field in task.fields:
            if getattr(cause, field) in (None, ""):
                errors.append(f"causa {i}: o campo '{_FIELD_PT[field]}' é obrigatório nesta tarefa")
        if "occurrence_datetime" in task.fields and cause.occurrence_datetime:
            try:
                moment = parse_local(cause.occurrence_datetime)
            except ValueError:
                errors.append(f"causa {i}: 'instante' fora do formato {LOCAL_FORMAT}")
            else:
                if not start <= moment <= end:
                    lo, hi = task.window_local
                    errors.append(f"causa {i}: 'instante' fora da janela da consulta ({lo} a {hi})")
        comp, reason = cause.component, cause.reason
        if "component" in task.fields and comp and comp not in candidates.components:
            errors.append(f"causa {i}: componente '{comp}' não é candidato")
        if "reason" in task.fields and reason and reason not in candidates.reasons:
            errors.append(f"causa {i}: motivo '{reason}' não é candidato")
    if not answer.explanation.strip():
        errors.append("'explicacao' está vazia")
    if not answer.cited_evidence:
        errors.append("cite ao menos uma evidência em 'evidencias_citadas'")
    unknown = sorted(set(answer.cited_evidence) - set(known_ids))
    if unknown:
        errors.append(f"evidências citadas que não existem nesta investigação: {unknown}")
    return errors


def official_prediction(answer: AgentAnswer, task: TaskSpec) -> str:
    """Prediction string for the official OpenRCA scorer: one object per failure, keys in the
    order its regex expects, only the fields the task asks for."""
    objects: dict[str, dict[str, str]] = {}
    for i, cause in enumerate(answer.root_causes, 1):
        obj: dict[str, str] = {}
        if "occurrence_datetime" in task.fields and cause.occurrence_datetime:
            obj["root cause occurrence datetime"] = cause.occurrence_datetime
        if "component" in task.fields and cause.component:
            obj["root cause component"] = cause.component
        if "reason" in task.fields and cause.reason:
            obj["root cause reason"] = cause.reason
        objects[str(i)] = obj
    return json.dumps(objects, ensure_ascii=False)


def local_minute(text: str) -> datetime:
    """UTC instant of a ``YYYY-MM-DD HH:MM:SS`` UTC+8 string (tool arguments, answers)."""
    return parse_local(text)
