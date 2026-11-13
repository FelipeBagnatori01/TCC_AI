"""The agent loop for one case (ADR 0017).

model call -> tool calls executed -> results appended -> ... -> final JSON answer, validated
against the task; an invalid answer goes back to the model with the errors (at most
``max_retries`` times). Every request carries the same tools and answer schema, so the cached
prompt prefix is stable, and the repetition index ``sample`` so repetitions are independent
samples that are still served from the cache on a re-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tcc_rca.agent.prompts import Prompts
from tcc_rca.agent.schema import (
    ANSWER_JSON_SCHEMA,
    AgentAnswer,
    official_prediction,
    parse_answer,
    validate_answer,
)
from tcc_rca.agent.tasks import Candidates, TaskSpec
from tcc_rca.agent.tools import Toolbox
from tcc_rca.llm.client import LLMClient
from tcc_rca.llm.types import Message


@dataclass
class CaseResult:
    system: str
    query_row: int
    task_index: str
    variant: str
    sample: int
    status: str  # valid | invalid | max_turns | refusal
    answer: AgentAnswer | None
    prediction: str  # official-format string ("{}" when there is no valid answer)
    errors: list[str]
    turns: int = 0
    retries: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    paid_calls: int = 0
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "query_row": self.query_row,
            "task_index": self.task_index,
            "variant": self.variant,
            "sample": self.sample,
            "status": self.status,
            "errors": self.errors,
            "answer": self.answer.model_dump(by_alias=True) if self.answer else None,
            "prediction": self.prediction,
            "turns": self.turns,
            "retries": self.retries,
            "tool_calls": self.tool_calls,
            "tools_used": [t["tool"] for t in self.tool_log],
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "latency_s": round(self.latency_s, 4),
            "paid_calls": self.paid_calls,
        }


def run_case(
    client: LLMClient,
    toolbox: Toolbox,
    task: TaskSpec,
    candidates: Candidates,
    prompts: Prompts,
    *,
    sample: int,
    max_turns: int,
    max_retries: int,
) -> CaseResult:
    messages: list[Message] = [
        Message(
            role="system",
            content=prompts.system(task.system, toolbox.variant, toolbox.max_tool_calls),
        ),
        Message(role="user", content=prompts.user(task, candidates)),
    ]
    result = CaseResult(
        system=task.system,
        query_row=task.query_row,
        task_index=task.task_index,
        variant=toolbox.variant,
        sample=sample,
        status="max_turns",
        answer=None,
        prediction="{}",
        errors=[],
    )
    tools = toolbox.specs
    for _ in range(max_turns):
        response = client.complete(
            tuple(messages), tools=tools, response_schema=ANSWER_JSON_SCHEMA, sample=sample
        )
        result.turns += 1
        result.input_tokens += response.input_tokens
        result.cached_input_tokens += response.cached_input_tokens
        result.cache_write_tokens += response.cache_write_tokens
        result.output_tokens += response.output_tokens
        result.cost_usd += response.cost_usd
        result.latency_s += response.latency_s
        result.paid_calls += int(not response.cached)
        messages.append(response.as_assistant_message())
        if response.stop_reason == "refusal":
            result.status, result.errors = "refusal", ["o modelo recusou a solicitação"]
            break
        if response.tool_calls:
            for call in response.tool_calls:
                content, is_error = toolbox.call(call.name, call.arguments)
                messages.append(
                    Message(role="tool", content=content, tool_call_id=call.id, is_error=is_error)
                )
            continue
        answer, errors = parse_answer(response.text)
        if answer is not None:
            errors = validate_answer(answer, task, candidates, toolbox.known_ids)
        if not errors and answer is not None:
            result.status, result.answer, result.errors = "valid", answer, []
            result.prediction = official_prediction(answer, task)
            break
        result.errors = errors
        if result.retries >= max_retries:
            result.status = "invalid"
            break
        result.retries += 1
        messages.append(Message(role="user", content=prompts.retry(errors)))
    result.tool_calls = toolbox.calls
    result.tool_log = toolbox.log
    result.transcript = [m.model_dump(mode="json", exclude_defaults=True) for m in messages]
    return result
