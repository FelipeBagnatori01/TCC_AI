"""OpenAI adapter: Responses API with function tools and structured output (E7).

* ``system`` messages become ``instructions``; the conversation is a list of input items;
* assistant turns are replayed with their raw output items (reasoning items, which carry
  ``encrypted_content``, and function calls), as the API requires for reasoning models;
  ``store=False``: nothing is kept server-side;
* tool results are ``function_call_output`` items; tools are strict functions;
* no sampling parameters (rejected by reasoning models, ADR 0017); ``reasoning.effort`` and
  ``text.format`` (strict JSON schema) as configured;
* ``usage.input_tokens`` includes cached tokens: the cached part is split out for cost.
"""

from __future__ import annotations

import json
from typing import Any

from tcc_rca.llm.types import LLMRequest, ProviderResult, ToolCall


def build_params(request: LLMRequest) -> dict[str, Any]:
    """Responses API keyword arguments for ``request`` (pure; tested without network)."""
    instructions = "\n\n".join(m.content for m in request.messages if m.role == "system")
    items: list[dict[str, Any]] = []
    for m in request.messages:
        if m.role == "system":
            continue
        if m.role == "user":
            items.append({"role": "user", "content": m.content})
        elif m.role == "tool":
            items.append(
                {"type": "function_call_output", "call_id": m.tool_call_id, "output": m.content}
            )
        elif m.raw:
            items += [dict(item) for item in m.raw]
        else:
            if m.content:
                items.append({"role": "assistant", "content": m.content})
            items += [
                {
                    "type": "function_call",
                    "call_id": c.id,
                    "name": c.name,
                    "arguments": json.dumps(c.arguments, ensure_ascii=False),
                }
                for c in m.tool_calls
            ]
    params: dict[str, Any] = {
        "model": request.model,
        "input": items,
        "max_output_tokens": request.max_tokens,
        "store": False,
    }
    if instructions:
        params["instructions"] = instructions
    if request.tools:
        params["tools"] = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
                "strict": True,
            }
            for t in request.tools
        ]
    if request.effort:
        params["reasoning"] = {"effort": request.effort}
    if request.response_schema is not None:
        params["text"] = {
            "format": {
                "type": "json_schema",
                "name": "resposta",
                "schema": request.response_schema,
                "strict": True,
            }
        }
    return params


def parse_response(response: Any) -> ProviderResult:
    """``ProviderResult`` from a Responses API response object."""
    text_parts, calls, raw = [], [], []
    refused = False
    for item in response.output:
        raw.append(item.model_dump(mode="json", exclude_none=True, by_alias=True))
        if item.type == "function_call":
            calls.append(
                ToolCall(id=item.call_id, name=item.name, arguments=json.loads(item.arguments))
            )
        elif item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    text_parts.append(part.text)
                elif part.type == "refusal":
                    refused = True
    usage = response.usage
    cached = int(getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0)
    if refused:
        stop = "refusal"
    elif calls:
        stop = "tool_use"
    elif getattr(response, "status", "completed") == "incomplete":
        stop = "max_tokens"
    else:
        stop = "end_turn"
    return ProviderResult(
        text="".join(text_parts),
        input_tokens=int(usage.input_tokens or 0) - cached,
        cached_input_tokens=cached,
        output_tokens=int(usage.output_tokens or 0),  # reasoning tokens included
        tool_calls=tuple(calls),
        stop_reason=stop,
        raw_output=tuple(raw),
    )


class OpenAIProvider:
    def __init__(self) -> None:
        import openai  # imported only when this provider is selected

        self._client = openai.OpenAI(max_retries=5)  # credentials: OPENAI_API_KEY (from .env)

    def complete(self, request: LLMRequest) -> ProviderResult:
        response = self._client.responses.create(**build_params(request))
        return parse_response(response)
