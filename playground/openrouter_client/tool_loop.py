"""The OpenAI-compatible tool-calling loop behind the harness tool contracts."""

import json
import time
from collections.abc import Callable
from copy import deepcopy
from typing import TypedDict

import httpx

from .config import ToolLoopResult, _build_headers, _get_provider_config

__all__ = ["query_text_with_tools"]


class _ToolCallFunction(TypedDict):
    """The name and raw JSON arguments of one requested tool call."""

    name: object
    arguments: str


class _AssistantToolCall(TypedDict):
    """One tool call echoed back into the conversation as the assistant sent it."""

    type: str
    id: object
    function: _ToolCallFunction


def query_text_with_tools(
    model: str,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]],
    tool_handler: Callable[[str, dict[str, object]], object],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 120.0,
    provider: str = "openrouter",
    forced_first_tool: str | None = None,
    response_format: dict[str, object] | None = None,
    max_tool_rounds: int = 6,
) -> ToolLoopResult:
    """Run an OpenAI-compatible tool loop and return the final text response."""
    if max_tool_rounds < 1:
        raise ValueError("max_tool_rounds must be at least 1")

    api_url, api_key, extra_headers = _get_provider_config(provider)
    headers = _build_headers(api_key, extra_headers)
    conversation = deepcopy(messages)
    initial_message_count = len(conversation)
    called_tools: list[str] = []
    usage_totals: dict[str, float] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }
    started_at = time.time()
    last_model = model

    for round_index in range(max_tool_rounds):
        payload: dict[str, object] = {
            "model": model,
            "messages": conversation,
            "tools": tools,
            "tool_choice": (
                {
                    "type": "function",
                    "function": {"name": forced_first_tool},
                }
                if round_index == 0 and forced_first_tool
                else "auto"
            ),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        with httpx.Client(timeout=timeout) as client:
            response = client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
        data = response.json()
        last_model = data.get("model", last_model)
        usage = data.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                usage_totals[key] += value
        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            usage_totals["cost"] += float(cost)

        choice = data["choices"][0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return {
                "content": message.get("content") or "",
                "model": last_model,
                "usage": usage_totals,
                "latency_ms": int((time.time() - started_at) * 1000),
                "finish_reason": choice.get("finish_reason"),
                "called_tools": called_tools,
                "tool_messages": conversation[initial_message_count:],
                "native_reasoning": (
                    message.get("reasoning_content")
                    or message.get("reasoning")
                    or ""
                ),
            }

        assistant_tool_calls: list[_AssistantToolCall] = []
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            assistant_tool_calls.append(
                {
                    "type": "function",
                    "id": tool_call.get("id"),
                    "function": {
                        "name": function.get("name"),
                        "arguments": function.get("arguments") or "{}",
                    },
                }
            )
        conversation.append(
            {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": assistant_tool_calls,
            }
        )

        for tool_call in assistant_tool_calls:
            function = tool_call["function"]
            name = function.get("name")
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = None
            if not isinstance(arguments, dict):
                result: object = {
                    "error": "Tool arguments must be a JSON object.",
                    "received": raw_arguments,
                }
            else:
                result = tool_handler(str(name or ""), arguments)
            called_tools.append(str(name or ""))
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": name,
                    "content": json.dumps(result, sort_keys=True),
                }
            )

    raise RuntimeError(
        f"Model did not return a final response after {max_tool_rounds} tool rounds"
    )
