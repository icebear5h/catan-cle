"""One model call per reasoning job, with attempt bookkeeping."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Dict, List, Protocol, Tuple

from evals.json_types import JsonDict, JsonValue
from evals.transcript_reasoning.job import NarratorReasoningError, ReasoningJob
from evals.transcript_reasoning.prompting import (
    _tool_handler,
    _user_prompt,
    parse_reasoning_response,
)
from evals.transcript_reasoning.protocol import (
    _SYSTEM_PROMPT,
    _TOOL_DEFINITIONS,
    ATTEMPT_SCHEMA,
    DEFAULT_MODEL,
    PROMPT_VERSION,
    RESULT_SCHEMA,
)
from evals.transcript_reasoning.support import json_value, utc_now
from playground.openrouter_client import query_text_with_tools


class ToolQuery(Protocol):
    """The tool-loop call shape shared by the real client and test doubles."""

    def __call__(
        self,
        model: str,
        messages: List[Dict[str, object]],
        tools: List[Dict[str, object]],
        tool_handler: Callable[[str, Dict[str, object]], object],
        *,
        temperature: float,
        max_tokens: int,
        timeout: float,
        provider: str,
        forced_first_tool: str | None,
        response_format: Dict[str, object] | None,
        max_tool_rounds: int,
    ) -> Mapping[str, object]: ...


def response_field(response: Mapping[str, object], key: str) -> JsonValue:
    """Read one JSON field of a tool-loop response (``None`` when absent)."""
    return json_value(response.get(key), f"response {key}")


def response_text(response: Mapping[str, object]) -> str:
    """Read the model's raw text, treating an absent or empty answer as ``""``."""
    content = response.get("content") or ""
    if not isinstance(content, str):
        raise TypeError(f"response content is not a string: {type(content).__name__}")
    return content


def first_tool_is_board(called_tools: JsonValue) -> bool:
    """Whether the recorded tool calls start with ``inspect_board``."""
    return (
        isinstance(called_tools, list)
        and bool(called_tools)
        and called_tools[0] == "inspect_board"
    )


def generate_reasoning_job(
    job: ReasoningJob,
    model_id: str = DEFAULT_MODEL,
    *,
    max_tokens: int = 1_200,
    timeout: float = 180.0,
    query: ToolQuery = query_text_with_tools,
) -> Tuple[JsonDict, JsonDict]:
    """Run one forced board-inspection tool loop and validate its paragraphs."""
    messages: List[Dict[str, object]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(job)},
    ]
    recorded_at = utc_now()
    response: Mapping[str, object] = {}
    try:
        response = query(
            model_id,
            messages,
            _TOOL_DEFINITIONS,
            lambda name, arguments: _tool_handler(job, name, arguments),
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=timeout,
            provider="openrouter",
            forced_first_tool="inspect_board",
            response_format={"type": "json_object"},
            max_tool_rounds=6,
        )
        called_tools = response_field(response, "called_tools") or []
        if not first_tool_is_board(called_tools):
            raise NarratorReasoningError("Model did not inspect the board first")
        paragraphs = parse_reasoning_response(job, response_text(response))
        result: JsonDict = {
            "schema": RESULT_SCHEMA,
            "job_id": job.job_id,
            "game_id": job.game_id,
            "replay_index": job.replay_index,
            "input_hash": job.input_hash,
            "model_id": model_id,
            "generator_version": PROMPT_VERSION,
            "status": "ready" if paragraphs else "empty",
            "recorded_at": recorded_at,
            "board_state_hash": job.board_snapshot["board_state_hash"],
            "window_start_s": job.window_start_s,
            "window_end_s": job.window_end_s,
            "called_tools": called_tools,
            "paragraphs": list(paragraphs),
            "usage": response_field(response, "usage") or {},
            "latency_ms": response_field(response, "latency_ms"),
            "error": None,
        }
        attempt: JsonDict = {
            "schema": ATTEMPT_SCHEMA,
            "job_id": job.job_id,
            "input_hash": job.input_hash,
            "model_id": model_id,
            "recorded_at": recorded_at,
            "status": result["status"],
            "messages": json_value(messages, "messages"),
            "tool_messages": response_field(response, "tool_messages") or [],
            "called_tools": called_tools,
            "raw_response": response_field(response, "content") or "",
            "usage": response_field(response, "usage") or {},
            "latency_ms": response_field(response, "latency_ms"),
            "error": None,
        }
        return result, attempt
    except Exception as exc:
        result = {
            "schema": RESULT_SCHEMA,
            "job_id": job.job_id,
            "game_id": job.game_id,
            "replay_index": job.replay_index,
            "input_hash": job.input_hash,
            "model_id": model_id,
            "generator_version": PROMPT_VERSION,
            "status": "error",
            "recorded_at": recorded_at,
            "board_state_hash": job.board_snapshot["board_state_hash"],
            "window_start_s": job.window_start_s,
            "window_end_s": job.window_end_s,
            "called_tools": response_field(response, "called_tools") or [],
            "paragraphs": [],
            "usage": response_field(response, "usage") or {},
            "latency_ms": response_field(response, "latency_ms"),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        attempt = {
            "schema": ATTEMPT_SCHEMA,
            "job_id": job.job_id,
            "input_hash": job.input_hash,
            "model_id": model_id,
            "recorded_at": recorded_at,
            "status": "error",
            "messages": json_value(messages, "messages"),
            "tool_messages": response_field(response, "tool_messages") or [],
            "called_tools": response_field(response, "called_tools") or [],
            "raw_response": response_field(response, "content") or "",
            "usage": response_field(response, "usage") or {},
            "latency_ms": response_field(response, "latency_ms"),
            "error": result["error"],
        }
        return result, attempt

