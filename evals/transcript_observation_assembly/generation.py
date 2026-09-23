"""One model call per assembly job, with attempt bookkeeping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, List, Tuple

from evals.json_types import JsonDict
from evals.transcript_observation_assembly.artifacts import utc_now
from evals.transcript_observation_assembly.job import (
    ObservationAssemblyError,
    ObservationAssemblyJob,
)
from evals.transcript_observation_assembly.prompting import (
    _tool_handler,
    _user_prompt,
    parse_assembly_response,
)
from evals.transcript_observation_assembly.protocol import (
    _SYSTEM_PROMPT,
    _TOOL_DEFINITIONS,
    ATTEMPT_SCHEMA,
    DEFAULT_MODEL,
    PROMPT_VERSION,
    RESULT_SCHEMA,
)
from evals.transcript_reasoning.generation import (
    ToolQuery,
    first_tool_is_board,
    response_field,
    response_text,
)
from evals.transcript_reasoning.support import json_value
from playground.openrouter_client import query_text_with_tools


def generate_assembly_job(
    job: ObservationAssemblyJob,
    model_id: str = DEFAULT_MODEL,
    *,
    max_tokens: int = 2_000,
    timeout: float = 180.0,
    query: ToolQuery = query_text_with_tools,
) -> Tuple[JsonDict, JsonDict]:
    """Run one forced board-inspection loop and validate exact evidence use."""
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
            raise ObservationAssemblyError("Model did not inspect the board first")
        paragraphs, omitted_ids = parse_assembly_response(
            job, response_text(response)
        )
        result: JsonDict = {
            "schema": RESULT_SCHEMA,
            "job_id": job.job_id,
            "game_id": job.game_id,
            "replay_index": job.replay_index,
            "previous_replay_index": job.previous_replay_index,
            "anchor_kind": job.anchor_kind,
            "decision_ids": list(job.decision_ids),
            "input_hash": job.input_hash,
            "model_id": model_id,
            "generator_version": PROMPT_VERSION,
            "status": "ready" if paragraphs else "empty",
            "recorded_at": recorded_at,
            "board_state_hash": job.board_snapshot["board_state_hash"],
            "window_start_s": job.window_start_s,
            "window_end_s": job.window_end_s,
            "called_tools": called_tools,
            "evidence_count": len(job.utterances),
            "omitted_evidence_ids": list(omitted_ids),
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
        error: JsonDict = {"type": type(exc).__name__, "message": str(exc)}
        result = {
            "schema": RESULT_SCHEMA,
            "job_id": job.job_id,
            "game_id": job.game_id,
            "replay_index": job.replay_index,
            "previous_replay_index": job.previous_replay_index,
            "anchor_kind": job.anchor_kind,
            "decision_ids": list(job.decision_ids),
            "input_hash": job.input_hash,
            "model_id": model_id,
            "generator_version": PROMPT_VERSION,
            "status": "error",
            "recorded_at": recorded_at,
            "board_state_hash": job.board_snapshot["board_state_hash"],
            "window_start_s": job.window_start_s,
            "window_end_s": job.window_end_s,
            "called_tools": response_field(response, "called_tools") or [],
            "evidence_count": len(job.utterances),
            "omitted_evidence_ids": [],
            "paragraphs": [],
            "usage": response_field(response, "usage") or {},
            "latency_ms": response_field(response, "latency_ms"),
            "error": error,
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
            "error": error,
        }
        return result, attempt

