"""Read-only reasoning diagnostics for accepted live sandbox decisions."""

from __future__ import annotations

import json
from typing import Any

from cle.harness.reasoning import (
    native_reasoning_enabled,
    native_reasoning_returned,
    reasoning_token_count,
)
from cle.sandbox.contracts import SandboxStepResult
from game_engine.json import GameEncoder


def build_live_reasoning_traces(
    sandbox,
    result: SandboxStepResult,
) -> list[dict[str, Any]]:
    """Project accepted agent attempts without recreating an action runtime."""
    traces = []
    for context, attempt, transition in zip(
        result.contexts,
        result.attempts,
        result.transitions,
    ):
        player = sandbox.players[context.actor]
        if player.status().get("kind") != "agent":
            continue
        choice = attempt.choice
        if choice is None:
            continue

        response = attempt.model_response
        usage = dict(response.usage if response is not None else choice.usage)
        native_reasoning = (
            response.native_reasoning
            if response is not None
            else choice.native_reasoning
        )
        native_details = (
            response.native_reasoning_details
            if response is not None
            else choice.native_reasoning_details
        )
        reasoning_request = dict(
            response.reasoning_request
            if response is not None
            else choice.reasoning_request
        )
        reasoning_returned = native_reasoning_returned(
            native_reasoning,
            native_details,
            usage,
        )
        reasoning_requested = bool(reasoning_request) and native_reasoning_enabled(
            reasoning_request
        )

        traces.append({
            "schema": "live-reasoning-trace-v1",
            "context_id": context.context_id,
            "player_color": context.actor.value,
            "turn_number": context.turn_number,
            "phase": context.phase,
            "prompt_key": context.prompt_key,
            "action_index": choice.action_index,
            "action_type": transition.requested_action.action_type.value,
            "game_plan": choice.game_plan,
            "rationale": choice.rationale,
            "rationale_source": "model_response_xml",
            "native_reasoning": native_reasoning,
            "native_reasoning_details": list(native_details),
            "native_reasoning_source": (
                "provider_response" if reasoning_returned else None
            ),
            "native_reasoning_requested": reasoning_requested,
            "native_reasoning_returned": reasoning_returned,
            "native_reasoning_missing": reasoning_requested and not reasoning_returned,
            "reasoning_request": reasoning_request,
            "reasoning_tokens": reasoning_token_count(usage),
            "finish_reason": response.finish_reason if response is not None else None,
            "provider_native_finish_reason": (
                response.provider_native_finish_reason
                if response is not None
                else choice.provider_native_finish_reason
            ),
            "provider_response_id": (
                response.provider_response_id
                if response is not None
                else choice.provider_response_id
            ),
            "provider_request_id": (
                response.provider_request_id
                if response is not None
                else choice.provider_request_id
            ),
            "model": response.model if response is not None else choice.model,
            "latency_ms": (
                response.latency_ms if response is not None else choice.latency_ms
            ),
            "usage": usage,
        })

    return json.loads(json.dumps(traces, cls=GameEncoder))
