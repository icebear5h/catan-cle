"""Viewer adapter for non-mutating decisions through the shared agent harness."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from cle.env.observation_formatter import CatanObservationFormatter
from cle.harness import CompletionTransport, load_context_suite
from cle.harness.decision import request_player_attempt
from cle.harness.providers import OpenRouterConfig, OpenRouterTransport
from cle.harness.reasoning import (
    native_reasoning_enabled,
    native_reasoning_returned,
    reasoning_token_count,
)
from cle.players.contracts import PlayerAttempt, PlayerContext
from cle.sandbox.replay import ReplaySandbox

TransportFactory = Callable[..., CompletionTransport]


async def generate_decision_preview(
    sandbox: ReplaySandbox,
    *,
    model: str,
    game_plan: str,
    reasoning_request: Mapping[str, Any],
    temperature: float = 0.2,
    max_tokens: int = 8_192,
    transport_factory: TransportFactory | None = None,
) -> dict[str, Any]:
    """Run one frozen replay context through the ordinary `AgentPlayer` path."""
    context, identity = sandbox.decision_context()
    suite = load_context_suite()
    if transport_factory is None:
        transport: CompletionTransport = OpenRouterTransport(
            OpenRouterConfig(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning_request,
            )
        )
        owns_transport = True
    else:
        transport = transport_factory(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning_request,
        )
        owns_transport = False

    try:
        attempt = await request_player_attempt(
            context,
            transport,
            game_plan=game_plan,
            session_id=(
                f"{identity['game_id']}:{context.actor.value}:"
                f"replay:{identity['replay_index']}"
            ),
            suite=suite,
        )
    finally:
        if owns_transport:
            await transport.aclose()

    result = serialize_decision_preview(
        attempt,
        context,
        model=model,
        game_id=identity["game_id"],
        replay_index=identity["replay_index"],
        reasoning_request=reasoning_request,
        max_tokens=max_tokens,
        suite_id=suite.id,
        suite_version=suite.version,
    )
    result["stale"] = sandbox.is_stale(identity)
    return result


def serialize_decision_preview(
    attempt: PlayerAttempt,
    context: PlayerContext,
    *,
    model: str,
    game_id: str | None,
    replay_index: int,
    reasoning_request: Mapping[str, Any],
    max_tokens: int,
    suite_id: str,
    suite_version: str,
) -> dict[str, Any]:
    """Return a JSON-safe viewer record without inventing replay prompt semantics."""
    response = attempt.model_response
    request = attempt.model_request
    choice = attempt.choice
    usage = dict(response.usage) if response is not None else {}
    native_reasoning = response.native_reasoning if response is not None else ""
    native_details = (
        response.native_reasoning_details if response is not None else ()
    )
    returned = native_reasoning_returned(
        native_reasoning,
        native_details,
        usage,
    )
    enabled = native_reasoning_enabled(reasoning_request)

    action_index = choice.action_index if choice is not None else None
    selected_action = (
        context.legal_actions[action_index]
        if action_index is not None and 0 <= action_index < len(context.legal_actions)
        else None
    )
    formatter = CatanObservationFormatter()
    available_actions = [
        {
            "index": index,
            "action": str(action),
            "description": formatter._format_single_action(action, context.observation),
        }
        for index, action in enumerate(context.legal_actions)
    ]
    messages = (
        [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ]
        if request is not None
        else []
    )

    return {
        "schema": "agent-decision-preview-v1",
        "context_version": f"{suite_id}@{suite_version}",
        "context_id": context.context_id,
        "game_id": game_id,
        "replay_index": replay_index,
        "phase": context.phase,
        "prompt_key": context.prompt_key,
        "player_color": context.actor.value,
        "requested_model": model,
        "model": response.model or model if response is not None else model,
        "generation_max_tokens": max_tokens,
        "game_plan": choice.game_plan if choice is not None else "",
        "rationale": choice.rationale if choice is not None else "",
        "action_index": action_index,
        "action": str(selected_action) if selected_action is not None else None,
        "action_description": (
            formatter._format_single_action(selected_action, context.observation)
            if selected_action is not None
            else None
        ),
        "parse_error": attempt.validation_error,
        "finish_reason": response.finish_reason if response is not None else None,
        "provider_native_finish_reason": (
            response.provider_native_finish_reason if response is not None else None
        ),
        "provider_response_id": (
            response.provider_response_id if response is not None else None
        ),
        "provider_request_id": (
            response.provider_request_id if response is not None else None
        ),
        "response_truncated": (
            response.finish_reason == "length" if response is not None else False
        ),
        "native_reasoning": native_reasoning,
        "native_reasoning_details": list(native_details),
        "native_reasoning_returned": returned,
        "native_reasoning_missing": enabled and not returned,
        "reasoning_request": dict(reasoning_request),
        "reasoning_tokens": reasoning_token_count(usage),
        "observation": formatter.format(
            context.observation,
            include_legal_actions=False,
        ).raw_str,
        "available_actions": available_actions,
        "raw_response": response.content if response is not None else "",
        "latency_ms": response.latency_ms if response is not None else None,
        "usage": usage,
        "model_messages": messages,
        "stale": False,
    }
