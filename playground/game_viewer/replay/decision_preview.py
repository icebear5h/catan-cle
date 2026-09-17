"""Viewer adapter for non-mutating decisions through the shared agent harness."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from typing import Any

from cle.env.observation_formatter import CatanObservationFormatter
from cle.game_engine.json import GameEncoder
from cle.game_engine.models.player import Color
from cle.harness import CompletionTransport
from cle.harness.board_surface import board_presentation_payload
from cle.harness.communication import parse_communication_suite
from cle.harness.decision import request_player_attempt
from cle.harness.prompt_store import resolve_prompt_suites
from cle.harness.providers import OpenRouterConfig, OpenRouterTransport
from cle.harness.reasoning import (
    native_reasoning_enabled,
    native_reasoning_returned,
    reasoning_token_count,
)
from cle.harness.shared_suite import parse_shared_prompt_suite
from cle.harness.suite import parse_context_suite
from cle.players.contracts import PlayerAttempt, PlayerContext
from cle.players.validation import action_from_choice, choice_followup_action
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
    notes_seed: str | None = None,
    notes_player: Color | None = None,
) -> dict[str, Any]:
    """Preview a cold, cursor-zero agent without accepting notes or changing replay."""
    context, identity = sandbox.decision_context()
    if notes_seed is not None and notes_player != context.actor:
        raise ValueError("Replay notes seed must belong to the current decision player")
    sources = resolve_prompt_suites()
    if sources.shared is not None:
        shared = parse_shared_prompt_suite(sources.shared.source)
        suite = shared.decision_suite()
        communication_suite = shared.communication_suite()
    else:
        suite = parse_context_suite(sources.decision.source)
        communication_suite = parse_communication_suite(sources.communication.source)
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
            communication_suite=communication_suite,
            notes_seed=notes_seed,
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
    if suite.context.memory_mode == "fresh_notes":
        result["context_bootstrap"] = "cold"
        result["input_start_sequence"] = 0
        result["notes_seeded"] = notes_seed is not None
    if sources.shared is not None:
        result["shared_suite"] = {
            "id": sources.shared.id,
            "version": sources.shared.version,
            "sha256": sources.shared.sha256,
        }
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
        action_from_choice(context, choice)
        if choice is not None and attempt.validation_error is None
        else None
    )
    requested_actions = [] if selected_action is None else [selected_action]
    if selected_action is not None:
        followup = choice_followup_action(context, choice)
        if followup is not None:
            requested_actions.append(followup)
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
    request_payload = None
    if request is not None:
        # Exclude image bytes before recursively projecting the request.
        request_payload = asdict(replace(request, board_presentation=None))
        request_payload["board_presentation"] = board_presentation_payload(
            request.board_presentation, include_text_content=True,
        )

    return {
        "schema": "agent-decision-preview-v2",
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
        "notes_update": choice.notes_update if choice is not None else None,
        "accepted": False,
        "request": (
            json.loads(json.dumps(request_payload, cls=GameEncoder))
            if request is not None else None
        ),
        "context_policy": request.context_policy if request is not None else None,
        "memory_revision": request.memory_revision if request is not None else None,
        "input_next_sequence": request.input_next_sequence if request is not None else None,
        "channel": request.channel if request is not None else None,
        "action_index": action_index,
        "action": str(selected_action) if selected_action is not None else None,
        "requested_action_sequence": [str(action) for action in requested_actions],
        "knight_destination": (
            list(choice.knight_destination)
            if selected_action is not None and choice.knight_destination is not None
            else None
        ),
        "action_description": (
            "; then ".join(
                formatter._format_single_action(action, context.observation)
                for action in requested_actions
            )
            if requested_actions
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
