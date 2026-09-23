"""One provider call per decision, with reasoning bookkeeping."""

from __future__ import annotations

import asyncio
import hashlib

from cle.env.observation_formatter import CatanObservationFormatter
from cle.harness.decision import request_player_attempt
from cle.harness.providers import OpenRouterConfig
from cle.harness.reasoning import native_reasoning_returned, reasoning_token_count
from cle.players.contracts import CommunicationChoice, PlayerAttempt, PlayerContext
from cle.players.validation import choice_followup_action
from evals import replay_action_diff
from evals.json_types import JsonDict, JsonList
from evals.replay_action_diff.contracts import SELECTION_CONTRACT
from evals.replay_action_diff.responses import reasoning_request_for_model


def _query_model(
    model_id: str,
    context: PlayerContext,
    reasoning_effort: str,
    max_tokens: int,
) -> JsonDict:
    reasoning = reasoning_request_for_model(model_id, reasoning_effort)
    suite = replay_action_diff.load_context_suite()
    transport = replay_action_diff.OpenRouterTransport(
        OpenRouterConfig(
            model=model_id,
            temperature=0.2,
            max_tokens=max_tokens,
            reasoning=reasoning,
        )
    )

    async def request_attempt() -> PlayerAttempt:
        try:
            return await request_player_attempt(
                context,
                transport,
                game_plan="",
                session_id=f"{context.context_id}:{model_id}",
                suite=suite,
            )
        finally:
            await transport.aclose()

    attempt = asyncio.run(request_attempt())
    response = attempt.model_response
    if response is None:
        raise RuntimeError("Agent attempt did not retain its model response")
    choice = attempt.choice
    if isinstance(choice, CommunicationChoice):
        raise ValueError("Player choice must be a PlayerChoice")
    usage = dict(response.usage)
    formatter = CatanObservationFormatter()
    available_actions: JsonList = [
        {
            "index": index,
            "action": str(action),
            "description": formatter._format_single_action(
                action,
                context.observation,
            ),
        }
        for index, action in enumerate(context.legal_actions)
    ]
    selected = (
        replay_action_diff.action_from_choice(context, choice)
        if choice is not None and attempt.validation_error is None
        else None
    )
    action_index = (
        choice.action_index if selected is not None and choice is not None else None
    )
    requested_actions = [] if selected is None else [selected]
    if selected is not None and choice is not None:
        followup = choice_followup_action(context, choice)
        if followup is not None:
            requested_actions.append(followup)
    native_returned = native_reasoning_returned(
        response.native_reasoning,
        response.native_reasoning_details,
        usage,
    )
    messages = attempt.model_request.messages if attempt.model_request is not None else ()
    return {
        "context_version": f"{suite.id}@{suite.version}",
        "selection_contract": SELECTION_CONTRACT,
        "response_format": suite.response.format,
        "context_suite_sha256": hashlib.sha256(suite.model_dump_json().encode("utf-8")).hexdigest(),
        "game_plan": choice.game_plan if choice is not None else "",
        "action_index": action_index,
        "action": str(selected) if selected is not None else None,
        "requested_action_sequence": [str(action) for action in requested_actions],
        "knight_destination": (
            list(choice.knight_destination)
            if selected is not None
            and choice is not None
            and choice.knight_destination is not None
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
        "finish_reason": response.finish_reason,
        "provider_native_finish_reason": response.provider_native_finish_reason,
        "provider_response_id": response.provider_response_id,
        "provider_request_id": response.provider_request_id,
        "response_truncated": response.finish_reason == "length",
        "available_actions": available_actions,
        "raw_response": response.content,
        "latency_ms": response.latency_ms,
        "usage": usage,
        "requested_model": model_id,
        "model": response.model or model_id,
        "native_reasoning": response.native_reasoning,
        "native_reasoning_details": list(response.native_reasoning_details),
        "native_reasoning_returned": native_returned,
        "native_reasoning_missing": not native_returned,
        "reasoning_request": {**reasoning},
        "reasoning_tokens": reasoning_token_count(usage),
        "system_prompt": messages[0].content if messages else "",
        "context_prompt": messages[-1].content if messages else "",
    }

