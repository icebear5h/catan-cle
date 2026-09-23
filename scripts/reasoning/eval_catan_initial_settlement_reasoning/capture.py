"""Single-request native-reasoning capture against OpenRouter."""

from __future__ import annotations

import httpx

from cle.harness.board_surface import board_presentation_payload
from cle.harness.decision import request_player_attempt
from cle.harness.providers import OpenRouterConfig, OpenRouterTransport
from cle.harness.reasoning import (
    native_reasoning_request,
    native_reasoning_returned,
    reasoning_token_count,
)
from cle.players.contracts import PlayerAttempt, PlayerChoice, PlayerContext
from scripts.reasoning.eval_catan_initial_settlement_reasoning.artifacts import (
    JsonDict,
    json_payload,
    message_payload,
    utc_now,
)
from scripts.reasoning.eval_catan_initial_settlement_reasoning.seeds import (
    SeedInput,
    legal_action_payload,
)

SCHEMA = "catan-initial-settlement-reasoning-trace/v1"

__all__ = ["SCHEMA", "capture_trace"]


def _choice_payload(attempt: PlayerAttempt, context: PlayerContext) -> JsonDict | None:
    choice = attempt.choice
    if choice is None:
        return None
    if not isinstance(choice, PlayerChoice):
        raise RuntimeError("Reasoning probe returned a non-action choice")
    actions = legal_action_payload(context)
    selected = actions[choice.action_index]
    return {
        "action_index": choice.action_index,
        "action": selected["action"],
        "action_description": selected["description"],
        "game_plan": choice.game_plan,
        "parse_warning": choice.parse_warning,
    }


async def capture_trace(
    seed_input: SeedInput,
    model_id: str,
    *,
    reasoning_effort: str,
    temperature: float,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> JsonDict:
    reasoning = native_reasoning_request(reasoning_effort)
    transport = OpenRouterTransport(
        OpenRouterConfig(
            model=model_id,
            temperature=temperature,
            max_tokens=None,
            timeout_seconds=900.0,
            max_retries=0,
            reasoning=reasoning,
        ),
        api_key=api_key,
        client=client,
    )
    try:
        attempt = await request_player_attempt(
            seed_input.context,
            transport,
            game_plan="",
            session_id=(
                f"fresh-initial-settlement-seed-{seed_input.manifest['seed']}:"
                f"{model_id}"
            ),
            suite=seed_input.suite,
        )
    finally:
        await transport.aclose()

    response = attempt.model_response
    request = attempt.model_request
    if response is None or request is None:
        raise RuntimeError("Reasoning probe did not retain its provider exchange")
    provider_request = response.provider_request_payload
    if not isinstance(provider_request, dict):
        raise RuntimeError("Reasoning probe did not retain provider request metadata")
    if "max_tokens" in provider_request:
        raise RuntimeError("Uncapped reasoning request unexpectedly sent max_tokens")
    if message_payload(request.messages) != seed_input.manifest["messages"]:
        raise RuntimeError("Model-specific reasoning prompt changed")
    board = board_presentation_payload(
        request.board_presentation,
        include_text_content=True,
    )
    if board != seed_input.manifest["board_presentation"]:
        raise RuntimeError("Model-specific board presentation changed")

    usage = dict(response.usage)
    trace: JsonDict = {
        "schema": SCHEMA,
        "trace_id": (
            f"fresh-initial-settlement-seed-{seed_input.manifest['seed']}:"
            f"{model_id}"
        ),
        "recorded_at": utc_now(),
        "input": dict(seed_input.manifest),
        "request": {
            "requested_model": model_id,
            "temperature": temperature,
            "reasoning": dict(reasoning),
            "max_tokens_omitted": True,
            "provider_payload": json_payload(provider_request),
        },
        "response": {
            "served_model": response.model,
            "provider_response_id": response.provider_response_id,
            "provider_request_id": response.provider_request_id,
            "finish_reason": response.finish_reason,
            "provider_native_finish_reason": response.provider_native_finish_reason,
            "latency_ms": response.latency_ms,
            "usage": dict(usage),
            "reasoning_tokens": reasoning_token_count(usage),
            "native_reasoning_returned": native_reasoning_returned(
                response.native_reasoning,
                response.native_reasoning_details,
                usage,
            ),
            "native_reasoning": response.native_reasoning,
            "native_reasoning_details": list(response.native_reasoning_details),
            "final_response": response.content,
            "provider_payload": json_payload(response.provider_response_payload),
        },
        "parse": {
            "error": attempt.validation_error,
            "choice": _choice_payload(attempt, seed_input.context),
        },
    }
    return trace
