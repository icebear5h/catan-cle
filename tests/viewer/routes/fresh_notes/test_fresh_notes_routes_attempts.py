"""Failed attempts expose typed notes and provenance, never provider bodies."""
import asyncio
import json
from dataclasses import replace
from typing import Any

import pytest

from cle.harness.catan_board_surface import ImageBoardPresenter
from cle.harness.decision import request_player_attempt
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.prompt_store import (
    resolve_prompt_suites,
)
from cle.harness.shared_suite import parse_shared_prompt_suite
from cle.players.contracts import PlayerAttempt, PlayerChoice
from cle.sandbox.factory import (
    LiveSandboxConfig,
    create_live_sandbox,
)
from playground.game_viewer.routes import live_game

from .support import (
    LocalTransport,
)


@pytest.mark.parametrize("image_board", [False, True])
def test_failed_attempt_exposes_typed_notes_and_provenance_not_provider_bodies(image_board: bool) -> None:
    board = None
    if image_board:
        sandbox = create_live_sandbox(LiveSandboxConfig(seed=9, shuffle_players=False))
        board = ImageBoardPresenter().present(sandbox.decision_context())
    request = ModelRequest(
        "decision", "session", (), memory_revision=4, input_next_sequence=12,
        context_policy="fresh_notes", channel="action", board_presentation=board,
    )
    payload: Any = live_game._failed_attempt_payload(PlayerAttempt(
        "decision", PlayerChoice(0, notes_update="unaccepted parsed notes"), "withheld",
        model_request=request,
        model_response=ModelResponse(
            "visible completion", provider_request_payload={"secret": "RAW_REQUEST"},
            provider_response_payload={"body": "RAW_RESPONSE"},
        ),
    ))
    assert payload["accepted"] is False
    assert payload["notes_update"] == "unaccepted parsed notes"
    assert payload["context_policy"] == "fresh_notes"
    assert payload["request"]["memory_revision"] == payload["memory_revision"] == 4
    assert payload["input_next_sequence"] == 12
    assert payload["channel"] == "action"
    assert "RAW_REQUEST" not in json.dumps(payload)
    assert "RAW_RESPONSE" not in json.dumps(payload)
    if image_board:
        assert payload["request"]["board_presentation"]["kind"] == "image"
        assert payload["request"]["board_presentation"]["data"] is None


def test_request_attempt_accepts_matched_pair_and_only_explicit_fresh_seed() -> None:
    source: Any = resolve_prompt_suites().shared
    bundle: Any = parse_shared_prompt_suite(source.source)
    transport: Any = LocalTransport()
    sandbox = create_live_sandbox(LiveSandboxConfig(seed=9, shuffle_players=False))
    context: Any = replace(sandbox.decision_context(), visible_through_sequence=-1)
    transport.context_factory = lambda request: context
    for seed in (None, "own explicit notes"):
        attempt: Any = asyncio.run(request_player_attempt(
            context, transport, suite=bundle.decision_suite(),
            communication_suite=bundle.communication_suite(),
            game_plan="DO_NOT_COPY_LEGACY_MEMORY", notes_seed=seed,
        ))
        assert attempt.validation_error is None
        assert attempt.choice.notes_update == "accepted action notes"
        notes: Any = next(item for item in attempt.model_request.components if item.id == "environment.notes")
        assert dict(notes.variables)["notes"] == (seed or "")
        assert all(
            "DO_NOT_COPY_LEGACY_MEMORY" not in message.content
            for message in attempt.model_request.messages
        )
