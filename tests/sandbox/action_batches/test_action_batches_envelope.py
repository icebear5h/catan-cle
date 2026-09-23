"""Envelope syntax, notes, and historical contracts."""
import asyncio
import json
import pickle
from dataclasses import fields, replace
from typing import Any

import pytest

from cle.game_engine.models.player import Color
from cle.harness.context import PlayerResponseParseError, PlayerResponseParser
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.shared_suite import load_shared_prompt_suite
from cle.harness.suite import load_context_suite
from cle.players.contracts import PlayerChoice
from cle.sandbox.catan import PlayerResponseError

from .support import batch, pair, sandbox


@pytest.mark.parametrize("tail", [
    {"tool": "say", "arguments": {}},
    {"tool": "offer_trade", "arguments": {"confirm_if_accepted_by": "ANY"}},
    {"tool": "confirm_trade", "arguments": {}},
    {"tool": "buy_development_card", "arguments": {}},
    {"tool": "roll_dice", "arguments": {}},
    {"tool": "play_knight", "arguments": {"tile": "<T00>"}},
    {"tool": "build_road", "arguments": {"edge": "<E00_01>", "notes": "buried"}},
    {"tool": "build_road", "arguments": {"edge": 1}},
    {"tool": "maritime_trade", "arguments": {"give": {"WOOD": True}, "receive": {"ORE": 1}}},
    {"tool": "maritime_trade", "arguments": {"give": {"WOOD": 4}, "receive": {"ORE": 2}}},
])
def test_whole_batch_syntax_is_checked_before_first_action(tail: dict[str, Any]) -> None:
    game, _ = sandbox()
    first, _ = pair(game.game_engine)
    parser = PlayerResponseParser(load_shared_prompt_suite().decision_suite())
    with pytest.raises(PlayerResponseParseError):
        parser.parse(game.decision_context(), ModelResponse(batch(first, tail)))
    assert game.revision == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("notes", [None, 3, {}, "x" * 4001])
async def test_invalid_notes_reject_entire_batch_before_state_or_memory(notes: object) -> None:
    game, transports = sandbox()
    transports[Color.RED].contents = [batch(*pair(game.game_engine), notes=notes)]
    before = pickle.dumps(game.snapshot())
    with pytest.raises(PlayerResponseError):
        await game.step()
    assert pickle.dumps(game.snapshot()) == before


@pytest.mark.parametrize("payload", [
    {"actions": []}, {"actions": [{}] * 5}, {"actions": {}},
    {"actions": [{"tool": "end_turn", "arguments": {}}, {"tool": "end_turn", "arguments": {}}]},
    {"actions": [{"tool": "end_turn", "arguments": {}, "notes": "buried"}]},
    {"actions": [{"tool": "end_turn", "arguments": {}}], "tool": "end_turn", "arguments": {}},
])
def test_strict_envelope(payload: dict[str, Any] | dict[str, list[dict[str, Any]]]) -> None:
    game, _ = sandbox()
    with pytest.raises(PlayerResponseParseError):
        PlayerResponseParser(load_shared_prompt_suite().decision_suite()).parse(game.decision_context(), ModelResponse(json.dumps(payload)))


def test_historical_contracts_do_not_admit_batches_and_old_slots_default_empty() -> None:
    game, _ = sandbox()
    response = ModelResponse(batch(*pair(game.game_engine)))
    suites = [load_context_suite("cle/harness/suites/catan_v11.yaml"), load_context_suite("cle/harness/suites/catan_v10.yaml")]
    old_shared = load_shared_prompt_suite().model_copy(update={"deterministic_batches": False}).decision_suite()
    for suite in [*suites, old_shared]:
        with pytest.raises(PlayerResponseParseError):
            PlayerResponseParser(suite).parse(game.decision_context(), response)
    for value, name, expected in [(game.snapshot(), "pending_action_batch", None), (PlayerChoice(0), "batch_actions", ())]:
        historical: Any = [getattr(value, field.name) for field in fields(value) if field.name != name]
        restored: Any = object.__new__(type(value))
        restored.__setstate__(historical)
        assert getattr(restored, name) == expected


@pytest.mark.asyncio
async def test_bad_saved_queue_rejected_without_mutating_game() -> None:
    game, transports = sandbox()
    transports[Color.RED].contents = [batch(*pair(game.game_engine))]
    await game.step()
    saved: Any = game.snapshot()
    before: Any = pickle.dumps(saved)
    for bad in [replace(saved.pending_action_batch, next_index=2), replace(saved.pending_action_batch, actor=Color.BLUE)]:
        with pytest.raises(ValueError):
            game.restore(replace(saved, pending_action_batch=bad))
        assert pickle.dumps(game.snapshot()) == before


@pytest.mark.asyncio
async def test_cancelled_inference_admits_no_plan() -> None:
    game, transports = sandbox()
    started = asyncio.Event()

    async def wait(request: ModelRequest) -> None:
        started.set()
        await asyncio.Future()

    transports[Color.RED].complete = wait
    task = asyncio.create_task(game.step())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert game.snapshot().pending_action_batch is None
    assert game.revision == 0


def test_duplicate_keys_escaped_tokens_and_buried_notes_rejected() -> None:
    game, _ = sandbox()
    response = batch(*pair(game.game_engine))
    parser = PlayerResponseParser(load_shared_prompt_suite().decision_suite())
    malformed = [
        response.replace('"actions":', '"actions":[],"actions":'),
        response.replace('"edge":', '"edg\\u0065":'),
        response.replace('"<E', '"\\u003cE'),
        response.replace('"arguments":', '"notes":"buried","arguments":', 1),
    ]
    for text in malformed:
        with pytest.raises(PlayerResponseParseError):
            parser.parse(game.decision_context(), ModelResponse(text))
