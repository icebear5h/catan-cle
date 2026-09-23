"""Authorization matching, parsing, and restore."""
import asyncio
import pickle
from dataclasses import fields, replace
from typing import Any

import pytest

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.harness.action_tools import parse_tool_choice
from cle.harness.context import PlayerResponseParseError, PlayerResponseParser
from cle.harness.models import ModelResponse
from cle.harness.shared_suite import load_shared_prompt_suite
from cle.harness.suite import load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.contracts import PlayerChoice
from cle.players.validation import action_from_choice
from cle.sandbox import CatanSandbox
from cle.sandbox.catan import PostActionCommunicationCancelled
from cle.sandbox.contracts import SandboxSnapshot

from .support import TERMS, reply, sandbox


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_after", ["offer", "barrier"])
async def test_post_action_cancellation_and_policy_rebinding_retain_admitted_instruction(monkeypatch: pytest.MonkeyPatch, cancel_after: str) -> None:
    game, transports = sandbox(["ORANGE", "BLUE"])
    original = game._run_communication
    if cancel_after == "barrier":
        await game.step()

    async def cancel(*args: object, **kwargs: object) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(game, "_run_communication", cancel)
    with pytest.raises(PostActionCommunicationCancelled):
        await game.step()
    assert game.players[Color.RED].session.memory_revision == 1
    snapshot = pickle.loads(pickle.dumps(game.snapshot()))
    game.restore(snapshot)
    monkeypatch.setattr(game, "_run_communication", original)

    def rebind(runtime: CatanSandbox) -> None:
        old: Any = runtime.players[Color.RED]
        replacement = AgentPlayer(Color.RED, transports[Color.RED], session_id=old.session.session_id)
        replacement.restore(old.snapshot())
        runtime.players[Color.RED] = replacement

    game._refresh_players = rebind
    if cancel_after == "offer":
        await game.step()
    else:
        rebind(game)

    def must_not_refresh(runtime: CatanSandbox) -> None:
        raise AssertionError("Automatic execution must not reinterpret the admitted model call")

    game._refresh_players = must_not_refresh
    result = await game.step()
    assert result.transitions[0].requested_action.value.counterparty == Color.ORANGE
    assert game.players[Color.RED].session.memory_revision == 1
    assert len(transports[Color.RED].requests) == 1


@pytest.mark.asyncio
async def test_saved_authorization_must_match_original_offer_event() -> None:
    game, _ = sandbox()
    await game.step()
    snapshot: Any = game.snapshot()
    before = pickle.dumps(snapshot)
    altered = replace(snapshot.trade_preauthorization, give=(2, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="original offer event"):
        game.restore(replace(snapshot, trade_preauthorization=altered))
    assert pickle.dumps(game.snapshot()) == before


@pytest.mark.parametrize("bad", [[], ["BLUE", "blue"], ["RED"], ["GOLD"], ["BLUE", 1], "BLUE", "any", "if BLUE accepts", None, {}, True])
def test_shared_parser_rejects_invalid_authorization_without_mutation(bad: object) -> None:
    game, _ = sandbox()
    before = pickle.dumps(game.snapshot())
    parser = PlayerResponseParser(load_shared_prompt_suite().decision_suite())
    with pytest.raises(PlayerResponseParseError):
        parser.parse(game.decision_context(), ModelResponse(reply("offer_trade", {**TERMS, "confirm_if_accepted_by": bad})))
    assert pickle.dumps(game.snapshot()) == before


def test_authorization_is_shared_only_exact_root_offer_and_old_slots_restore() -> None:
    game, _ = sandbox()
    context: Any = game.decision_context()
    args: Any = {**TERMS, "confirm_if_accepted_by": ["BLUE"]}
    with pytest.raises(ValueError):
        parse_tool_choice(context, "offer_trade", args)
    historical: Any = PlayerResponseParser(load_context_suite("cle/harness/suites/catan_v11.yaml"))
    with pytest.raises(PlayerResponseParseError):
        historical.parse(context, ModelResponse(reply("offer_trade", args)))
    for args in ({**args, "give_any": 1}, {**args, "receive_any": 1}, {**args, "condition": "nobody accepts"}):
        with pytest.raises(ValueError):
            parse_tool_choice(context, "offer_trade", args, shared=True)
    choice: Any = parse_tool_choice(context, "offer_trade", TERMS, shared=True)
    narrow = replace(choice.trade_offer, audience=frozenset({Color.BLUE}))
    narrow_context = replace(context, legal_actions=(Action(Color.RED, ActionType.OFFER_TRADE, narrow),))
    with pytest.raises(ValueError, match="audience"):
        action_from_choice(narrow_context, PlayerChoice(0, confirm_if_accepted_by=(Color.WHITE,)))
    old = PlayerChoice(0)
    restored = object.__new__(PlayerChoice)
    restored.__setstate__([getattr(old, field.name) for field in fields(old)[:-1]])
    assert restored.confirm_if_accepted_by is None
    old_snapshot = game.snapshot()
    restored_snapshot = object.__new__(SandboxSnapshot)
    restored_snapshot.__setstate__([getattr(old_snapshot, field.name) for field in fields(old_snapshot)[:-1]])
    assert restored_snapshot.trade_preauthorization is None
