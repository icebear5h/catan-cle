"""Trade offer, barrier, and typed counter boundaries."""
import asyncio
from typing import Any

import pytest

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer
from cle.harness.context import PlayerResponseParseError, PlayerResponseParser
from cle.harness.models import ModelResponse
from cle.harness.suite import default_suite_path, load_context_suite
from cle.players.contracts import (
    PlayerAttempt,
    PlayerChoice,
    PlayerContext,
)
from cle.sandbox import CatanSandbox
from cle.sandbox.catan import PlayerResponseError

from .support import COLORS, BoundaryDefect, open_root


@pytest.mark.filterwarnings("ignore:Prompt suite .* is deprecated:DeprecationWarning")
def test_duplicate_trade_offer_xml_must_be_rejected(trade_sandbox: CatanSandbox) -> None:
    context = trade_sandbox.decision_context()
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))
    index: Any = next(
        i
        for i, action in enumerate(context.legal_actions)
        if action.action_type == ActionType.OFFER_TRADE
    )
    text = (
        f"<action>{index}</action>"
        '<trade_offer>{"give":{"WOOD":1},"receive":{"ORE":1}}</trade_offer>'
        '<trade_offer>{"give":{"WOOD":4},"receive":{"ORE":1}}</trade_offer>'
    )
    try:
        choice: Any = PlayerResponseParser(suite).parse(
            context, ModelResponse(content=text)
        )
    except PlayerResponseParseError:
        return
    assert choice.action_index == index
    assert choice.trade_offer.give == (1, 0, 0, 0, 0)
    assert choice.trade_offer.receive == (0, 0, 0, 0, 1)
    raise BoundaryDefect("Conflicting trade terms silently became the first offer")


@pytest.mark.asyncio
async def test_mutating_returned_offer_must_not_partially_commit_barrier(
    trade_sandbox: CatanSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox: Any = trade_sandbox
    engine: Any = sandbox.game_engine
    root: Any = open_root(sandbox)
    offer = TradeOffer(
        Color.WHITE,
        frozenset({Color.RED}),
        (0, 0, 0, 0, 1),
        (2, 0, 0, 0, 0),
        parent_offer_id=root.id,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    orange_choose = sandbox.players[Color.ORANGE].choose

    async def white_choose(context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
        index = next(
            i
            for i, action in enumerate(context.legal_actions)
            if action.action_type == ActionType.COUNTER_OFFER
        )
        return PlayerAttempt(context.context_id, PlayerChoice(index, trade_offer=offer))

    async def delayed_choose(context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
        entered.set()
        await release.wait()
        return await orange_choose(context, feedback)

    monkeypatch.setattr(sandbox.players[Color.WHITE], "choose", white_choose)
    monkeypatch.setattr(sandbox.players[Color.ORANGE], "choose", delayed_choose)
    before = engine.revision
    pending = asyncio.create_task(sandbox.step())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert engine.is_action_valid(Action(Color.WHITE, ActionType.COUNTER_OFFER, offer))
        offer.give = (0, 0, 0, 0, 99)
        assert not engine.is_action_valid(Action(Color.WHITE, ActionType.COUNTER_OFFER, offer))
        release.set()
        try:
            result = await asyncio.wait_for(pending, 2)
        except PlayerResponseError:
            assert engine.revision == before
            assert all(player.accepted_choices == 0 for player in sandbox.players.values())
            return
        except ValueError as exc:
            assert "not playable right now" in str(exc)
            assert "color=C.WHITE, action_type=AT.COUNTER_OFFER" in str(exc)
            if engine.revision != before:
                assert engine.revision == before + 1
                assert engine.state.actions[-1] == Action(
                    Color.BLUE, ActionType.REJECT_TRADE, root.id
                )
                assert engine.state.trade_window.offers[root.id].declined_by == {Color.BLUE}
                assert [sandbox.players[c].accepted_choices for c in COLORS] == [0, 1, 0, 0]
                raise BoundaryDefect(
                    "BLUE committed before mutated WHITE counteroffer failed"
                ) from exc
            assert all(player.accepted_choices == 0 for player in sandbox.players.values())
            return
        assert len(result.transitions) == 3
        assert result.transitions[1].requested_action.value.give == (0, 0, 0, 0, 1)
        assert [sandbox.players[c].accepted_choices for c in COLORS] == [0, 1, 1, 1]
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_typed_counter_parent_must_match_selected_menu(trade_sandbox: CatanSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox: Any = trade_sandbox
    old: Any = open_root(sandbox)
    await sandbox.step()
    assert sandbox.game_engine.state.trade_window.offers[old.id].declined_by == set(COLORS[1:])
    new = open_root(sandbox, wood=2)
    original_choose = sandbox.players[Color.BLUE].choose
    selected = []

    async def choose(context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
        if feedback:
            return await original_choose(context, feedback)
        assert all(old.id not in str(action.value) for action in context.legal_actions)
        index = next(
            i
            for i, action in enumerate(context.legal_actions)
            if action.action_type == ActionType.COUNTER_OFFER
        )
        selected.append(context.legal_actions[index])
        return PlayerAttempt(
            context.context_id,
            PlayerChoice(
                index,
                trade_offer=TradeOffer(
                    Color.BLUE,
                    frozenset({Color.RED}),
                    (0, 0, 0, 0, 1),
                    (3, 0, 0, 0, 0),
                    parent_offer_id=old.id,
                ),
            ),
        )

    monkeypatch.setattr(sandbox.players[Color.BLUE], "choose", choose)
    result = await sandbox.step()
    assert len(selected) == 1
    assert selected[0].value.startswith(f"COUNTER_OFFER:{new.id}:")
    committed = result.transitions[0].requested_action
    assert committed.color == Color.BLUE
    if committed.action_type == ActionType.COUNTER_OFFER:
        if committed.value.parent_offer_id != new.id:
            assert committed.value.parent_offer_id == old.id
            raise BoundaryDefect("Selected counter for o2 committed a counter for answered o1")
    else:
        assert committed == Action(Color.BLUE, ActionType.REJECT_TRADE, new.id)
        assert sandbox.decision_trace[-1].validation_error
