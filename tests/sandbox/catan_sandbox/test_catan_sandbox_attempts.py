"""Attempt validation, callbacks, and typed exhaustion."""
import pickle
from typing import Any

import pytest

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    AcceptanceResult,
    PlayerAttempt,
    PlayerChoice,
    PlayerContext,
)
from cle.sandbox import CatanSandbox, RetryPolicy
from cle.sandbox.catan import PlayerResponseError

from .support import COLORS, NoCommunicationPolicy, _trade_engine, _trade_offer


@pytest.mark.asyncio
async def test_callback_context_and_acceptance_cannot_mutate_canonical_records() -> None:
    contexts = []
    returned: Any = []

    class MutatingPlayer(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            contexts.append(context)
            assert context.observation.my_resources["WOOD"] == 8
            assert context.observation.valid_actions == list(context.legal_actions)
            context.observation.my_resources["WOOD"] = 99
            context.observation.valid_actions.clear()
            if feedback is None:
                return PlayerAttempt(context.context_id, {"action_index": 0})
            index = next(
                i for i, action in enumerate(context.legal_actions)
                if action.action_type == ActionType.OFFER_TRADE
            )
            attempt = PlayerAttempt(context.context_id, PlayerChoice(index, trade_offer=_trade_offer()))
            returned.append(attempt)
            return attempt

        def accept(self, attempt: PlayerAttempt, result: AcceptanceResult) -> None:
            assert attempt is result.attempts[0]
            assert result.context.observation.my_resources["WOOD"] == 8
            attempt.choice.trade_offer.give = (99, 0, 0, 0, 0)
            result.context.observation.my_resources["WOOD"] = 100
            result.transitions[0].resolved_action.value.give = (100, 0, 0, 0, 0)
            result.transitions[0].events[0].public_payload["give"]["WOOD"] = 100
            super().accept(attempt, result)

    engine = _trade_engine()
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = MutatingPlayer(Color.RED)
    sandbox = CatanSandbox(engine, players, communication_policy=NoCommunicationPolicy())

    result = await sandbox.step()

    assert len(contexts) == 2 and contexts[0] is not contexts[1]
    assert sandbox.decision_trace[0].choice is None
    assert result.context.observation.my_resources["WOOD"] == 8
    assert result.context.observation.valid_actions == list(result.context.legal_actions)
    assert result.attempts[0].choice.trade_offer.give == (1, 0, 0, 0, 0)
    assert engine.events[0].public_payload["give"]["WOOD"] == 1
    before = pickle.dumps(engine.snapshot())
    returned[0].choice.trade_offer.give = (20, 0, 0, 0, 0)
    result.transitions[0].resolved_action.value.give = (30, 0, 0, 0, 0)
    result.transitions[0].events[0].public_payload["give"]["WOOD"] = 30
    assert pickle.dumps(engine.snapshot()) == before
    assert result.attempts[0].choice.trade_offer.give == (1, 0, 0, 0, 0)


@pytest.mark.asyncio
async def test_barrier_accept_callbacks_run_only_after_all_live_actions() -> None:
    engine = _trade_engine()
    root: Any = engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer())).resolved_action.value
    seen = []

    class ObservingPlayer(FirstLegalPlayer):
        def accept(self, attempt: PlayerAttempt, result: AcceptanceResult) -> None:
            assert engine.revision == 4
            assert engine.state.trade_window.offers[root.id].declined_by == set(COLORS[1:])
            seen.append(self.color)
            result.context.observation.trade_window.offers[root.id].give = (99, 0, 0, 0, 0)
            super().accept(attempt, result)

    sandbox = CatanSandbox(
        engine, {color: ObservingPlayer(color) for color in COLORS},
        communication_policy=NoCommunicationPolicy(),
    )
    result: Any = await sandbox.step()

    assert seen == list(COLORS[1:])
    assert len(result.transitions) == 3
    assert all(context.observation.trade_window.offers[root.id].give == (1, 0, 0, 0, 0) for context in result.contexts)
    assert engine.state.trade_window.offers[root.id].give == (1, 0, 0, 0, 0)


@pytest.mark.asyncio
async def test_barrier_revalidates_funding_after_all_replies_without_partial_commit() -> None:
    engine: Any = _trade_engine()
    root = engine.step(Action(Color.RED, ActionType.OFFER_TRADE, _trade_offer())).resolved_action.value

    class CounterPlayer(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            index = next(i for i, action in enumerate(context.legal_actions) if action.action_type == ActionType.COUNTER_OFFER)
            return PlayerAttempt(context.context_id, PlayerChoice(index, trade_offer=TradeOffer(
                self.color, frozenset({Color.RED}), (0, 0, 0, 0, 1), (2, 0, 0, 0, 0),
                parent_offer_id=root.id,
            )))

    class LastPlayer(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            # Simulate an out-of-band change after WHITE's initial admission.
            engine.state.player_state["P2_ORE_IN_HAND"] = 0
            return await super().choose(context, feedback)

    players: Any = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.WHITE] = CounterPlayer(Color.WHITE)
    players[Color.ORANGE] = LastPlayer(Color.ORANGE)
    sandbox = CatanSandbox(engine, players, retry_policy=RetryPolicy(1), communication_policy=NoCommunicationPolicy())

    with pytest.raises(PlayerResponseError) as caught:
        await sandbox.step()

    assert caught.value.player == Color.WHITE
    assert engine.revision == 1
    assert len(engine.state.actions) == 1
    assert engine.state.trade_window.offers[root.id].declined_by == set()
    assert all(player.accepted_choices == 0 for player in players.values())
