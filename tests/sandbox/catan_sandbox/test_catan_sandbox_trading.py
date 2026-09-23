"""Trade barriers, willingness, and counterparty selection."""
import asyncio
from typing import Any

import pytest

from cle.env.observation_formatter import CatanObservationFormatter
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state import ensure_trade_window
from cle.game_engine.trading import TradeCandidate, TradeWindow
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    PlayerAttempt,
    PlayerChoice,
    PlayerContext,
)
from cle.sandbox import CatanSandbox

from .support import COLORS, _trade_offer


@pytest.mark.asyncio
async def test_trade_barrier_waits_concurrently_and_applies_in_table_order() -> None:
    active = 0
    peak = 0

    class DelayedPlayer(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(
                {Color.BLUE: 0.03, Color.WHITE: 0.02, Color.ORANGE: 0.01}[self.color]
            )
            active -= 1
            return await super().choose(context, feedback)

    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    window = ensure_trade_window(engine.state)
    window.create_offer(
        _trade_offer(receive=(0, 1, 0, 0, 0)),
    )
    players = {Color.RED: FirstLegalPlayer(Color.RED)}
    players.update({color: DelayedPlayer(color) for color in COLORS[1:]})
    sandbox = CatanSandbox(engine, players)

    result = await sandbox.step()

    assert peak == 3
    assert [transition.requested_action.color for transition in result.transitions] == [
        Color.BLUE,
        Color.WHITE,
        Color.ORANGE,
    ]
    assert all(
        transition.requested_action.action_type.value == "REJECT_TRADE"
        for transition in result.transitions
    )


@pytest.mark.asyncio
async def test_multiple_players_signal_willingness_but_turn_player_may_decline() -> None:
    class WillingPlayer(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            index = next(
                (
                    index
                    for index, action in enumerate(context.legal_actions)
                    if action.action_type == ActionType.ACCEPT_TRADE
                ),
                0,
            )
            return PlayerAttempt(
                context_id=context.context_id,
                choice=PlayerChoice(action_index=index),
            )

    class DecliningTurnPlayer(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            index = next(
                index
                for index, action in enumerate(context.legal_actions)
                if action.action_type == ActionType.END_TURN
            )
            return PlayerAttempt(
                context_id=context.context_id,
                choice=PlayerChoice(action_index=index),
            )

    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.player_state["P0_HAS_ROLLED"] = True
    engine.state.player_state["P0_WOOD_IN_HAND"] = 1
    engine.state.player_state["P1_ORE_IN_HAND"] = 1
    engine.state.player_state["P2_ORE_IN_HAND"] = 1
    window = TradeWindow(
        id="trade-window-test",
        turn_player=Color.RED,
        participants=COLORS,
    )
    offer = window.create_offer(_trade_offer())
    engine.state.trade_window = window
    engine.state.playable_actions = generate_playable_actions(engine.state)
    players = {
        Color.RED: DecliningTurnPlayer(Color.RED),
        Color.BLUE: WillingPlayer(Color.BLUE),
        Color.WHITE: WillingPlayer(Color.WHITE),
        Color.ORANGE: WillingPlayer(Color.ORANGE),
    }
    sandbox = CatanSandbox(engine, players)

    barrier = await sandbox.step()

    assert [transition.requested_action.action_type for transition in barrier.transitions] == [
        ActionType.ACCEPT_TRADE,
        ActionType.ACCEPT_TRADE,
        ActionType.REJECT_TRADE,
    ]
    assert offer.willing_by == {Color.BLUE, Color.WHITE}
    assert window.selected_candidate is None

    declined = await sandbox.step()

    assert declined.transitions[0].requested_action.action_type == ActionType.END_TURN
    assert engine.state.player_state["P0_WOOD_IN_HAND"] == 1
    assert engine.state.player_state["P1_ORE_IN_HAND"] == 1
    assert engine.state.player_state["P2_ORE_IN_HAND"] == 1


@pytest.mark.asyncio
async def test_parameterized_trade_choice_becomes_strict_engine_action() -> None:
    class TradePlayer(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            index = next(
                index
                for index, action in enumerate(context.legal_actions)
                if action.action_type.value == "OFFER_TRADE"
            )
            return PlayerAttempt(
                context_id=context.context_id,
                choice=PlayerChoice(
                    action_index=index,
                    trade_offer=_trade_offer(receive=(0, 1, 0, 0, 0)),
                ),
            )

    engine: Any = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.player_state["P0_HAS_ROLLED"] = True
    engine.state.player_state["P0_WOOD_IN_HAND"] = 1
    engine.state.playable_actions = generate_playable_actions(engine.state)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = TradePlayer(Color.RED)
    sandbox = CatanSandbox(engine, players)

    result = await sandbox.step()

    assert result.transitions[0].requested_action.action_type.value == "OFFER_TRADE"
    requested_offer = result.transitions[0].requested_action.value
    resolved_offer: Any = result.transitions[0].resolved_action.value
    assert requested_offer.give == (1, 0, 0, 0, 0)
    assert requested_offer.receive == (0, 1, 0, 0, 0)
    assert resolved_offer.id is not None
    assert engine.state.trade_window.active_offers == (resolved_offer,)


def test_turn_player_may_ignore_all_willing_trade_partners() -> None:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.player_state["P0_HAS_ROLLED"] = True
    engine.state.player_state["P0_WOOD_IN_HAND"] = 1
    engine.state.player_state["P1_ORE_IN_HAND"] = 1
    window = TradeWindow(
        id="trade-window-test",
        turn_player=Color.RED,
        participants=COLORS,
    )
    offer: Any = window.create_offer(_trade_offer())
    window.signal_willingness(offer.id, Color.BLUE)
    engine.state.trade_window = window
    engine.state.playable_actions = generate_playable_actions(engine.state)

    confirmations: Any = [
        action
        for action in engine.state.playable_actions
        if action.action_type == ActionType.CONFIRM_TRADE
    ]
    end_turn = next(
        action
        for action in engine.state.playable_actions
        if action.action_type == ActionType.END_TURN
    )

    assert confirmations == [
        Action(
            Color.RED,
            ActionType.CONFIRM_TRADE,
            TradeCandidate(offer.id, Color.RED, Color.BLUE),
        )
    ]
    engine.step(end_turn)
    assert engine.state.player_state["P0_WOOD_IN_HAND"] == 1
    assert engine.state.player_state["P1_ORE_IN_HAND"] == 1


def test_turn_player_selects_exactly_one_willing_counterparty() -> None:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.player_state["P0_HAS_ROLLED"] = True
    engine.state.player_state["P0_WOOD_IN_HAND"] = 1
    engine.state.player_state["P1_ORE_IN_HAND"] = 1
    engine.state.player_state["P2_ORE_IN_HAND"] = 1
    window: Any = TradeWindow(
        id="trade-window-test",
        turn_player=Color.RED,
        participants=COLORS,
    )
    offer = window.create_offer(_trade_offer())
    window.signal_willingness(offer.id, Color.BLUE)
    window.signal_willingness(offer.id, Color.WHITE)
    engine.state.trade_window = window
    engine.state.playable_actions = generate_playable_actions(engine.state)
    confirm_white = next(
        action
        for action in engine.state.playable_actions
        if action.action_type == ActionType.CONFIRM_TRADE
        and action.value.counterparty == Color.WHITE
    )
    description = CatanObservationFormatter()._format_single_action(
        confirm_white,
        engine.observe(Color.RED),
    )

    assert "WHITE" in description
    assert "give 1 WOOD" in description
    assert "receive 1 ORE" in description
    engine.step(confirm_white)

    assert engine.state.player_state["P0_WOOD_IN_HAND"] == 0
    assert engine.state.player_state["P0_ORE_IN_HAND"] == 1
    assert engine.state.player_state["P1_ORE_IN_HAND"] == 1
    assert engine.state.player_state["P1_WOOD_IN_HAND"] == 0
    assert engine.state.player_state["P2_ORE_IN_HAND"] == 0
    assert engine.state.player_state["P2_WOOD_IN_HAND"] == 1
    assert window.selected_candidate.counterparty == Color.WHITE
