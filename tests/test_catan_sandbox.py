import asyncio
from dataclasses import dataclass

import pytest

from cle.env.observation_formatter import CatanObservationFormatter
from cle.players.baseline import FirstLegalPlayer, ScriptedPlayer
from cle.players.contracts import PlayerAttempt, PlayerChoice
from cle.sandbox import CatanSandbox, RetryPolicy
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.state import ensure_trade_window
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeCandidate, TradeOffer, TradeWindow


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _trade_offer(give=(1, 0, 0, 0, 0), receive=(0, 0, 0, 0, 1)):
    return TradeOffer(
        offered_by=Color.RED,
        audience=frozenset(COLORS[1:]),
        give=give,
        receive=receive,
    )


def _sandbox(red=None):
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    players = {
        color: (red if color == Color.RED and red is not None else FirstLegalPlayer(color))
        for color in COLORS
    }
    return CatanSandbox(engine, players), players


def test_settlement_action_descriptions_do_not_guess_opponent_nearness():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.step(engine.state.playable_actions[0])
    engine.step(engine.state.playable_actions[0])
    observation = engine.observe(Color.BLUE)
    settlement_actions = tuple(
        action
        for action in observation.valid_actions
        if action.action_type == ActionType.BUILD_SETTLEMENT
    )
    red_settlement = observation.opponent_settlements[Color.RED][0]

    assert engine.state.current_color() == Color.BLUE
    assert settlement_actions
    assert any(
        abs(action.value - red_settlement) <= 5
        for action in settlement_actions
    )
    descriptions = [
        CatanObservationFormatter()._format_single_action(action, observation)
        for action in settlement_actions
    ]
    assert all("Near:" not in description for description in descriptions)


def test_view_is_pure_and_only_exposes_legal_menu_to_current_player():
    sandbox, _ = _sandbox()

    red_before = sandbox.view(Color.RED)
    blue_before = sandbox.view(Color.BLUE)
    red_after = sandbox.view(Color.RED)

    assert red_before == red_after
    assert sandbox.revision == 0
    assert red_before.legal_actions
    assert blue_before.legal_actions == ()
    assert red_before.events == ()
    assert blue_before.events == ()


@pytest.mark.asyncio
async def test_step_applies_sole_roll_without_asking_player():
    class NoChoosePlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            raise AssertionError("forced ROLL must not ask the player")

        async def communicate(self, context):
            raise AssertionError("forced ROLL must not ask for pre-action speech")

    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.playable_actions = generate_playable_actions(engine.state)
    red = NoChoosePlayer(Color.RED)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = red
    sandbox = CatanSandbox(engine, players)
    player_snapshot = red.snapshot()

    result = await sandbox.step()

    assert [action.action_type for action in engine.state.actions] == [ActionType.ROLL]
    assert result.transitions[0].requested_action == Action(
        Color.RED,
        ActionType.ROLL,
        None,
    )
    assert result.transitions[0].resolved_action.action_type == ActionType.ROLL
    assert result.contexts == ()
    assert result.attempts == ()
    assert red.snapshot() == player_snapshot
    assert engine.state.last_dice_roll is not None


@pytest.mark.asyncio
async def test_step_asks_player_when_knight_is_legal_before_roll():
    class RollChoosingPlayer(FirstLegalPlayer):
        choose_calls = 0

        async def choose(self, context, feedback=None):
            self.choose_calls += 1
            roll_index = next(
                index
                for index, action in enumerate(context.legal_actions)
                if action.action_type == ActionType.ROLL
            )
            return PlayerAttempt(
                context_id=context.context_id,
                choice=PlayerChoice(action_index=roll_index),
            )

    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.player_state["P0_KNIGHT_IN_HAND"] = 1
    engine.state.player_state["P0_KNIGHT_OWNED_AT_START"] = True
    engine.state.playable_actions = generate_playable_actions(engine.state)
    red = RollChoosingPlayer(Color.RED)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = red
    sandbox = CatanSandbox(engine, players)

    assert {action.action_type for action in engine.state.playable_actions} == {
        ActionType.PLAY_KNIGHT_CARD,
        ActionType.ROLL,
    }

    result = await sandbox.step()

    assert red.choose_calls == 1
    assert red.accepted_choices == 1
    assert len(result.contexts) == 1
    assert len(result.attempts) == 1
    assert result.transitions[0].requested_action.action_type == ActionType.ROLL


@pytest.mark.asyncio
async def test_step_uses_exact_engine_menu_and_acknowledges_player():
    red = ScriptedPlayer(Color.RED, choices=[1])
    sandbox, _ = _sandbox(red)
    expected = sandbox.view(Color.RED).legal_actions[1]

    result = await sandbox.step()

    assert result.transitions[0].requested_action == expected
    assert result.transitions[0].resolved_action == expected
    assert red.accepted_choices == 1
    assert sandbox.revision == 1


@pytest.mark.asyncio
async def test_snapshot_restore_recovers_engine_events_rng_and_players():
    sandbox, players = _sandbox()
    await sandbox.step()
    snapshot = sandbox.snapshot()
    events = sandbox.view(Color.RED).events

    await sandbox.step()
    assert sandbox.revision == 2

    sandbox.restore(snapshot)

    assert sandbox.revision == 1
    assert sandbox.view(Color.RED).events == events
    assert players[Color.RED].accepted_choices == 1
    assert players[Color.RED].event_cursor == 0


@dataclass
class InvalidThenValidPlayer(FirstLegalPlayer):
    attempts: int = 0

    async def choose(self, context, feedback=None):
        self.attempts += 1
        index = 999 if self.attempts == 1 else 0
        return PlayerAttempt(
            context_id=context.context_id,
            choice=PlayerChoice(action_index=index),
        )


@pytest.mark.asyncio
async def test_step_retries_out_of_menu_choice_without_mutating_engine():
    red = InvalidThenValidPlayer(Color.RED)
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = red
    sandbox = CatanSandbox(
        engine,
        players,
        retry_policy=RetryPolicy(max_decision_attempts=2),
    )

    result = await sandbox.step()

    assert result.after_revision == 1
    assert red.attempts == 2
    assert len(sandbox.decision_trace) == 1
    assert len(engine.state.actions) == 1


@pytest.mark.asyncio
async def test_trade_barrier_waits_concurrently_and_applies_in_table_order():
    active = 0
    peak = 0

    class DelayedPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep({Color.BLUE: 0.03, Color.WHITE: 0.02, Color.ORANGE: 0.01}[self.color])
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
async def test_multiple_players_signal_willingness_but_turn_player_may_decline():
    class WillingPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
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
        async def choose(self, context, feedback=None):
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

    assert [
        transition.requested_action.action_type
        for transition in barrier.transitions
    ] == [
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
async def test_parameterized_trade_choice_becomes_strict_engine_action():
    class TradePlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
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

    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
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
    resolved_offer = result.transitions[0].resolved_action.value
    assert requested_offer.give == (1, 0, 0, 0, 0)
    assert requested_offer.receive == (0, 1, 0, 0, 0)
    assert resolved_offer.id is not None
    assert engine.state.trade_window.active_offers == (resolved_offer,)


def test_turn_player_may_ignore_all_willing_trade_partners():
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
    offer = window.create_offer(_trade_offer())
    window.signal_willingness(offer.id, Color.BLUE)
    engine.state.trade_window = window
    engine.state.playable_actions = generate_playable_actions(engine.state)

    confirmations = [
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


def test_turn_player_selects_exactly_one_willing_counterparty():
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


def test_sandbox_hot_path_has_no_threading_locks_or_duplicate_event_buffer():
    sandbox, _ = _sandbox()

    assert not hasattr(sandbox, "_lock")
    assert not hasattr(sandbox, "_step_lock")
    assert not hasattr(sandbox, "_events")
    assert not hasattr(sandbox, "_accepted_decisions")
