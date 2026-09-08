from dataclasses import replace

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions, trade_response_actions
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state import ensure_trade_window
from cle.game_engine.trading import TradeOffer
from cle.sandbox.decision import build_decision_context


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def test_default_decision_menu_belongs_to_current_actor():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)

    context = build_decision_context(engine)

    assert context.actor == Color.RED
    assert context.legal_actions == tuple(engine.state.playable_actions)
    assert context.observation.valid_actions == list(context.legal_actions)


def test_off_turn_context_does_not_expose_private_development_card_menu():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.player_state["P0_KNIGHT_IN_HAND"] = 1
    engine.state.player_state["P0_KNIGHT_OWNED_AT_START"] = True
    engine.state.playable_actions = generate_playable_actions(engine.state)
    assert any(a.action_type == ActionType.PLAY_KNIGHT_CARD for a in engine.state.playable_actions)

    with pytest.raises(ValueError, match="No legal actions"):
        build_decision_context(engine, Color.BLUE)


def test_explicit_empty_menu_does_not_fall_back_to_engine_menu():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)

    with pytest.raises(ValueError, match="No legal actions"):
        build_decision_context(engine, advertised_actions=())


def test_mismatched_action_actor_is_rejected():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)

    with pytest.raises(ValueError, match="must belong"):
        build_decision_context(engine, Color.BLUE, tuple(engine.state.playable_actions))


def test_off_turn_trade_context_uses_only_explicit_actor_owned_menu():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    window = ensure_trade_window(engine.state)
    offer = window.create_offer(
        TradeOffer(
            offered_by=Color.RED,
            audience=frozenset(COLORS[1:]),
            give=(1, 0, 0, 0, 0),
            receive=(0, 0, 0, 0, 1),
        )
    )
    actions = tuple(trade_response_actions(engine.state, Color.BLUE))

    context = build_decision_context(engine, Color.BLUE, actions)

    assert context.legal_actions == (Action(Color.BLUE, ActionType.REJECT_TRADE, offer.id),)
    assert all(action.color == Color.BLUE for action in context.observation.valid_actions)


@pytest.mark.parametrize("override", [False, True])
def test_terminal_guard_precedes_even_an_explicit_advertised_menu(override):
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    actions = tuple(engine.state.playable_actions)
    engine.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = engine.vps_to_win
    assert engine.winning_color() == Color.RED
    with pytest.raises(ValueError, match="terminal"):
        build_decision_context(engine, advertised_actions=actions if override else None)


@pytest.mark.parametrize("actor", COLORS)
def test_terminal_preview_opt_in_restores_only_current_actors_exact_menu(actor):
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    index = COLORS.index(actor)
    engine.state.current_turn_index = index
    engine.state.current_player_index = index
    engine.state.playable_actions = generate_playable_actions(engine.state)
    actions = tuple(engine.state.playable_actions)
    engine.state.player_state[f"P{index}_ACTUAL_VICTORY_POINTS"] = engine.vps_to_win
    assert engine.winning_color() == actor
    assert engine.observe(actor).valid_actions == []

    context = build_decision_context(engine, actor, context_revision=99, allow_terminal=True)

    assert context.actor == actor
    assert context.context_id.endswith(f":99:{actor.value}")
    assert context.legal_actions == actions
    assert context.observation.valid_actions == list(actions)
    assert engine.winning_color() == actor
    assert engine.observe(actor).valid_actions == []
    assert tuple(engine.state.playable_actions) == actions
    with pytest.raises(ValueError, match="No legal actions"):
        build_decision_context(engine, actor, advertised_actions=(), allow_terminal=True)


def test_terminal_preview_opt_in_does_not_reveal_offturn_private_menu():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    state = engine.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.player_state["P0_KNIGHT_IN_HAND"] = 1
    state.player_state["P0_KNIGHT_OWNED_AT_START"] = True
    window = ensure_trade_window(state)
    root = window.create_offer(TradeOffer(
        Color.RED, frozenset(COLORS[1:]), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1),
    ))
    state.playable_actions = generate_playable_actions(state)
    state.player_state["P0_ACTUAL_VICTORY_POINTS"] = engine.vps_to_win
    assert any(action.action_type == ActionType.PLAY_KNIGHT_CARD for action in state.playable_actions)

    with pytest.raises(ValueError, match="No legal actions"):
        build_decision_context(engine, Color.BLUE, allow_terminal=True)
    with pytest.raises(ValueError, match="must belong"):
        build_decision_context(engine, Color.BLUE, tuple(state.playable_actions), allow_terminal=True)
    actions = tuple(trade_response_actions(state, Color.BLUE))
    context = build_decision_context(engine, Color.BLUE, actions, allow_terminal=True)
    assert context.legal_actions == (Action(Color.BLUE, ActionType.REJECT_TRADE, root.id),)
    assert context.observation.valid_actions == list(actions)
    assert context.observation.my_dev_cards["KNIGHT"] == 0


@pytest.mark.parametrize("flag", [None, 1, "true"])
def test_terminal_preview_flag_requires_an_explicit_boolean(flag):
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    with pytest.raises(ValueError, match="allow_terminal must be a boolean"):
        build_decision_context(engine, allow_terminal=flag)


@pytest.mark.parametrize("actor", ["", False, Color.BLACK])
def test_terminal_preview_does_not_default_an_invalid_actor_to_current_player(actor):
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    with pytest.raises(ValueError, match="not a participant"):
        build_decision_context(engine, actor, allow_terminal=True)


@pytest.mark.parametrize("actor", COLORS)
@pytest.mark.parametrize("overflow", [False, True])
def test_decision_social_facts_are_projected_bounded_and_detached(actor, overflow):
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.communication_limits = replace(engine.communication_limits, recent_message_window=2)
    engine.append_message(
        speaker=Color.BLUE, text="private promise", audience=(Color.RED,), intent="BRIBE",
        causation_id="private", commitment=("spare BLUE", "BLUE offers ORE", 9),
    )
    for index in range(3 if overflow else 1):
        engine.append_message(
            speaker=Color.RED, text=f"public-{index}", audience=COLORS[1:], intent="TRADE",
            causation_id=f"public:{index}",
        )
    engine.step(engine.state.playable_actions[0])
    context = build_decision_context(
        engine, actor, (Action(actor, ActionType.END_TURN, None),)
    )
    expected = (
        ["public-1", "public-2"] if overflow
        else ["private promise", "public-0"] if actor in (Color.RED, Color.BLUE)
        else ["public-0"]
    )
    assert [event.payload["text"] for event in context.recent_messages] == expected
    assert len(context.events) == 1 and context.events[0].event_type == "BUILD_SETTLEMENT"
    assert len(context.active_commitments) == int(actor in (Color.RED, Color.BLUE))
    if context.active_commitments:
        assert context.active_commitments[0].promise == "BLUE offers ORE"
        context.active_commitments[0].promise = "edited"
        assert engine.commitments[0].promise == "BLUE offers ORE"
    context.recent_messages[0].payload["text"] = "edited"
    assert engine.project_messages(actor)[0].payload["text"] == expected[0]


def test_decision_discard_count_uses_the_actors_own_hand():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.player_state["P1_WOOD_IN_HAND"] = 5
    engine.state.player_state["P1_ORE_IN_HAND"] = 4
    context = build_decision_context(
        engine, Color.BLUE, (Action(Color.BLUE, ActionType.DISCARD, None),)
    )
    assert context.discard_count == 4
    assert context.observation.my_resources["WOOD"] == 5
    assert build_decision_context(engine).discard_count == 0
