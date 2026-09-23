"""Strict engine execution across reduced menus."""
from collections import Counter
from copy import deepcopy
from typing import Any

import pytest

from cle.game_engine.board_tokens import edge_token, node_token, tile_token
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import (
    generate_playable_actions,
    trade_response_actions,
)
from cle.game_engine.models.enums import ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.harness.action_tools import parse_tool_choice, render_action_tools
from cle.players.validation import action_from_choice

from .support import COLORS, _case, _context, _turn_engine


def test_initial_spatial_tools_execute_through_strict_engine_without_menu_indices_in_guidance() -> None:
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    while engine.state.is_initial_build_phase:
        actor = engine.state.current_color()
        context: Any = _context(engine, actor=actor)
        action = engine.state.playable_actions[-1]
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            tool, arguments = "build_settlement", {"node": node_token(action.value)}
        else:
            tool, arguments = "build_road", {"edge": edge_token(action.value)}
        choice = parse_tool_choice(context, tool, arguments)
        resolved = action_from_choice(context, choice)
        assert engine.is_action_valid(resolved)
        assert resolved == action
        engine.step(resolved)


def test_reduced_main_turn_menu_tools_pass_strict_engine_execution() -> None:
    engine: Any = _turn_engine()
    context: Any = _context(engine)
    saw: Any = set()
    for action in context.legal_actions:
        kind: Any = action.action_type
        if kind in {ActionType.OFFER_TRADE, ActionType.MARITIME_TRADE}:
            tool: Any = "offer_trade" if kind == ActionType.OFFER_TRADE else "maritime_trade"
            arguments: Any = {"give": {"WOOD": 1}, "receive": {"ORE": 1}}
            if kind == ActionType.MARITIME_TRADE:
                arguments = {
                    "give": dict(Counter(r for r in action.value[:4] if r)),
                    "receive": {action.value[-1]: 1},
                }
        elif kind in {ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY, ActionType.BUILD_ROAD}:
            tool = {
                ActionType.BUILD_SETTLEMENT: "build_settlement",
                ActionType.BUILD_CITY: "upgrade_city",
                ActionType.BUILD_ROAD: "build_road",
            }[kind]
            arguments = (
                {"edge": edge_token(action.value)}
                if kind == ActionType.BUILD_ROAD
                else {"node": node_token(action.value)}
            )
        elif kind == ActionType.PLAY_YEAR_OF_PLENTY:
            tool, arguments = "play_year_of_plenty", {"take": dict(Counter(action.value))}
        elif kind == ActionType.PLAY_MONOPOLY:
            tool, arguments = "play_monopoly", {"resource": action.value}
        elif kind == ActionType.PLAY_KNIGHT_CARD:
            tool = "play_knight"
            tile: Any = next(
                tile
                for coord, tile in context.observation.board_map.land_tiles.items()
                if coord != context.observation.robber_position
            )
            arguments = {"tile": tile_token(tile.id)}
        else:
            tool = {
                ActionType.BUY_DEVELOPMENT_CARD: "buy_development_card",
                ActionType.PLAY_ROAD_BUILDING: "play_road_building",
                ActionType.END_TURN: "end_turn",
            }[kind]
            arguments = {}
        choice: Any = parse_tool_choice(context, tool, arguments)
        resolved: Any = action_from_choice(context, choice)
        assert engine.is_action_valid(resolved)
        branch: Any = deepcopy(engine)
        branch.step(resolved)
        saw.add(tool)
    assert {
        "play_knight",
        "play_monopoly",
        "play_year_of_plenty",
        "play_road_building",
        "buy_development_card",
        "maritime_trade",
        "offer_trade",
        "build_road",
        "upgrade_city",
        "end_turn",
    } <= saw


def test_trade_tools_execute_strict_root_counter_confirm_and_cancel_lifecycle() -> None:
    engine: Any = _turn_engine()
    context = _context(engine)
    offer = parse_tool_choice(context, "offer_trade", {"give": {"WOOD": 2}, "receive": {"ORE": 1}})
    engine.step(action_from_choice(context, offer))
    root_id: Any = next(iter(engine.state.trade_window.offers))
    cancel_context = _context(engine)
    cancelled: Any = deepcopy(engine)
    cancelled.step(
        action_from_choice(
            cancel_context,
            parse_tool_choice(cancel_context, "cancel_trade", {"offer_id": root_id}),
        )
    )
    assert not cancelled.state.trade_window.offers[root_id].active
    response: Any = _context(
        engine, actor=Color.BLUE, actions=trade_response_actions(engine.state, Color.BLUE)
    )
    for tool in ("accept_offer", "reject_offer"):
        branch: Any = deepcopy(engine)
        choice: Any = parse_tool_choice(response, tool, {"offer_id": root_id})
        branch.step(action_from_choice(response, choice))
        root: Any = branch.state.trade_window.offers[root_id]
        assert Color.BLUE in (root.willing_by if tool == "accept_offer" else root.declined_by)
    counter = parse_tool_choice(
        response, "counter_offer", {"offer_id": root_id, "give": {"ORE": 2}, "receive": {"WOOD": 1}}
    )
    engine.step(action_from_choice(response, counter))
    candidate = engine.state.trade_window.executable_candidates()[0]
    assert candidate.counterparty == Color.BLUE
    confirm_context = _context(engine)
    confirm = parse_tool_choice(
        confirm_context, "confirm_trade", {"offer_id": candidate.offer_id, "counterparty": "BLUE"}
    )
    engine.step(action_from_choice(confirm_context, confirm))
    assert engine.state.player_state["P0_WOOD_IN_HAND"] == 3
    assert engine.state.player_state["P0_ORE_IN_HAND"] == 6
    assert engine.state.player_state["P1_WOOD_IN_HAND"] == 3
    assert engine.state.player_state["P1_ORE_IN_HAND"] == 0


def test_wildcard_offer_acceptance_does_not_invent_an_executable_trade() -> None:
    engine: Any = _turn_engine()
    context = _context(engine)
    proposal = parse_tool_choice(
        context, "offer_trade", {"give": {"WOOD": 1}, "receive": {}, "receive_any": 1}
    )
    engine.step(action_from_choice(context, proposal))
    offer_id = next(iter(engine.state.trade_window.offers))
    response = _context(
        engine, actor=Color.BLUE, actions=trade_response_actions(engine.state, Color.BLUE)
    )
    choice = parse_tool_choice(response, "accept_offer", {"offer_id": offer_id})
    engine.step(action_from_choice(response, choice))
    assert engine.state.trade_window.executable_candidates() == ()
    context = _context(engine)
    assert "confirm_trade(" not in render_action_tools(context)
    with pytest.raises(ValueError, match="unavailable"):
        parse_tool_choice(context, "confirm_trade", {"offer_id": offer_id, "counterparty": "BLUE"})


@pytest.mark.parametrize("tool", ["roll_dice", "discard", "move_robber", "steal_from"])
def test_reduced_interrupt_and_roll_menus_execute_strictly(tool: str) -> None:
    engine = _turn_engine()
    state = engine.state
    arguments: Any = {}
    discard_count = 0
    if tool == "roll_dice":
        state.player_state["P0_HAS_ROLLED"] = False
    elif tool == "discard":
        state.current_prompt = ActionPrompt.DISCARD
        state.is_discarding = True
        discard_count = 10
        arguments = {"cards": {"WOOD": 4, "BRICK": 4, "ORE": 2}}
    elif tool == "move_robber":
        state.current_prompt = ActionPrompt.MOVE_ROBBER
        state.is_moving_knight = True
        tile: Any = next(
            tile
            for coord, tile in state.board.map.land_tiles.items()
            if coord != state.board.robber_coordinate
        )
        arguments = {"tile": tile_token(tile.id)}
    else:
        state.current_prompt = ActionPrompt.STEAL
        tile: Any = state.board.map.land_tiles[state.board.robber_coordinate]
        node = next(iter(tile.nodes.values()))
        state.board.buildings[node] = (Color.BLUE, "SETTLEMENT")
        arguments = {"player": "BLUE"}
    state.playable_actions = generate_playable_actions(state)
    context: Any = _context(engine, discard_count=discard_count)
    choice = parse_tool_choice(context, tool, arguments)
    action = action_from_choice(context, choice)
    assert engine.is_action_valid(action)
    engine.step(action)


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({}, frozenset(COLORS[1:])),
        ({"player": "BLUE"}, frozenset({Color.BLUE})),
        ({"player": "blue"}, frozenset({Color.BLUE})),
        ({"audience": ["BLUE", "WHITE"]}, frozenset({Color.BLUE, Color.WHITE})),
    ],
)
def test_shared_offer_trade_targets_a_player_or_audience_on_the_parameterized_menu(
    extra: dict[str, Any], expected: frozenset[Color]
) -> None:
    context, arguments, _ = _case("offer_trade")
    choice: Any = parse_tool_choice(context, "offer_trade", {**arguments, **extra}, shared=True)
    assert choice.trade_offer.audience == expected
    assert choice.trade_offer.offered_by == context.actor


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"player": "BLUE", "audience": ["BLUE"]}, "not both"),
        ({"player": "RED"}, "other participants"),
        ({"player": 3}, "must be a string"),
        ({"audience": []}, "nonempty"),
        ({"player": "BLUE", "confirm_if_accepted_by": ["WHITE"]}, "audience"),
    ],
)
def test_shared_offer_trade_rejects_bad_targets(extra: dict[str, Any], match: str) -> None:
    context, arguments, _ = _case("offer_trade")
    with pytest.raises(ValueError, match=match):
        choice = parse_tool_choice(context, "offer_trade", {**arguments, **extra}, shared=True)
        action_from_choice(context, choice)


def test_shared_offer_trade_doc_describes_targeting() -> None:
    context, _, _ = _case("offer_trade")
    rendered = render_action_tools(context, shared=True)
    assert "player or audience optional" in rendered
    assert "Never pass a player" not in rendered
