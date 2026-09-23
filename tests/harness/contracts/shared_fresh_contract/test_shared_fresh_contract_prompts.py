"""Responder prompts, observation slots, and resource blocks."""
import pickle
from dataclasses import fields
from typing import Any

import pytest

from cle.game_engine.board_tokens import node_token
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import SETTLEMENT, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.observation import PlayerObservation
from cle.harness.action_tools import (
    parse_tool_choice,
    render_shared_legal_actions,
)
from cle.harness.context import ContextAssembler
from cle.harness.models import PlayerSession
from cle.harness.shared_suite import load_shared_prompt_suite
from cle.sandbox.decision import build_decision_context

from .support import COLORS, context, main_engine, offer, values


def test_historical_observation_slots_load_with_additive_defaults() -> None:
    observation = main_engine().observe(Color.RED)
    additions = {"my_actual_vp", "current_prompt", "setup_road_anchor", "free_roads_available"}
    historical_slots = {field.name: getattr(observation, field.name) for field in fields(observation) if field.name not in additions}
    restored = PlayerObservation.__new__(PlayerObservation)
    restored.__setstate__((None, historical_slots))
    assert restored.my_actual_vp is None and restored.free_roads_available == 0
    roundtrip = pickle.loads(pickle.dumps(restored))
    assert roundtrip.my_resources == restored.my_resources
    assert roundtrip.my_actual_vp is None and roundtrip.current_prompt == ""


def test_trade_responder_sees_only_trade_tools_and_no_end_turn() -> None:
    engine = main_engine()
    offer(engine)
    ctx = context(engine, Color.BLUE)
    rendered = render_shared_legal_actions(ctx)
    legal_line = next(
        line for line in rendered.splitlines() if line.startswith("YOUR CURRENTLY LEGAL TOOLS")
    )
    assert "end_turn" not in legal_line
    assert "reject_offer" in legal_line
    assert "responding to RED's open trade offer" in rendered
    with pytest.raises(ValueError, match="end_turn is unavailable"):
        parse_tool_choice(ctx, "end_turn", {}, shared=True)


def test_spatial_parser_rejects_bare_node_token_without_brackets() -> None:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    ctx = build_decision_context(engine)
    assert ctx.legal_actions and ctx.legal_actions[0].action_type == ActionType.BUILD_SETTLEMENT
    literal = node_token(ctx.legal_actions[0].value)
    assert literal.startswith("<") and literal.endswith(">")
    with pytest.raises(ValueError, match="Invalid node token"):
        parse_tool_choice(ctx, "build_settlement", {"node": literal[1:-1]}, shared=True)
    with pytest.raises(ValueError, match="Invalid node token"):
        parse_tool_choice(ctx, "build_settlement", {"node": "N999"}, shared=True)


def test_batch_mode_responder_prompt_names_unavailable_end_turn() -> None:
    engine = main_engine()
    offer(engine)
    ctx = context(engine, Color.BLUE)
    assert load_shared_prompt_suite().deterministic_batches is True
    request = ContextAssembler(load_shared_prompt_suite().decision_suite()).assemble(
        ctx, PlayerSession(Color.BLUE, "responder-note"),
    )
    text = request.messages[-1].content
    assert "YOUR CURRENTLY LEGAL TOOLS" not in text
    assert "responding to RED's open trade offer" in text
    assert "end_turn is not available to you" in text


def test_resources_block_states_bank_rates_ports_and_discard_exposure() -> None:
    engine: Any = main_engine()
    state = engine.state
    for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"):
        state.player_state[f"P0_{resource}_IN_HAND"] = 0
    state.player_state["P0_WOOD_IN_HAND"] = 5
    state.player_state["P0_ORE_IN_HAND"] = 4
    state.playable_actions = generate_playable_actions(state)
    resources: Any = values(engine)["resources"]

    # 9 cards: the exposure line names the exact cost of a 7.
    assert "Total: 9 cards" in resources
    assert "DISCARD EXPOSURE: 9 cards held; any 7 rolled costs you 4 cards" in resources
    # No ports yet: every resource trades 4:1, and the line says the bank needs no partner.
    rates_line = next(line for line in resources.splitlines() if "BANK TRADE" in line)
    assert "always available, no partner; call the maritime_trade tool" in rates_line
    assert all(f"{name} 4" in rates_line for name in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"))
    assert "port" not in rates_line

    # A settlement on a 3:1 port drops every rate to 3; a 2:1 port drops its own resource to 2.
    board = state.board
    generic_port_node = next(iter(board.map.port_nodes[None]))
    two_to_one: Any = next((res, nodes) for res, nodes in board.map.port_nodes.items() if res is not None)
    # The observation lists settlements from the state registry, so place them
    # on both the board and the registry, as the engine's build path does.
    for node in (generic_port_node, next(iter(two_to_one[1]))):
        board.build_settlement(Color.RED, node, initial_build_phase=True)
        state.buildings_by_color[Color.RED][SETTLEMENT].append(node)
    state.playable_actions = generate_playable_actions(state)
    rates_line = next(line for line in values(engine)["resources"].splitlines() if "BANK TRADE" in line)
    port_name = two_to_one[0].name if hasattr(two_to_one[0], "name") else str(two_to_one[0])
    assert f"{port_name} 2 (2:1 port)" in rates_line
    others = [n for n in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE") if n != port_name]
    assert all(f"{name} 3 (3:1 port)" in rates_line for name in others)

    # At or under the limit, no exposure line.
    state.player_state["P0_WOOD_IN_HAND"] = 3
    state.playable_actions = generate_playable_actions(state)
    assert "DISCARD EXPOSURE" not in values(engine)["resources"]
