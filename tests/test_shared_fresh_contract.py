"""Shared model boundary exercised against engine states, not prompt-derived menus."""

import json
import pickle
from copy import deepcopy
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest

from cle.game_engine.board_tokens import edge_token, node_token
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions, trade_response_actions
from cle.game_engine.models.enums import SETTLEMENT, Action, ActionType
from cle.game_engine.observation import PlayerObservation
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import player_key
from cle.game_engine.trading import TradeCandidate, TradeOffer
from cle.harness.action_tools import parse_tool_choice, render_action_tools, render_shared_legal_actions
from cle.harness.board_surface import openai_messages_with_board
from cle.harness.components import observation_component_values
from cle.harness.context import ContextAssembler, PlayerResponseParseError, PlayerResponseParser
from cle.harness.models import ModelMessage, ModelResponse, PlayerSession
from cle.harness.shared_suite import load_shared_prompt_suite
from cle.players.agent import AgentPlayer
from cle.players.validation import action_from_choice
from cle.replay.runtime.step_executor import _apply_trade_closures, _publish_replay_action, _publish_trade_overlay
from cle.sandbox.decision import build_decision_context


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
WOOD = (1, 0, 0, 0, 0)
ORE = (0, 0, 0, 0, 1)


class RecordedReply:
    content = ""

    async def complete(self, request):
        return ModelResponse(self.content, native_reasoning="PRIOR PRIVATE REASONING")


def values(engine, color=Color.RED):
    return observation_component_values(engine.observe(color), include_initial_placement_order=True)


def main_engine():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    while engine.state.is_initial_build_phase:
        engine.step(engine.state.playable_actions[0])
    engine.state.player_state["P0_HAS_ROLLED"] = True
    for color in COLORS:
        key = player_key(engine.state, color)
        # Explicit reduced-state fixture with conserved public bank supply.
        for resource in ("WOOD", "ORE"):
            index = 0 if resource == "WOOD" else 4
            current = engine.state.player_state[f"{key}_{resource}_IN_HAND"]
            engine.state.resource_freqdeck[index] -= 3 - current
            engine.state.player_state[f"{key}_{resource}_IN_HAND"] = 3
    engine.state.playable_actions = generate_playable_actions(engine.state)
    return engine


def offer(engine, actor=Color.RED, parent=None, give=WOOD, receive=ORE, **kwargs):
    proposal = TradeOffer(
        actor, frozenset({Color.RED} if parent else COLORS[1:]), give, receive,
        parent_offer_id=parent, **kwargs,
    )
    transition = engine.step(Action(actor, ActionType.COUNTER_OFFER if parent else ActionType.OFFER_TRADE, proposal))
    return transition.resolved_action.value


def context(engine, actor=Color.RED):
    actions = tuple(trade_response_actions(engine.state, actor)) if actor != Color.RED else None
    if actions == ():
        return replace(build_decision_context(engine), actor=actor, observation=engine.observe(actor), legal_actions=())
    return build_decision_context(
        engine, actor, actions,
    )


def semantic(engine, tool, actor=Color.RED, **arguments):
    ctx = context(engine, actor)
    choice = parse_tool_choice(ctx, tool, arguments, shared=True)
    return action_from_choice(ctx, choice)


@pytest.mark.asyncio
async def test_real_setup_repeated_requests_are_fresh_and_exact():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    transports = {color: RecordedReply() for color in COLORS}
    players = {color: AgentPlayer(color, transports[color], session_id=f"setup:{color.value}") for color in COLORS}
    definitions = None
    seen = {color: [] for color in COLORS}
    for step in range(16):
        ctx = build_decision_context(engine)
        actor = ctx.actor
        player = players[actor]
        player.session.messages.append(ModelMessage("assistant", "OLD TRANSCRIPT"))
        action = ctx.legal_actions[0]
        road = action.action_type == ActionType.BUILD_ROAD
        arguments = {"edge": edge_token(action.value)} if road else {"node": node_token(action.value)}
        transports[actor].content = json.dumps({
            "tool": "build_road" if road else "build_settlement", "arguments": arguments,
            "notes": f"private {actor.value} plan {step}",
        })
        attempt = await player.choose(ctx)
        assert attempt.validation_error is None
        request = attempt.model_request
        assert len(request.messages) == 2
        transmitted = openai_messages_with_board(request.messages, request.board_presentation, allow_image_input=False)
        text = "\n".join(message["content"] for message in transmitted)
        assert "OLD TRANSCRIPT" not in text and "PRIOR PRIVATE REASONING" not in text
        assert "DECISION FACTS" not in text
        assert "No active trade window is present in the supplied observation." in text
        assert "EXACT LEGAL" not in text and "VALID ACTIONS" not in text
        tools = next(c.value for c in request.components if c.id == "environment.legal_actions")
        stable, _, _ = tools.partition("YOUR CURRENTLY LEGAL TOOLS")
        definitions = stable if definitions is None else definitions
        assert stable == definitions
        ordinal = "first" if len(seen[actor]) < 2 else "second"
        assert f"must place the {ordinal} {'road' if road else 'settlement'}" in text
        if road:
            anchor = ctx.observation.my_settlements[-1]
            assert f"already placed at {node_token(anchor)}" in text
            assert anchor in action.value
        assert "starting hand comes only from the second settlement" in text
        event_component = next(c.value for c in request.components if c.id == "environment.visible_events")
        if seen[actor]:
            prior_sequence = seen[actor][-1]
            assert all(int(line.split(".", 1)[0]) >= prior_sequence for line in event_component.splitlines())
            assert player.session.strategic_memory in text
        transition = engine.step(action_from_choice(ctx, attempt.choice))
        if not road:
            actual = engine.observe(actor).my_resources
            expected = dict.fromkeys(actual, 0)
            if ordinal == "second":
                for tile in engine.state.board.map.adjacent_tiles[action.value]:
                    if tile.resource is not None:
                        expected[tile.resource] += 1
            assert actual == expected
        player.accept(attempt, SimpleNamespace(context=ctx, after_revision=transition.after_revision))
        seen[actor].append(step)
    assert not engine.state.is_initial_build_phase


def test_inventory_actual_vp_and_free_roads_are_authoritative():
    engine = main_engine()
    state = engine.state
    for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"):
        state.player_state[f"P0_{resource}_IN_HAND"] = 0
    state.player_state["P0_KNIGHT_IN_HAND"] = 1
    state.player_state["P0_KNIGHT_OWNED_AT_START"] = True
    state.player_state["P0_VICTORY_POINT_IN_HAND"] = 2
    state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 4
    state.player_state["P1_VICTORY_POINT_IN_HAND"] = 3
    state.player_state["P1_ACTUAL_VICTORY_POINTS"] = 5
    state.playable_actions = generate_playable_actions(state)
    rendered = values(engine)
    assert "Total: 0 cards" in rendered["resources"]
    assert "KNIGHT: 1 (playable now)" in rendered["resources"]
    assert "VICTORY_POINT: 2 (passive VP" in rendered["resources"]
    assert "Your actual VP: 4/10 (public: 2)" in rendered["phase_info"]
    assert "BLUE: 2 public VP" in rendered["opponents"]
    assert "BLUE: 5" not in rendered["opponents"]
    state.player_state["P0_KNIGHT_OWNED_AT_START"] = False
    state.playable_actions = generate_playable_actions(state)
    assert "KNIGHT: 1 (not playable now)" in values(engine)["resources"]
    state.player_state["P0_KNIGHT_OWNED_AT_START"] = True
    state.player_state["P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = True
    state.playable_actions = generate_playable_actions(state)
    assert "KNIGHT: 1 (not playable now)" in values(engine)["resources"]
    state.player_state["P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = False
    state.player_state["P0_ROAD_BUILDING_IN_HAND"] = 1
    state.player_state["P0_ROAD_BUILDING_OWNED_AT_START"] = True
    state.playable_actions = generate_playable_actions(state)
    engine.step(Action(Color.RED, ActionType.PLAY_ROAD_BUILDING, None))
    assert "2 free road placements remaining" in values(engine)["phase_info"]
    engine.step(state.playable_actions[0])
    assert "1 free road placements remaining" in values(engine)["phase_info"]
    assert "Setup:" not in values(engine)["phase_info"]


def test_semantic_trade_lifecycle_and_counter_direction():
    engine = main_engine()
    root = offer(engine)
    terms = {"player": "RED", "give": {"ORE": 1}, "receive": {"WOOD": 1}}
    for tool, action_type, actor in (("reject_offer", ActionType.REJECT_TRADE, Color.WHITE), ("accept_offer", ActionType.ACCEPT_TRADE, Color.BLUE)):
        action = semantic(engine, tool, actor, **terms)
        assert action == Action(actor, action_type, root.id)
        event = engine.step(action).events[0]
        assert event.public_payload["offer"]["give"] == {"WOOD": 1}
        assert event.public_payload["offer"]["receive"] == {"ORE": 1}
    engine.step(semantic(engine, "accept_offer", Color.ORANGE, **terms))
    confirm = semantic(engine, "confirm_trade", player="BLUE", give={"WOOD": 1}, receive={"ORE": 1})
    assert confirm.value == TradeCandidate(root.id, Color.RED, Color.BLUE)
    event = engine.step(confirm).events[0]
    assert event.public_payload["offer"]["id"] == root.id
    assert engine.state.trade_window.status.value == "closed"

    engine = main_engine()
    root = offer(engine)
    counter = semantic(engine, "counter_offer", Color.BLUE, player="RED",
                       original={"give": {"ORE": 1}, "receive": {"WOOD": 1}},
                       proposed={"give": {"ORE": 2}, "receive": {"WOOD": 1}})
    counter_event = engine.step(counter).events[0]
    assert counter_event.public_payload["original"]["give"] == {"WOOD": 1}
    confirm = semantic(engine, "confirm_trade", player="BLUE", give={"WOOD": 1}, receive={"ORE": 2})
    assert confirm.value.offer_id != root.id
    engine.step(confirm)


def test_semantic_matching_rejects_stale_wrong_direction_and_ambiguity():
    engine = main_engine()
    root = offer(engine)
    with pytest.raises(ValueError, match="No active visible"):
        semantic(engine, "accept_offer", Color.BLUE, player="RED", give={"WOOD": 1}, receive={"ORE": 1})
    # Equivalent terms from distinct offers must not be guessed, even if only one
    # corresponding menu action is currently executable.
    duplicate = deepcopy(root)
    duplicate.id = "another-active-offer"
    engine.state.trade_window.offers[duplicate.id] = duplicate
    with pytest.raises(ValueError, match="Ambiguous trade"):
        semantic(engine, "accept_offer", Color.BLUE, player="RED", give={"ORE": 1}, receive={"WOOD": 1})
    duplicate.status = type(duplicate.status).WITHDRAWN
    cancel = semantic(engine, "cancel_trade", player="BLUE", give={"WOOD": 1}, receive={"ORE": 1})
    event = engine.step(cancel).events[0]
    assert event.public_payload["offer"]["give"] == {"WOOD": 1}
    with pytest.raises(ValueError, match="stale"):
        semantic(engine, "accept_offer", Color.BLUE, player="RED", give={"ORE": 1}, receive={"WOOD": 1})


def test_wildcard_events_visible_without_legal_menu_or_hidden_hands():
    engine = main_engine()
    root = offer(engine, receive=(0, 0, 0, 0, 0), receive_any=1)
    event_text = ContextAssembler._format_events(tuple(engine.project_events(Color.BLUE))[-1:], shared=True)
    assert "RED gives 1 WOOD, receives 1 ANY" in event_text
    assert "not executable" in event_text
    assert root.id not in event_text
    ctx = context(engine, Color.BLUE)
    assert not engine.state.trade_window.executable_candidates()
    assert "accept_offer(" in render_action_tools(ctx, shared=True)
    action = semantic(engine, "reject_offer", Color.BLUE, player="RED", give={}, receive={"WOOD": 1}, give_any=1)
    engine.step(action)
    rendered = ContextAssembler._format_events(tuple(engine.project_events(Color.BLUE))[-1:], shared=True)
    assert "REJECT_TRADE RED gives 1 WOOD, receives 1 ANY" in rendered
    assert "IN_HAND" not in rendered


@pytest.mark.asyncio
async def test_invalid_shared_call_preserves_state_notes_and_feedback_then_retries():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    transport = RecordedReply()
    player = AgentPlayer(Color.RED, transport, session_id="invalid-call")
    player.session.strategic_memory = "keep accepted notes"
    ctx = build_decision_context(engine)
    before_engine, before_session = pickle.dumps(engine.snapshot()), player.session.snapshot()
    transport.content = json.dumps({"tool": "end_turn", "arguments": {}, "notes": "must not commit"})
    attempt = await player.choose(ctx)
    assert attempt.choice is None and "end_turn is unavailable" in attempt.validation_error
    assert pickle.dumps(engine.snapshot()) == before_engine
    assert player.session.snapshot() == before_session
    transport.content = json.dumps({"tool": "build_settlement", "arguments": {"node": node_token(ctx.legal_actions[0].value)}})
    retry = await player.choose(ctx, attempt.validation_error)
    assert retry.validation_error is None and retry.choice.notes_update is None
    text = retry.model_request.messages[-1].content
    assert "CORRECTION FROM THE SANDBOX" in text and attempt.validation_error in text
    assert "VALID ACTIONS" not in text and "playable_actions=" not in text
    transition = engine.step(action_from_choice(ctx, retry.choice))
    player.accept(retry, SimpleNamespace(context=ctx, after_revision=transition.after_revision))
    assert player.session.strategic_memory == "keep accepted notes"


def test_shared_parser_requires_terms_while_legacy_id_parser_remains_valid():
    engine = main_engine()
    root = offer(engine)
    ctx = context(engine, Color.BLUE)
    assert parse_tool_choice(ctx, "accept_offer", {"offer_id": root.id}).action_index >= 0
    parser = PlayerResponseParser(load_shared_prompt_suite().decision_suite())
    with pytest.raises(PlayerResponseParseError, match="Expected arguments"):
        parser.parse(ctx, ModelResponse(json.dumps({"tool": "accept_offer", "arguments": {"offer_id": root.id}})))


def test_historical_observation_slots_load_with_additive_defaults():
    observation = main_engine().observe(Color.RED)
    additions = {"my_actual_vp", "current_prompt", "setup_road_anchor", "free_roads_available"}
    historical_slots = {field.name: getattr(observation, field.name) for field in fields(observation) if field.name not in additions}
    restored = PlayerObservation.__new__(PlayerObservation)
    restored.__setstate__((None, historical_slots))
    assert restored.my_actual_vp is None and restored.free_roads_available == 0
    roundtrip = pickle.loads(pickle.dumps(restored))
    assert roundtrip.my_resources == restored.my_resources
    assert roundtrip.my_actual_vp is None and roundtrip.current_prompt == ""


def test_replay_source_only_proposals_keep_terms_in_responses_and_closure():
    engine = main_engine()
    source = TradeOffer(Color.RED, frozenset(COLORS[1:]), WOOD, (0, 0, 0, 0, 0), receive_any=1, id="source-only")
    _publish_replay_action(engine, Action(Color.RED, ActionType.OFFER_TRADE, source))
    assert engine.state.trade_window is None
    runtime = SimpleNamespace(
        current_game=engine, replay_index=0,
        replay_data={"colonist_color_to_engine_idx": {"1": 0, "2": 1}},
    )
    _publish_replay_action(engine, Action(Color.BLUE, ActionType.REJECT_TRADE, source.id))
    _publish_trade_overlay(runtime, {"type": "CLEAR_TRADE_RESPONSE", "player": 2, "trade_id": source.id})
    _apply_trade_closures({"type": "CLOSE_TRADE", "creator": 1, "trade_id": source.id, "reason": "withdrawn"}, runtime)
    events = tuple(engine.project_events(Color.BLUE))[-3:]
    for event in events:
        assert event.payload["offer"]["receive_any"] == 1
        rendered = ContextAssembler._format_events((event,), shared=True)
        assert "RED gives 1 WOOD, receives 1 ANY" in rendered
        assert source.id not in rendered
    assert ContextAssembler._format_events((events[0],)).endswith("REJECT_TRADE source-only")


def test_trade_responder_sees_only_trade_tools_and_no_end_turn():
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


def test_spatial_parser_rejects_bare_node_token_without_brackets():
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    ctx = build_decision_context(engine)
    assert ctx.legal_actions and ctx.legal_actions[0].action_type == ActionType.BUILD_SETTLEMENT
    literal = node_token(ctx.legal_actions[0].value)
    assert literal.startswith("<") and literal.endswith(">")
    with pytest.raises(ValueError, match="Invalid node token"):
        parse_tool_choice(ctx, "build_settlement", {"node": literal[1:-1]}, shared=True)
    with pytest.raises(ValueError, match="Invalid node token"):
        parse_tool_choice(ctx, "build_settlement", {"node": "N999"}, shared=True)


def test_batch_mode_responder_prompt_names_unavailable_end_turn():
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


def test_resources_block_states_bank_rates_ports_and_discard_exposure():
    engine = main_engine()
    state = engine.state
    for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"):
        state.player_state[f"P0_{resource}_IN_HAND"] = 0
    state.player_state["P0_WOOD_IN_HAND"] = 5
    state.player_state["P0_ORE_IN_HAND"] = 4
    state.playable_actions = generate_playable_actions(state)
    resources = values(engine)["resources"]

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
    two_to_one = next((res, nodes) for res, nodes in board.map.port_nodes.items() if res is not None)
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
