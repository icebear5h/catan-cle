"""Fresh request construction and authoritative inventory."""
import json
import pickle
from types import SimpleNamespace
from typing import Any

import pytest

from cle.game_engine.board_tokens import edge_token, node_token
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.harness.board_surface import openai_messages_with_board
from cle.harness.models import ModelMessage
from cle.players.agent import AgentPlayer
from cle.players.validation import action_from_choice
from cle.sandbox.decision import build_decision_context

from .support import COLORS, RecordedReply, main_engine, values


@pytest.mark.asyncio
async def test_real_setup_repeated_requests_are_fresh_and_exact() -> None:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    transports: Any = {color: RecordedReply() for color in COLORS}
    players: Any = {color: AgentPlayer(color, transports[color], session_id=f"setup:{color.value}") for color in COLORS}
    definitions: Any = None
    seen: Any = {color: [] for color in COLORS}
    for step in range(16):
        ctx: Any = build_decision_context(engine)
        actor: Any = ctx.actor
        player: Any = players[actor]
        player.session.messages.append(ModelMessage("assistant", "OLD TRANSCRIPT"))
        action: Any = ctx.legal_actions[0]
        road: Any = action.action_type == ActionType.BUILD_ROAD
        arguments: Any = {"edge": edge_token(action.value)} if road else {"node": node_token(action.value)}
        transports[actor].content = json.dumps({
            "tool": "build_road" if road else "build_settlement", "arguments": arguments,
            "notes": f"private {actor.value} plan {step}",
        })
        attempt: Any = await player.choose(ctx)
        assert attempt.validation_error is None
        request: Any = attempt.model_request
        assert len(request.messages) == 2
        transmitted: Any = openai_messages_with_board(request.messages, request.board_presentation, allow_image_input=False)
        text: Any = "\n".join(message["content"] for message in transmitted)
        assert "OLD TRANSCRIPT" not in text and "PRIOR PRIVATE REASONING" not in text
        assert "DECISION FACTS" not in text
        assert "No active trade window is present in the supplied observation." in text
        assert "EXACT LEGAL" not in text and "VALID ACTIONS" not in text
        tools: Any = next(c.value for c in request.components if c.id == "environment.legal_actions")
        stable, _, _ = tools.partition("YOUR CURRENTLY LEGAL TOOLS")
        definitions = stable if definitions is None else definitions
        assert stable == definitions
        ordinal: Any = "first" if len(seen[actor]) < 2 else "second"
        assert f"must place the {ordinal} {'road' if road else 'settlement'}" in text
        if road:
            anchor: Any = ctx.observation.my_settlements[-1]
            assert f"already placed at {node_token(anchor)}" in text
            assert anchor in action.value
        assert "starting hand comes only from the second settlement" in text
        event_component: Any = next(c.value for c in request.components if c.id == "environment.visible_events")
        if seen[actor]:
            prior_sequence: Any = seen[actor][-1]
            assert all(int(line.split(".", 1)[0]) >= prior_sequence for line in event_component.splitlines())
            assert player.session.strategic_memory in text
        transition: Any = engine.step(action_from_choice(ctx, attempt.choice))
        if not road:
            actual: Any = engine.observe(actor).my_resources
            expected: Any = dict.fromkeys(actual, 0)
            if ordinal == "second":
                for tile in engine.state.board.map.adjacent_tiles[action.value]:
                    if tile.resource is not None:
                        expected[tile.resource] += 1
            assert actual == expected
        player.accept(attempt, SimpleNamespace(context=ctx, after_revision=transition.after_revision))
        seen[actor].append(step)
    assert not engine.state.is_initial_build_phase


def test_inventory_actual_vp_and_free_roads_are_authoritative() -> None:
    engine: Any = main_engine()
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
    rendered: Any = values(engine)
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


@pytest.mark.asyncio
async def test_invalid_shared_call_preserves_state_notes_and_feedback_then_retries() -> None:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    transport = RecordedReply()
    player = AgentPlayer(Color.RED, transport, session_id="invalid-call")
    player.session.strategic_memory = "keep accepted notes"
    ctx = build_decision_context(engine)
    before_engine, before_session = pickle.dumps(engine.snapshot()), player.session.snapshot()
    transport.content = json.dumps({"tool": "end_turn", "arguments": {}, "notes": "must not commit"})
    attempt: Any = await player.choose(ctx)
    assert attempt.choice is None and "end_turn is unavailable" in attempt.validation_error
    assert pickle.dumps(engine.snapshot()) == before_engine
    assert player.session.snapshot() == before_session
    transport.content = json.dumps({"tool": "build_settlement", "arguments": {"node": node_token(ctx.legal_actions[0].value)}})
    retry: Any = await player.choose(ctx, attempt.validation_error)
    assert retry.validation_error is None and retry.choice.notes_update is None
    text = retry.model_request.messages[-1].content
    assert "CORRECTION FROM THE SANDBOX" in text and attempt.validation_error in text
    assert "VALID ACTIONS" not in text and "playable_actions=" not in text
    transition = engine.step(action_from_choice(ctx, retry.choice))
    player.accept(retry, SimpleNamespace(context=ctx, after_revision=transition.after_revision))
    assert player.session.strategic_memory == "keep accepted notes"
