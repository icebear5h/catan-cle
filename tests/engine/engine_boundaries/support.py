"""Shared helpers for isolated engine-boundary regressions using reduced, bank-balanced positions."""
import pickle
from typing import Any

from cle.game_engine.events import GameEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.board import STATIC_GRAPH
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import (
    build_road,
    build_settlement,
    maintain_longest_road,
)

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _message(engine: GameEngine, **changes: object) -> GameEvent:
    arguments: Any = {
        "speaker": Color.RED,
        "text": "Leave the robber elsewhere and I will offer ORE.",
        "audience": (Color.BLUE,),
        "causation_id": "talk:boundary",
        "commitment": ("Leave the robber elsewhere", "Offer ORE", 2),
    }
    arguments.update(changes)
    return engine.append_message(**arguments)


def _snapshot_road_position(
    engine: GameEngine,
    paths: tuple[tuple[Color, tuple[int, ...]], ...],
    settlements: tuple[tuple[Color, int], ...] = (),
) -> None:
    """Place bank-balanced free pieces in a reduced position, not a replay."""
    state: Any = engine.state
    for color, node in settlements:
        state.board.build_settlement(color, node, initial_build_phase=True)
        build_settlement(state, color, node, is_free=True)
    for color, path in paths:
        for left, right in zip(path, path[1:]):
            assert STATIC_GRAPH.has_edge(left, right)
            assert (left, right) not in state.board.roads
            state.board.roads[left, right] = state.board.roads[right, left] = color
            build_road(state, color, tuple(sorted((left, right))), is_free=True)
    maintain_longest_road(state, *state.board.recompute_road_state())
    state.playable_actions = generate_playable_actions(state)


def _assert_road_restore_preserves_material(
    restored: GameEngine, saved: GameEngine, expected_player_state: dict[str, Any]
) -> None:
    state = restored.state
    assert state.player_state == expected_player_state
    assert {
        key: value for key, value in vars(state).items()
        if key not in {"board", "rng", "player_state", "playable_actions"}
    } == {
        key: value for key, value in vars(saved.state).items()
        if key not in {"board", "rng", "player_state", "playable_actions"}
    }
    for name in ("buildings", "roads", "board_buildable_ids", "robber_coordinate"):
        assert getattr(state.board, name) == getattr(saved.state.board, name)
    assert pickle.dumps(state.board.map) == pickle.dumps(saved.state.board.map)
    assert set(state.board.buildable_subgraph.edges) == set(saved.state.board.buildable_subgraph.edges)
    assert restored.rng is state.rng
    assert restored.rng is not saved.state.rng
    assert restored.rng.getstate() == saved.state.rng.getstate()
    assert (restored.id, restored.seed, restored.vps_to_win) == (
        saved.engine_id, saved.seed, saved.vps_to_win,
    )
    assert tuple(restored.events) == saved.events
    assert tuple(restored.commitments) == saved.commitments
    assert restored.capture_history == saved.capture_history
    assert restored.communication_limits == saved.communication_limits
    assert pickle.dumps(tuple(restored.history)) == pickle.dumps(saved.history)
