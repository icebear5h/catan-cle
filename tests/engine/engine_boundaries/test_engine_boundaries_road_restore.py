"""Longest-road restore and snapshot-material boundaries."""
import json
import pickle
from copy import deepcopy
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.json import GameEncoder
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import (
    VICTORY_POINT,
    Action,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import (
    player_key,
)

from .support import (
    COLORS,
    _assert_road_restore_preserves_material,
    _message,
    _snapshot_road_position,
)


@pytest.mark.parametrize("stale_part", ["cache", "menu", "both", "inactive-cache"])
def test_restore_removes_legacy_enemy_crossing_road_candidates(engine: GameEngine, stale_part: str) -> None:
    _snapshot_road_position(
        engine, ((Color.RED, (37, 14, 15)),), ((Color.RED, 37), (Color.BLUE, 15)),
    )
    if stale_part == "inactive-cache":
        engine.step(Action(Color.RED, ActionType.END_TURN, None))
    state = engine.state
    blocked = Action(Color.RED, ActionType.BUILD_ROAD, (4, 15))
    expected_actions = deepcopy(state.playable_actions)
    assert blocked not in expected_actions
    assert state.board.road_lengths[Color.RED] == 2
    if stale_part in {"cache", "both", "inactive-cache"}:
        state.board.buildable_edges_cache[Color.RED].append(blocked.value)
    if stale_part in {"menu", "both"}:
        state.playable_actions.append(blocked)
    saved = pickle.loads(pickle.dumps(engine.snapshot()))
    saved_bytes = pickle.dumps(saved)
    source_bytes = pickle.dumps(engine)
    restored = GameEngine(COLORS, initialize=False)

    restored.restore(saved)

    assert blocked.value not in restored.state.board.buildable_edges(Color.RED)
    assert restored.state.playable_actions == expected_actions
    assert not restored.is_action_valid(blocked)
    _assert_road_restore_preserves_material(restored, saved, saved.state.player_state)
    assert pickle.dumps(saved) == saved_bytes
    assert pickle.dumps(engine) == source_bytes
    before_rejection = pickle.dumps(restored)
    with pytest.raises(ValueError, match="not playable"):
        restored.step(blocked)
    assert pickle.dumps(restored) == before_rejection


@pytest.mark.parametrize("stale_board", [False, True], ids=["counters-only", "board-and-counters"])
def test_restore_revokes_legacy_below_five_award_without_rewriting_history(
    engine: GameEngine, offer_action: Action, stale_board: bool,
) -> None:
    _snapshot_road_position(
        engine,
        ((Color.RED, (29, 30, 31, 32, 33, 34)), (Color.BLUE, (12, 11, 32))),
        ((Color.RED, 29), (Color.BLUE, 12), (Color.BLUE, 32)),
    )
    state = engine.state
    state.development_listdeck.remove(VICTORY_POINT)
    state.player_state["P0_VICTORY_POINT_IN_HAND"] += 1
    state.player_state["P0_ACTUAL_VICTORY_POINTS"] += 1
    expected_player_state: Any = state.player_state.copy()
    assert state.board.road_lengths[Color.RED] == 3
    assert state.board.road_color is None
    if stale_board:
        state.board.connected_components[Color.RED] = [{29, 30, 31, 32, 33, 34}]
        state.board.road_lengths[Color.RED] = 5
        state.board.road_length = 5
        state.board.road_color = Color.RED
    state.player_state["P0_LONGEST_ROAD_LENGTH"] = 5
    state.player_state["P0_HAS_ROAD"] = True
    state.player_state["P0_VICTORY_POINTS"] += 2
    state.player_state["P0_ACTUAL_VICTORY_POINTS"] += 2
    _message(engine)
    engine.step(offer_action)
    saved = pickle.loads(pickle.dumps(engine.snapshot()))
    saved_bytes = pickle.dumps(saved)
    restored = GameEngine(COLORS, initialize=False)

    restored.restore(saved)

    assert restored.state.board.road_color is None
    assert restored.state.board.road_length == 3
    assert restored.state.board.road_lengths[Color.RED] == 3
    assert restored.state.board.connected_components[Color.RED] == [
        {29, 30, 31, 32}, {32, 33, 34},
    ]
    assert restored.state.player_state["P0_VICTORY_POINTS"] == 1
    assert restored.state.player_state["P0_ACTUAL_VICTORY_POINTS"] == 2
    assert restored.state.playable_actions == generate_playable_actions(restored.state)
    _assert_road_restore_preserves_material(restored, saved, expected_player_state)
    assert restored.history[-1][0].player_state["P0_HAS_ROAD"] is True
    assert pickle.dumps(saved) == saved_bytes
    normalized = restored.snapshot()
    restored.restore(normalized)
    assert pickle.dumps(restored.snapshot()) == pickle.dumps(normalized)


@pytest.mark.parametrize("incumbent", [None, Color.RED, Color.BLUE])
def test_restore_uses_saved_board_incumbent_for_corrected_five_road_tie(engine: GameEngine, incumbent: Color | None) -> None:
    _snapshot_road_position(
        engine,
        ((Color.RED, (29, 30, 31, 32, 33, 34)), (Color.BLUE, (49, 50, 51, 52, 23, 6))),
    )
    state = engine.state
    state.board.road_color = incumbent
    state.board.road_lengths[Color.RED] = 6
    state.board.road_length = 6
    state.player_state["P0_LONGEST_ROAD_LENGTH"] = 6
    # Award flags are derived counters, not evidence of a different historical incumbent.
    wrong_flag = Color.RED if incumbent != Color.RED else Color.BLUE
    key = player_key(state, wrong_flag)
    state.player_state[f"{key}_HAS_ROAD"] = True
    state.player_state[f"{key}_VICTORY_POINTS"] += 2
    state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 2
    saved = engine.snapshot()
    restored = GameEngine(COLORS, initialize=False)

    restored.restore(saved)

    assert restored.state.board.road_color == incumbent
    assert restored.state.board.road_length == 5
    for color in COLORS:
        key = player_key(state, color)
        assert restored.state.player_state[f"{key}_LONGEST_ROAD_LENGTH"] == (
            5 if color in {Color.RED, Color.BLUE} else 0
        )
        assert restored.state.player_state[f"{key}_HAS_ROAD"] == (color == incumbent)
        assert restored.state.player_state[f"{key}_VICTORY_POINTS"] == 2 * (color == incumbent)
        assert restored.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] == (
            2 * (color == incumbent)
        )


@pytest.mark.parametrize("opening_steps", [0, 1, 16])
def test_current_snapshot_preserves_material_menu_order_and_forward_suffix(opening_steps: int) -> None:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False, capture_history=True)
    for _ in range(opening_steps):
        engine.step(engine.state.playable_actions[0])
    _message(engine)
    # Equivalent cache order and absent zero-length entries must not trigger normalization.
    for color in COLORS:
        engine.state.board.buildable_edges(color).reverse()
    engine.state.playable_actions = generate_playable_actions(engine.state)
    saved = pickle.loads(pickle.dumps(engine.snapshot()))
    saved_bytes = pickle.dumps(saved)
    restored = GameEngine(COLORS, initialize=False)

    restored.restore(saved)

    assert pickle.dumps(restored.snapshot()) == saved_bytes
    assert restored.state.playable_actions == engine.state.playable_actions
    _assert_road_restore_preserves_material(restored, saved, saved.state.player_state)
    for _ in range(20):
        assert restored.state.playable_actions == engine.state.playable_actions
        action = engine.state.playable_actions[0]
        assert restored.step(action) == engine.step(action)
        assert json.loads(json.dumps(restored, cls=GameEncoder)) == json.loads(
            json.dumps(engine, cls=GameEncoder)
        )
        assert restored.state.playable_actions == engine.state.playable_actions
        assert restored.state.player_state == engine.state.player_state
        assert restored.state.development_listdeck == engine.state.development_listdeck
        assert restored.state.resource_freqdeck == engine.state.resource_freqdeck
        assert restored.rng.getstate() == engine.rng.getstate()
        assert restored.events == engine.events
        assert restored.commitments == engine.commitments
    assert pickle.dumps(saved) == saved_bytes


@pytest.mark.parametrize("difference", ["reordered", "missing-menu", "duplicate-menu", "missing-cache"])
def test_restore_stale_counter_preserves_only_equivalent_menu_and_cache_order(
    engine: GameEngine, monkeypatch: pytest.MonkeyPatch, difference: str,
) -> None:
    _snapshot_road_position(
        engine, ((Color.RED, (37, 14, 15)),), ((Color.RED, 37), (Color.BLUE, 15)),
    )
    state = engine.state
    for color in COLORS:
        state.board.buildable_edges(color).reverse()
    state.playable_actions = list(reversed(generate_playable_actions(state)))
    assert state.playable_actions != generate_playable_actions(state)
    expected: Any = engine.copy()
    expected_actions = deepcopy(state.playable_actions)
    state.player_state["P0_LONGEST_ROAD_LENGTH"] = 5
    road_index = next(
        index for index, action in enumerate(state.playable_actions)
        if action.action_type == ActionType.BUILD_ROAD
    )
    if difference == "missing-menu":
        state.playable_actions.pop(road_index)
    elif difference == "duplicate-menu":
        state.playable_actions[road_index] = state.playable_actions[0]
    elif difference == "missing-cache":
        state.board.buildable_edges_cache[Color.RED].pop()
    if difference in {"missing-menu", "duplicate-menu"}:
        expected_actions = generate_playable_actions(expected.state)
    saved: Any = pickle.loads(pickle.dumps(engine.snapshot()))
    saved_bytes = pickle.dumps(saved)
    restored: Any = GameEngine(COLORS, initialize=False)
    # Menu identity comparison must work even when Actions cannot be hashed.
    monkeypatch.setattr(Action, "__hash__", None)

    restored.restore(saved)

    assert restored.state.player_state["P0_LONGEST_ROAD_LENGTH"] == 2
    assert restored.state.playable_actions == expected_actions
    for color, edges in expected.state.board.buildable_edges_cache.items():
        restored_edges = restored.state.board.buildable_edges_cache[color]
        if difference == "missing-cache" and color == Color.RED:
            assert set(restored_edges) == set(edges)
            assert len(restored_edges) == len(edges)
        else:
            assert restored_edges == edges
    _assert_road_restore_preserves_material(restored, saved, expected.state.player_state)
    assert pickle.dumps(saved) == saved_bytes

    if difference == "reordered":
        road = next(
            action for action in expected_actions if action.action_type == ActionType.BUILD_ROAD
        )
        assert restored.step(road) == expected.step(road)
        for _ in range(20):
            assert restored.state.playable_actions == expected.state.playable_actions
            action = expected.state.playable_actions[0]
            assert restored.step(action) == expected.step(action)
            assert json.loads(json.dumps(restored, cls=GameEncoder)) == json.loads(
                json.dumps(expected, cls=GameEncoder)
            )
            assert restored.rng.getstate() == expected.rng.getstate()
            assert restored.state.development_listdeck == expected.state.development_listdeck
            assert restored.events == expected.events
            assert restored.commitments == expected.commitments
