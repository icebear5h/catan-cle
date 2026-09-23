"""Branch, history, and observation detachment boundaries."""
import json
import pickle
from copy import deepcopy
from typing import Any

import pytest

from cle.game_engine.communication import CommitmentStatus
from cle.game_engine.game import GameEngine
from cle.game_engine.json import GameEncoder
from cle.game_engine.models.board import Board
from cle.game_engine.models.enums import (
    CITY,
    Action,
    ActionType,
)
from cle.game_engine.models.map import CatanMap
from cle.game_engine.models.player import Color
from cle.game_engine.observation import observe_state
from cle.game_engine.state import GameState

from .support import COLORS, _message


def test_observation_does_not_populate_empty_building_caches() -> None:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    before = pickle.dumps(engine)
    for color in COLORS:
        observation = engine.observe(color)
        assert observation.my_roads == []
    assert pickle.dumps(engine) == before


@pytest.mark.parametrize("boundary", ["snapshot", "copy", "restore"])
def test_branch_mutations_cannot_change_source_or_saved_snapshot(engine: GameEngine, offer_action: Action, boundary: str) -> None:
    _message(engine)
    engine.step(offer_action)
    engine.step(Action(Color.BLUE, ActionType.ACCEPT_TRADE, engine.state.actions[-1].value.id))
    saved: Any = engine.snapshot()
    original_bytes = pickle.dumps(engine)
    saved_bytes = pickle.dumps(saved)
    if boundary == "snapshot":
        branch: Any = engine.snapshot()
    elif boundary == "copy":
        branch = engine.copy()
    else:
        branch = GameEngine(COLORS, seed=9, shuffle_players=False)
        branch.restore(saved)
    assert branch.state.rng is not engine.rng
    assert branch.state.rng.getstate() == engine.rng.getstate()
    if boundary != "snapshot":
        assert branch.rng is branch.state.rng

    coordinate = next(iter(branch.state.board.map.land_tiles))
    tile = branch.state.board.map.land_tiles[coordinate]
    assert branch.state.board.map.tiles[coordinate] is tile
    assert branch.state.board.map.tiles_by_id[tile.id] is tile
    tile.nodes.clear()
    tile.number = 12
    branch.state.board.map.node_production.clear()
    branch.state.actions[0].value.willing_by.add(Color.WHITE)
    branch.state.trade_window.offers[branch.state.actions[0].value.id].give = (4, 0, 0, 0, 0)
    branch.events[1].public_payload["give"]["WOOD"] = 9
    branch.events[0].private_overlays[0][1]["text"] = "Branch-only text"
    branch.commitments[0].promise = "Branch-only promise"
    branch.history[0][1].value.give = (3, 0, 0, 0, 0)
    branch.history[1][0].actions[0].value.declined_by.add(Color.WHITE)
    branch.history[0][0].board.map.land_tiles[coordinate].edges.clear()
    branch.history[0][3][0].status = CommitmentStatus.VIOLATED
    branch.state.rng.random()

    assert pickle.dumps(engine) == original_bytes
    assert pickle.dumps(saved) == saved_bytes


def test_history_capture_owns_nested_map_actions_and_commitments(engine: GameEngine, offer_action: Action) -> None:
    _message(engine)
    engine.step(offer_action)
    offer_id: Any = engine.state.actions[0].value.id
    accept = Action(Color.BLUE, ActionType.ACCEPT_TRADE, offer_id)
    engine.step(accept)
    coordinate = next(iter(engine.state.board.map.land_tiles))
    original_nodes = dict(engine.state.board.map.land_tiles[coordinate].nodes)
    history_before = pickle.dumps(engine.history)

    engine.state.board.map.land_tiles[coordinate].nodes.clear()
    engine.state.actions[0].value.give = (4, 0, 0, 0, 0)
    offer_action.value.give = (3, 0, 0, 0, 0)
    engine.commitments[0].promise = "Rewritten after history capture"

    assert pickle.dumps(engine.history) == history_before
    assert engine.undo() == accept
    assert engine.state.board.map.land_tiles[coordinate].nodes == original_nodes
    assert engine.state.actions[0].value.give == (1, 0, 0, 0, 0)
    assert engine.state.trade_window.offers[offer_id].willing_by == set()
    assert engine.commitments[0].promise == "Offer ORE"
    assert engine.revision == 2
    assert engine.rng is engine.state.rng


def test_restore_detaches_from_later_snapshot_mutation_and_replays_rng(engine: GameEngine) -> None:
    engine.step(Action(Color.RED, ActionType.END_TURN, None))
    saved = engine.snapshot()
    expected = engine.step(Action(Color.BLUE, ActionType.ROLL, None))
    expected_rng = engine.rng.getstate()
    engine.restore(saved)
    before = pickle.dumps(engine)
    saved.state.board.map.land_tiles.clear()
    saved.state.actions.clear()
    saved.history[0][0].resource_freqdeck[0] = 0
    saved.state.rng.random()

    assert pickle.dumps(engine) == before
    assert engine.rng is engine.state.rng
    assert engine.step(Action(Color.BLUE, ActionType.ROLL, None)) == expected
    assert engine.rng.getstate() == expected_rng


@pytest.mark.parametrize("projection", ["engine", "state"])
def test_observation_detaches_nested_map_actions_history_and_trade(
    engine: GameEngine, offer_action: Action, projection: str
) -> None:
    engine.step(offer_action)
    # Lists are supported forced replay dice and exercise nested action values.
    engine.step(Action(Color.RED, ActionType.ROLL, [1, 2]), force=True)
    engine.state.playable_actions = [offer_action]
    recent: Any = engine.state.actions
    observation: Any = (
        engine.observe(Color.RED)
        if projection == "engine"
        else observe_state(engine.state, Color.RED, recent_events=recent)
    )
    other = engine.observe(Color.BLUE)
    before = pickle.dumps(engine)
    other_bytes = pickle.dumps(other)

    next(iter(observation.board_map.land_tiles.values())).nodes.clear()
    next(iter(observation.board_map.node_production.values())).clear()
    next(iter(observation.board_map.adjacent_tiles.values())).clear()
    observation.buildings_dict[0] = (Color.RED, CITY)
    observation.my_settlements.append(0)
    observation.opponent_roads[Color.BLUE].append((0, 1))
    observation.my_resources["WOOD"] = 99
    observation.valid_actions[0].value.willing_by.add(Color.WHITE)
    observation.trade_window.offers[recent[0].value.id].give = (4, 0, 0, 0, 0)
    if isinstance(observation.last_dice_roll, list):
        observation.last_dice_roll[0] = 6
    else:
        assert observation.last_dice_roll == (1, 2)
    if projection == "state":
        observation.recent_events[0].value.declined_by.add(Color.WHITE)
        if isinstance(observation.recent_events[-1].value, list):
            observation.recent_events[-1].value[1] = 6
        else:
            assert observation.recent_events[-1].value == (1, 2)

    assert pickle.dumps(engine) == before
    assert pickle.dumps(other) == other_bytes
    assert tuple(engine.observe(Color.RED).last_dice_roll) == (1, 2)


@pytest.mark.parametrize("terminal", [False, True])
def test_serialization_and_observation_agree_on_terminal_menu(engine: GameEngine, terminal: bool) -> None:
    engine.step(Action(Color.RED, ActionType.ROLL, (1, 2)), force=True)
    engine.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 10 if terminal else 9
    before = pickle.dumps(engine)
    raw_menu = deepcopy(engine.state.playable_actions)
    assert raw_menu

    payload = json.loads(json.dumps(engine, cls=GameEncoder))

    assert payload["winning_color"] == ("RED" if terminal else None)
    assert payload["current_playable_actions"] == json.loads(
        json.dumps([] if terminal else raw_menu, cls=GameEncoder)
    )
    assert engine.observe(Color.RED).valid_actions == ([] if terminal else raw_menu)
    assert all(engine.observe(color).valid_actions == [] for color in COLORS[1:])
    assert payload["actions"] == [["RED", "ROLL", [1, 2]]]
    assert engine.state.playable_actions == raw_menu
    assert pickle.dumps(engine) == before


def test_history_disabled_step_does_not_deepcopy_state_or_board(engine: GameEngine, offer_action: Action, monkeypatch: pytest.MonkeyPatch) -> None:
    engine.capture_history = False

    def reject_full_copy(self: object, memo: dict[int, object]) -> None:
        pytest.fail("A history-disabled step must not deepcopy state, board, or map")

    for cls in (GameState, Board, CatanMap):
        monkeypatch.setattr(cls, "__deepcopy__", reject_full_copy, raising=False)

    engine.step(offer_action)
    _message(engine)
    assert engine.revision == 2
    assert engine.history == []
