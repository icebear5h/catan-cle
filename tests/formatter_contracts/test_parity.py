"""Full pre-refactor text witnesses plus public identity and sharing contracts."""

import hashlib
import json
import os
import pickle
import subprocess
from dataclasses import MISSING, fields, replace
from itertools import product
from pathlib import Path

from cle.env.observation_formatter import (
    CatanObservation,
    CatanObservationFormatter,
    FormattedObservation,
    create_observation_from_state,
)
from cle.game_engine.models.actions import trade_response_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.observation import PlayerObservation, observe_state

from .fixtures import COLORS, funded_engine, game_states


def witnesses() -> dict[str, str]:
    result: dict[str, str] = {}
    formatter = CatanObservationFormatter()
    for label, engine in game_states():
        for color in COLORS:
            observations = (
                create_observation_from_state(engine.state, color, engine.state.actions),
                observe_state(engine.state, color, engine.state.actions),
            )
            for obs in observations:
                prefix = f"{label}/{color.value}/{type(obs).__name__}"
                # The public action helper must work before the coordinate cache exists.
                actions = [*obs.valid_actions, *trade_response_actions(engine.state, color)]
                for index, action in enumerate(actions):
                    result[f"{prefix}/action-{index}"] = (
                        CatanObservationFormatter()._format_single_action(action, obs)
                    )
                result[f"{prefix}/discard-count"] = formatter._format_single_action(
                    Action(color, ActionType.DISCARD, None), obs, discard_count=3,
                )
                for legal, order, shared in product((False, True), repeat=3):
                    formatted = formatter.format(
                        obs, include_legal_actions=legal,
                        include_initial_placement_order=order, shared=shared,
                    )
                    for item in fields(formatted):
                        result[f"{prefix}/{int(legal)}{int(order)}{int(shared)}/{item.name}"] = (
                            getattr(formatted, item.name)
                        )
    return result


def test_full_formatting_witnesses() -> None:
    # Engine action sets depend on the process hash seed as well as the game RNG.
    # Keep the captured menu order, rather than sorting away observable differences.
    if os.environ.get("PYTHONHASHSEED") != "0":
        completed = subprocess.run(
            ["uv", "run", "--no-sync", "python", "-m", "pytest",
             f"{__file__}::test_full_formatting_witnesses", "-q", "-s"],
            env={**os.environ, "PYTHONHASHSEED": "0"},
            capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return
    actual = witnesses()
    encoded = json.dumps(actual, ensure_ascii=False, separators=(",", ":")).encode()
    # Captured with the original, verified-clean 1,034-line module from Git.
    assert len(actual) == 24700
    assert len(encoded) == 13656754
    assert hashlib.sha256(encoded).hexdigest() == (
        "4c76ae8ffe95a569dba5b5dcdf0c3be67ec84d24b095a463d6716148c9d17df7"
    )
    destination = os.environ.get("FORMATTER_WITNESS")
    if destination:
        path = Path(destination)
        if os.environ.get("FORMATTER_CAPTURE") == "1":
            assert not path.exists(), "Never overwrite the pre-edit witness"
            path.write_bytes(encoded)
            engine = funded_engine()
            obs = create_observation_from_state(engine.state, Color.RED)
            formatter = CatanObservationFormatter()
            path.with_suffix(".pickle").write_bytes(pickle.dumps((
                obs, formatter.format(obs), formatter,
            )))
        else:
            expected_bytes = path.read_bytes()
            expected = json.loads(expected_bytes)
            differences = [key for key in actual if actual[key] != expected.get(key)]
            assert not differences, [
                (key, expected.get(key), actual[key]) for key in differences[:3]
            ]
            assert encoded == expected_bytes
            restored = pickle.loads(path.with_suffix(".pickle").read_bytes())
            assert type(restored[0]) is CatanObservation
            assert type(restored[1]) is FormattedObservation
            assert type(restored[2]) is CatanObservationFormatter
            assert restored[2].format(restored[0]) == restored[1]
    print(f"Witnesses: {len(actual)} strings; {len(encoded)} bytes; "
          f"sha256={hashlib.sha256(encoded).hexdigest()}")


def test_historical_identity_defaults_and_factory_sharing() -> None:
    engine = funded_engine()
    events: list[Action] = []
    obs = create_observation_from_state(engine.state, Color.RED, events)
    assert type(obs) is CatanObservation and not isinstance(obs, PlayerObservation)
    for cls in (CatanObservation, FormattedObservation, CatanObservationFormatter):
        assert cls.__module__ == "cle.env.observation_formatter"
        assert pickle.loads(pickle.dumps(cls)) is cls
    assert fields(CatanObservation)[-1].name == "recent_events"
    assert fields(CatanObservation)[-1].default is None
    assert all(item.default is MISSING for item in fields(CatanObservation)[:-1])
    assert all(item.default_factory is MISSING for item in fields(CatanObservation))
    assert all(item.default is MISSING for item in fields(FormattedObservation))
    assert tuple(item.name for item in fields(CatanObservation)) == (
        "my_color", "my_settlements", "my_cities", "my_roads", "opponent_settlements",
        "opponent_cities", "opponent_roads", "my_resources", "my_dev_cards",
        "opponent_resource_counts", "opponent_dev_card_counts", "current_turn", "current_phase",
        "turn_order", "last_dice_roll", "robber_position", "my_vp", "opponent_vps",
        "longest_road_holder", "largest_army_holder", "my_longest_road_length", "valid_actions",
        "board_map", "buildings_dict", "trade_window", "is_my_turn", "turn_player_color",
        "recent_events",
    )
    assert tuple(item.name for item in fields(FormattedObservation)) == (
        "raw_str", "board_state", "resources", "opponents", "valid_actions",
        "strategic_context", "trade_context",
    )
    assert obs.recent_events is events
    assert obs.valid_actions is engine.state.playable_actions
    assert obs.board_map is engine.state.board.map
    assert obs.buildings_dict is engine.state.board.buildings
    assert obs.my_settlements is engine.state.buildings_by_color[Color.RED]["SETTLEMENT"]
    assert obs.my_roads is engine.state.buildings_by_color[Color.RED]["ROAD"]
    # Missing building collections historically return fresh empty lists.
    assert "CITY" not in engine.state.buildings_by_color[Color.RED]
    assert obs.my_cities == []
    assert obs.opponent_settlements[Color.BLUE] is (
        engine.state.buildings_by_color[Color.BLUE]["SETTLEMENT"]
    )
    assert obs.trade_window is engine.state.trade_window
    assert obs.last_dice_roll is None
    other = create_observation_from_state(engine.state, Color.RED)
    third = create_observation_from_state(engine.state, Color.RED)
    assert other.recent_events == third.recent_events == []
    assert other.recent_events is not third.recent_events
    explicit_none = replace(obs, recent_events=None)
    assert explicit_none.recent_events == [] and explicit_none.recent_events is not events
    assert obs.my_cities is not other.my_cities
    formatter = CatanObservationFormatter()
    assert not hasattr(formatter, "_node_coords")
    formatter.format(obs)
    previous = formatter._node_coords
    formatter.format(other)
    assert formatter._node_coords == previous and formatter._node_coords is not previous


def test_opponent_private_cards_do_not_change_rendering() -> None:
    engine = funded_engine()
    before = [factory(engine.state, Color.RED) for factory in
              (create_observation_from_state, observe_state)]
    engine.state.player_state["P1_WOOD_IN_HAND"] -= 1
    engine.state.player_state["P1_ORE_IN_HAND"] += 1
    engine.state.player_state["P1_VICTORY_POINT_IN_HAND"] -= 1
    engine.state.player_state["P1_KNIGHT_IN_HAND"] += 1
    engine.state.player_state["P1_ACTUAL_VICTORY_POINTS"] -= 1
    after = [factory(engine.state, Color.RED) for factory in
             (create_observation_from_state, observe_state)]
    for old, new in zip(before, after):
        for shared in (False, True):
            assert CatanObservationFormatter().format(old, shared=shared) == (
                CatanObservationFormatter().format(new, shared=shared)
            )
