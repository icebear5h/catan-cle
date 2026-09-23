"""Colonist replay loading and stepping for benchmark generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.players.legacy import SimplePlayer
from cle.replay.colonist.coordinates import create_map_from_colonist
from cle.replay.colonist.event_parser import parse_colonist_events_to_actions
from cle.replay.contracts import ReplayArchive, ReplayOutcome
from cle.replay.runtime.step_executor import replay_step_logic
from evals.catan_board_bench.builder.constants import (
    COLONIST_COLOR_NAMES,
    COLONIST_TO_ENGINE_COLOR,
    FALLBACK_ENGINE_COLORS,
)
from evals.catan_board_bench.builder.records import _quiet_context
from evals.catan_board_bench.paths import PROJECT_ROOT
from evals.json_types import JsonDict
from playground.game_viewer.state import ServerState


def _project_relative_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def load_colonist_replay(replay_file: Path, *, quiet: bool = True) -> ServerState:
    """Load one Colonist replay into the same ServerState shape as the viewer."""

    with _quiet_context(quiet):
        raw_data = json.loads(Path(replay_file).read_text())

        if "data" in raw_data:
            events = raw_data["data"].get("eventHistory", {}).get("events", [])
        elif "events" in raw_data:
            events = raw_data["events"]
        else:
            events = raw_data.get("eventHistory", {}).get("events", [])

        if not events:
            raise ValueError(f"no events found in replay {replay_file}")

        initial_state = None
        if "data" in raw_data:
            initial_state = raw_data["data"].get("eventHistory", {}).get(
                "initialState"
            ) or raw_data["data"].get("initialState")
        elif "eventHistory" in raw_data:
            initial_state = raw_data["eventHistory"].get("initialState") or raw_data.get(
                "initialState"
            )

        tile_hex_states = {}
        if initial_state:
            tile_hex_states = initial_state.get("mapState", {}).get("tileHexStates", {})

        parsed_actions = parse_colonist_events_to_actions(events, tile_hex_states)

    state = ServerState()

    colonist_players: List[JsonDict] = []
    play_order_indices: List[int] = []
    play_order_colors: Sequence[int] = []
    player_states: Sequence[JsonDict] = []

    if "data" in raw_data:
        player_states = raw_data["data"].get("playerUserStates", [])
        play_order_colors = raw_data["data"].get("playOrder", [])

        color_to_player_idx = {}
        for idx, player_state in enumerate(player_states):
            color_id = player_state.get("selectedColor")
            color_to_player_idx[color_id] = idx
            colonist_players.append(
                {
                    "username": player_state.get("username"),
                    "color": (
                        COLONIST_COLOR_NAMES.get(color_id, f"color_{color_id}")
                        if isinstance(color_id, int)
                        else f"color_{color_id}"
                    ),
                    "userId": str(player_state.get("userId")),
                }
            )

        for color_id in play_order_colors:
            player_idx = color_to_player_idx.get(color_id)
            if player_idx is not None:
                play_order_indices.append(player_idx)

    with _quiet_context(quiet):
        catan_map = create_map_from_colonist(initial_state) if initial_state else None

    if play_order_colors:
        players = []
        fallback_idx = 0
        for color_id in play_order_colors:
            engine_color = COLONIST_TO_ENGINE_COLOR.get(color_id)
            if engine_color is None:
                engine_color = FALLBACK_ENGINE_COLORS[fallback_idx % len(FALLBACK_ENGINE_COLORS)]
                fallback_idx += 1
            players.append(SimplePlayer(engine_color))
    else:
        players = [
            SimplePlayer(Color.RED),
            SimplePlayer(Color.BLUE),
            SimplePlayer(Color.WHITE),
            SimplePlayer(Color.ORANGE),
        ]

    state.current_game = GameEngine(
        [player.color for player in players], catan_map=catan_map, shuffle_players=False
    )
    game_id = Path(replay_file).stem.replace("_sample", "")
    state.replay_data = {
        "game_id": game_id,
        "events": events,
        "parsed_actions": parsed_actions,
        "total_events": len(parsed_actions),
        "file": str(replay_file),
        "initial_state": initial_state or {},
        "end_game_state": raw_data.get("data", {}).get("eventHistory", {}).get("endGameState", {})
        if "data" in raw_data
        else raw_data.get("eventHistory", {}).get("endGameState", {}),
        "game_settings": raw_data.get("data", {}).get("gameSettings", {})
        if "data" in raw_data
        else raw_data.get("gameSettings", {}),
        "colonist_players": colonist_players,
        "play_order": play_order_indices,
        "colonist_color_to_engine_idx": {
            str(color_id): idx for idx, color_id in enumerate(play_order_colors)
        },
        "tile_hex_states": tile_hex_states,
    }
    state.replay_index = 0
    state.replay_actions_per_step = []
    state.first_divergence_step = {}
    state.replay_semantic_issues = []
    state.replay_final_state_synced = False
    state.replay_pending_dev_card = None
    state.replay_mode = True
    state.game_running = True
    state.game_log = []
    return state


def step_replay(state: ServerState, *, quiet: bool = True) -> ReplayOutcome:
    with _quiet_context(quiet):
        result = replay_step_logic(state, lambda: None)
    return result


def require_game(state: ServerState) -> GameEngine:
    """Return the loaded replay's engine; a missing one is a caller bug."""
    if state.current_game is None:
        raise RuntimeError("CatanBoardBench builder has no replay loaded: no game engine")
    return state.current_game


def require_replay_data(state: ServerState) -> ReplayArchive:
    """Return the loaded replay archive; a missing one is a caller bug."""
    if state.replay_data is None:
        raise RuntimeError("CatanBoardBench builder has no replay loaded: no replay archive")
    return state.replay_data
