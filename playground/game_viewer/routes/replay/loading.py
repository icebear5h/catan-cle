"""Reading one Colonist replay file and installing it as the live sandbox."""

import json
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from flask import Response, jsonify, request

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.players.baseline import ScriptedPlayer
from cle.replay.colonist.coordinates import create_map_from_colonist
from cle.replay.colonist.event_parser import parse_colonist_events_to_actions
from cle.replay.colonist.types import ColonistTileState
from cle.replay.runtime.revision import bump_replay_revision
from cle.sandbox.replay import ReplaySandbox

from ...replay.model_traces import load_paired_model_traces
from ...replay.narrator_reasoning import load_paired_narrator_reasoning
from ...replay.transcript import get_curated_replay_path, load_paired_transcript
from ...state import ServerState
from .blueprint import _broadcast

__all__ = ["_load_replay_transaction"]


def _load_replay_transaction(
    state: ServerState,
) -> Response | tuple[Response, int]:
    data = request.json or {}
    game_id = data.get('game_id', '').strip()

    if not game_id:
        return jsonify({"error": "game_id required"}), 400

    # One package level deeper than the old module, so one more parent.
    replay_dir = Path(__file__).parent.parent.parent.parent.parent / "data_pipeline" / "bootstrapping" / "data" / "raw_replays"
    curated_replay = get_curated_replay_path(game_id)
    replay_file = curated_replay if curated_replay and curated_replay.exists() else None

    if replay_file is None:
        for pattern in [f"{game_id}.json", f"{game_id}_*.json", f"*{game_id}*.json"]:
            matches = list(replay_dir.glob(pattern))
            if matches:
                replay_file = matches[0]
                break

    if not replay_file:
        return jsonify({"error": f"Replay {game_id} not found in {replay_dir}"}), 404

    try:
        with open(replay_file) as f:
            raw_data = json.load(f)

        if "data" in raw_data:
            events = raw_data["data"].get("eventHistory", {}).get("events", [])
        elif "events" in raw_data:
            events = raw_data["events"]
        else:
            events = raw_data.get("eventHistory", {}).get("events", [])

        if not events:
            return jsonify({"error": "No events found in replay"}), 400

        initial_state = None
        if "data" in raw_data:
            initial_state = (
                raw_data["data"].get("eventHistory", {}).get("initialState")
                or raw_data["data"].get("initialState")
            )
        elif "eventHistory" in raw_data:
            initial_state = raw_data["eventHistory"].get("initialState") or raw_data.get("initialState")

        tile_hex_states: dict[str, ColonistTileState] = {}
        if initial_state:
            tile_hex_states = initial_state.get("mapState", {}).get("tileHexStates", {})
            print(f"Loaded {len(tile_hex_states)} tile hex states for robber matching")

        replay_player_ids = (
            raw_data["data"].get("playOrder", [])
            if "data" in raw_data
            else raw_data.get("playOrder", [])
        )
        parsed_actions = parse_colonist_events_to_actions(
            events,
            tile_hex_states,
            player_ids=replay_player_ids,
        )
        paired_transcript = load_paired_transcript(game_id, events, parsed_actions)
        archived_player_perspective = (
            raw_data.get("data", {}).get("playerPerspective")
            if "data" in raw_data
            else raw_data.get("playerPerspective")
        )
        paired_model_traces = load_paired_model_traces(
            game_id,
            narrator=cast(
                Mapping[str, object] | None, (paired_transcript or {}).get("narrator")
            ),
            archived_player_perspective=archived_player_perspective,
        )
        paired_narrator_reasoning = load_paired_narrator_reasoning(
            game_id,
            paired_transcript,
        )

        COLONIST_COLOR_NAMES = {
            1: "red", 2: "blue", 3: "orange", 4: "green", 5: "black",
            6: "bronze", 7: "silver", 8: "gold", 9: "white",
            10: "pink", 11: "mystic_blue",
        }

        COLONIST_TO_ENGINE_COLOR = {
            1: Color.RED, 2: Color.BLUE, 3: Color.ORANGE, 4: Color.GREEN,
            5: Color.BLACK, 6: Color.BRONZE, 7: Color.SILVER,
            8: Color.GOLD, 9: Color.WHITE, 10: Color.PINK,
            11: Color.MYSTIC_BLUE,
        }

        FALLBACK_ENGINE_COLORS = [Color.ORANGE, Color.BRONZE, Color.SILVER, Color.GOLD, Color.PINK, Color.MYSTIC_BLUE]

        colonist_players: list[dict[str, object]] = []
        play_order_indices: list[int] = []
        play_order_colors = []
        player_states = []

        if "data" in raw_data:
            player_states = raw_data["data"].get("playerUserStates", [])
            play_order_colors = raw_data["data"].get("playOrder", [])

            color_to_player_idx: dict[object, int] = {}
            for idx, p in enumerate(player_states):
                color_id = p.get("selectedColor")
                color_to_player_idx[color_id] = idx
                colonist_players.append({
                    "username": p.get("username"),
                    "color": COLONIST_COLOR_NAMES.get(color_id, f"color_{color_id}"),
                    "userId": str(p.get("userId")),
                })

            for color_id in play_order_colors:
                player_idx = color_to_player_idx.get(color_id)
                if player_idx is not None:
                    play_order_indices.append(player_idx)

        catan_map = None
        if initial_state:
            catan_map = create_map_from_colonist(initial_state)

        if play_order_colors:
            colors: list[Color] = []
            fallback_idx = 0
            for color_id in play_order_colors:
                if color_id in COLONIST_TO_ENGINE_COLOR:
                    engine_color = COLONIST_TO_ENGINE_COLOR[color_id]
                else:
                    engine_color = FALLBACK_ENGINE_COLORS[fallback_idx % len(FALLBACK_ENGINE_COLORS)]
                    fallback_idx += 1
                    print(f"Warning: Unknown Colonist color ID {color_id}, using fallback {engine_color}")
                colors.append(engine_color)
            print(f"Created players with colors: {colors}")
        else:
            colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]

        engine = GameEngine(
            colors,
            catan_map=catan_map,
            shuffle_players=False,
            capture_history=True,
        )

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
            "colonist_color_to_engine_idx": {str(color_id): idx for idx, color_id in enumerate(play_order_colors)},
            "tile_hex_states": tile_hex_states,
            "player_perspective": archived_player_perspective,
            "paired_transcript": paired_transcript,
            "paired_model_traces": paired_model_traces,
            "paired_narrator_reasoning": paired_narrator_reasoning,
        }
        state.replay_index = 0
        state.replay_actions_per_step = []
        state.first_divergence_step = {}
        state.replay_semantic_issues = []
        state.replay_final_state_synced = False
        state.replay_pending_dev_card = None
        state.replay_step_checkpoints = []
        state.replay_trade_ledger = {}
        state.current_sandbox = ReplaySandbox(
            state,
            engine,
            players={color: ScriptedPlayer(color) for color in engine.state.colors},
        )
        bump_replay_revision(state)
        state.replay_mode = True
        state.game_running = True
        state.game_log = [{
            "type": "general",
            "timestamp": time.time(),
            "message": f"Loaded Colonist replay {game_id} ({len(parsed_actions)} actions)"
        }]

        _broadcast()

        return jsonify({
            "status": "loaded",
            "game_id": game_id,
            "total_events": len(parsed_actions),
            "file": str(replay_file),
            "has_paired_transcript": paired_transcript is not None,
            "has_paired_model_traces": paired_model_traces is not None,
            "has_paired_narrator_reasoning": paired_narrator_reasoning is not None,
        })

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
