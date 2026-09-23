"""Authoritative public-board snapshot at one pre-action transcript cursor."""

from __future__ import annotations

from typing import List

from cle.game_engine.models.enums import CITY, SETTLEMENT
from cle.game_engine.state import GameState
from cle.replay.activity import (
    format_visible_replay_activity,
    select_recent_activity_rows,
)
from cle.replay.contracts import (
    ReplayRuntimeState,
    mapping_field,
    optional_mapping_field,
)
from evals.json_types import JsonDict, as_int
from evals.transcript_reasoning.support import (
    _colonist_mapping,
    _color_name,
    _normalize_edge,
    _stable_hash,
    json_value,
    parsed_action_rows,
    require_engine,
    require_replay_data,
)


def _public_player_summaries(game_state: GameState) -> List[JsonDict]:
    board = game_state.board
    unique_roads = {
        (*_normalize_edge(edge), _color_name(color))
        for edge, color in board.roads.items()
    }
    summaries: List[JsonDict] = []
    for player_index, color in enumerate(game_state.colors):
        color_name = _color_name(color)
        buildings = game_state.buildings_by_color.get(color, {})
        summaries.append(
            {
                "color": color_name,
                "public_victory_points": game_state.player_state.get(
                    f"P{player_index}_VICTORY_POINTS", 0
                ),
                "settlements": len(buildings.get(SETTLEMENT, [])),
                "cities": len(buildings.get(CITY, [])),
                "roads": sum(road_color == color_name for _, _, road_color in unique_roads),
                "longest_road_length": game_state.player_state.get(
                    f"P{player_index}_LONGEST_ROAD_LENGTH", 0
                ),
                "played_knights": game_state.player_state.get(
                    f"P{player_index}_PLAYED_KNIGHT", 0
                ),
            }
        )
    return summaries


def _tile_sort_key(item: JsonDict) -> int:
    return as_int(item["tile_id"], "adjacent tile_id")


def _road_sort_key(item: JsonDict) -> tuple[bool, int]:
    edge_id = item["colonist_edge_id"]
    return edge_id is None, -1 if edge_id is None else as_int(edge_id, "colonist_edge_id")


def build_public_board_snapshot(state: ReplayRuntimeState) -> JsonDict:
    """Return public board facts without hidden hands or the upcoming action."""
    game_state = require_engine(state).state
    replay_data = require_replay_data(state)
    board = game_state.board
    catan_map = board.map
    corner_to_node = _colonist_mapping(state.corner_to_node_map)
    node_to_corner = {node: corner for corner, node in corner_to_node.items()}
    edge_to_nodes = {
        int(key.removeprefix("_")): _normalize_edge(value)
        for key, value in state.edge_to_edge_map.items()
    }
    nodes_to_edge = {nodes: edge for edge, nodes in edge_to_nodes.items()}
    coordinate_by_tile_identity = {
        id(tile): tuple(coordinate) for coordinate, tile in catan_map.land_tiles.items()
    }

    tiles: List[JsonDict] = []
    for coordinate, tile in sorted(catan_map.land_tiles.items()):
        tiles.append(
            {
                "tile_id": int(tile.id),
                "coordinate": list(coordinate),
                "resource": tile.resource,
                "number": tile.number,
                "robber": tuple(coordinate) == tuple(board.robber_coordinate),
            }
        )

    buildings: List[JsonDict] = []
    for node, (color, building) in sorted(board.buildings.items()):
        adjacent_tiles: List[JsonDict] = []
        for tile in catan_map.adjacent_tiles[node]:
            adjacent_tiles.append(
                {
                    "tile_id": int(tile.id),
                    "coordinate": list(coordinate_by_tile_identity[id(tile)]),
                    "resource": tile.resource,
                    "number": tile.number,
                }
            )
        ports: List[str] = []
        for resource, port_nodes in catan_map.port_nodes.items():
            if node in port_nodes:
                ports.append("3:1" if resource is None else f"2:1 {resource}")
        sorted_tiles = sorted(adjacent_tiles, key=_tile_sort_key)
        sorted_ports = sorted(ports)
        buildings.append(
            {
                "colonist_corner_id": node_to_corner.get(int(node)),
                "engine_node_id": int(node),
                "color": _color_name(color),
                "building": _color_name(building),
                "adjacent_tiles": list(sorted_tiles),
                "ports": list(sorted_ports),
            }
        )

    roads: List[JsonDict] = []
    seen_roads = set()
    for edge, color in board.roads.items():
        normalized = _normalize_edge(edge)
        if normalized in seen_roads:
            continue
        seen_roads.add(normalized)
        roads.append(
            {
                "colonist_edge_id": nodes_to_edge.get(normalized),
                "engine_nodes": list(normalized),
                "color": _color_name(color),
            }
        )
    roads.sort(key=_road_sort_key)

    activity_rows, activity_window = select_recent_activity_rows(
        parsed_action_rows(replay_data), state.replay_index
    )
    public_activity = [
        format_visible_replay_activity(
            action,
            replay_data,
            game_state.colors,
            observer_color=None,
        )
        for action in activity_rows
    ]
    narrator = optional_mapping_field(
        optional_mapping_field(replay_data, "paired_transcript"), "narrator"
    )
    narrator_index = mapping_field(replay_data, "colonist_color_to_engine_idx").get(
        str(narrator.get("colonist_color"))
    )
    narrator_color = (
        _color_name(game_state.colors[narrator_index])
        if isinstance(narrator_index, int) and 0 <= narrator_index < len(game_state.colors)
        else None
    )

    public_payload: JsonDict = {
        "game_id": str(replay_data.get("game_id")),
        "replay_index": state.replay_index,
        "phase": (
            "initial_placement" if game_state.is_initial_build_phase else "main_game"
        ),
        "turn_number": game_state.num_turns,
        "current_player_color": _color_name(game_state.current_color()),
        "narrator_color": narrator_color,
        "tiles": list(tiles),
        "buildings": list(buildings),
        "roads": list(roads),
        "players": list(_public_player_summaries(game_state)),
        "recent_public_activity": list(public_activity),
        "activity_window": json_value(activity_window, "activity_window"),
    }
    public_payload["board_state_hash"] = _stable_hash(public_payload)
    return public_payload

