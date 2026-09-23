"""
Classes to encode/decode catanatron classes to JSON format.
"""

import json
from collections.abc import Mapping, Sequence
from enum import Enum

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.map import PORT_DIRECTION_TO_NODEREFS, LandTile, Port, Water
from cle.game_engine.models.player import Color
from cle.game_engine.state import GameState
from cle.game_engine.state_functions import get_longest_road_length, get_state_index
from cle.game_engine.trading import RESOURCE_NAMES, ResourceBundle, TradeCandidate, TradeOffer


def longest_roads_by_player(state: GameState) -> dict[str, int]:
    result: dict[str, int] = dict()
    for color in state.colors:
        result[color.value] = get_longest_road_length(state, color)
    return result


def played_knights_by_player(state: GameState) -> dict[str, int]:
    result: dict[str, int] = dict()
    for idx, color in enumerate(state.colors):
        player_key = f"P{idx}"
        result[color.value] = state.player_state.get(f"{player_key}_PLAYED_KNIGHT", 0)
    return result


def _json_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object, got {type(value).__name__}")
    return value


def _json_sequence(value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Expected a JSON array, got {type(value).__name__}")
    return value


def _json_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected an integer, got {type(value).__name__}")
    return value


def _json_str(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Expected a string, got {type(value).__name__}")
    return value


def _json_optional_str(value: object) -> str | None:
    return None if value is None else _json_str(value)


def _bundle_from_json(value: object) -> ResourceBundle:
    named = _json_mapping(value)
    counts = tuple(_json_int(named.get(resource, 0)) for resource in RESOURCE_NAMES)
    return (counts[0], counts[1], counts[2], counts[3], counts[4])


def _trade_offer_from_json(data: Mapping[str, object], actor: Color) -> TradeOffer:
    give = _bundle_from_json(data.get("give", {}))
    receive = _bundle_from_json(data.get("receive", {}))
    return TradeOffer(
        id=_json_optional_str(data.get("id")),
        offered_by=Color(data.get("offered_by", actor.value)),
        audience=frozenset(Color(color) for color in _json_sequence(data["audience"])),
        give=give,
        receive=receive,
        give_any=_json_int(data.get("give_any", 0)),
        receive_any=_json_int(data.get("receive_any", 0)),
        parent_offer_id=_json_optional_str(data.get("parent_offer_id")),
    )


def action_from_json(data: Sequence[object]) -> Action:
    color = Color[_json_str(data[0])]
    action_type = ActionType[_json_str(data[1])]
    if action_type == ActionType.BUILD_ROAD:
        action = Action(color, action_type, tuple(_json_sequence(data[2])))
    elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        resources = tuple(_json_sequence(data[2]))
        if len(resources) not in [1, 2]:
            raise ValueError("Year of Plenty action must have 1 or 2 resources")
        action = Action(color, action_type, resources)
    elif action_type == ActionType.MOVE_ROBBER:
        # MOVE_ROBBER now only has coordinate (STEAL is separate)
        coordinate = tuple(_json_sequence(data[2]))
        action = Action(color, action_type, coordinate)
    elif action_type == ActionType.STEAL:
        victim_name, resource = _json_sequence(data[2])
        victim = Color[_json_str(victim_name)] if victim_name else None
        action = Action(color, action_type, (victim, resource))
    elif action_type == ActionType.MARITIME_TRADE:
        value = tuple(_json_sequence(data[2]))
        action = Action(color, action_type, value)
    elif action_type in {ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER}:
        action = Action(color, action_type, _trade_offer_from_json(_json_mapping(data[2]), color))
    elif action_type == ActionType.CONFIRM_TRADE:
        candidate = _json_mapping(data[2])
        action = Action(
            color,
            action_type,
            TradeCandidate(
                offer_id=_json_str(candidate["offer_id"]),
                turn_player=Color[_json_str(candidate["turn_player"])],
                counterparty=Color[_json_str(candidate["counterparty"])],
            ),
        )
    else:
        action = Action(color, action_type, data[2])
    return action


class GameEncoder(json.JSONEncoder):
    def default(self, obj: object) -> object:
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, TradeOffer):
            return obj.to_payload()
        if isinstance(obj, TradeCandidate):
            return obj.to_payload()
        if isinstance(obj, tuple):
            return obj
        if isinstance(obj, GameEngine):
            return self._encode_engine(obj)
        if isinstance(obj, Water):
            return {"type": "WATER"}
        if isinstance(obj, Port):
            # Map resource to emoji for frontend display
            resource_emoji_map: dict[str | None, str] = {
                "WOOD": "🪵",
                "BRICK": "🧱",
                "SHEEP": "🐑",
                "WHEAT": "🌾",
                "ORE": "⛰️",
                None: "3:1"  # Generic 3:1 port
            }
            # Get the 2 nodes this port connects
            node_refs = PORT_DIRECTION_TO_NODEREFS[obj.direction]
            port_nodes = [obj.nodes[node_refs[0]], obj.nodes[node_refs[1]]]
            return {
                "id": obj.id,
                "type": "PORT",
                "direction": self.default(obj.direction),
                "resource": self.default(obj.resource),
                "emoji": resource_emoji_map.get(obj.resource, "?"),
                "port_nodes": port_nodes,  # The 2 node IDs this port connects
            }
        if isinstance(obj, LandTile):
            if obj.resource is None:
                return {"id": obj.id, "type": "DESERT"}
            return {
                "id": obj.id,
                "type": "RESOURCE_TILE",
                "resource": self.default(obj.resource),
                "number": obj.number,
            }
        return json.JSONEncoder.default(self, obj)

    def _encode_engine(self, obj: GameEngine) -> dict[str, object]:
        nodes: dict[int, dict[str, object]] = {}
        edges: dict[tuple[int, ...], dict[str, object]] = {}
        for coordinate, tile in obj.state.board.map.tiles.items():
            for direction, node_id in tile.nodes.items():
                building = obj.state.board.buildings.get(node_id, None)
                color = None if building is None else building[0]
                building_type = None if building is None else building[1]
                nodes[node_id] = {
                    "id": node_id,
                    "tile_coordinate": coordinate,
                    "direction": self.default(direction),
                    "building": self.default(building_type),
                    "color": self.default(color),
                }
            for edge_direction, edge in tile.edges.items():
                road_color = obj.state.board.roads.get(edge, None)
                edge_id = tuple(sorted(edge))
                edges[edge_id] = {
                    "id": edge_id,
                    "tile_coordinate": coordinate,
                    "direction": self.default(edge_direction),
                    "color": self.default(road_color),
                }
        return {
            "tiles": [
                {"coordinate": coordinate, "tile": self.default(tile)}
                for coordinate, tile in obj.state.board.map.tiles.items()
            ],
            "adjacent_tiles": obj.state.board.map.adjacent_tiles,
            "nodes": nodes,
            "edges": list(edges.values()),
            "actions": [self.default(a) for a in obj.state.actions],
            "player_state": obj.state.player_state,
            "colors": obj.state.colors,
            "bot_colors": [],
            "is_initial_build_phase": obj.state.is_initial_build_phase,
            "robber_coordinate": obj.state.board.robber_coordinate,
            "current_color": obj.state.current_color(),
            "current_prompt": obj.state.current_prompt,
            "current_playable_actions": (
                obj.state.playable_actions if obj.winning_color() is None else []
            ),
            "longest_roads_by_player": longest_roads_by_player(obj.state),
            "played_knights_by_player": played_knights_by_player(obj.state),
            "winning_color": obj.winning_color(),
            "state_index": get_state_index(obj.state),
        }
