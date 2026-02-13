"""
Classes to encode/decode catanatron classes to JSON format.
"""

import json
from enum import Enum

from engine.models.map import Water, Port, LandTile, PORT_DIRECTION_TO_NODEREFS
from engine.game import Game
from engine.models.player import Color
from engine.models.enums import RESOURCES, Action, ActionType
from engine.state_functions import get_longest_road_length, get_state_index


def longest_roads_by_player(state):
    result = dict()
    for color in state.colors:
        result[color.value] = get_longest_road_length(state, color)
    return result


def played_knights_by_player(state):
    result = dict()
    for idx, color in enumerate(state.colors):
        player_key = f"P{idx}"
        result[color.value] = state.player_state.get(f"{player_key}_PLAYED_KNIGHT", 0)
    return result


def action_from_json(data):
    color = Color[data[0]]
    action_type = ActionType[data[1]]
    if action_type == ActionType.BUILD_ROAD:
        action = Action(color, action_type, tuple(data[2]))
    elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        resources = tuple(data[2])
        if len(resources) not in [1, 2]:
            raise ValueError("Year of Plenty action must have 1 or 2 resources")
        action = Action(color, action_type, resources)
    elif action_type == ActionType.MOVE_ROBBER:
        # MOVE_ROBBER now only has coordinate (STEAL is separate)
        coordinate = tuple(data[2])
        action = Action(color, action_type, coordinate)
    elif action_type == ActionType.STEAL:
        victim, resource = data[2]
        victim = Color[victim] if victim else None
        action = Action(color, action_type, (victim, resource))
    elif action_type == ActionType.MARITIME_TRADE:
        value = tuple(data[2])
        action = Action(color, action_type, value)
    else:
        action = Action(color, action_type, data[2])
    return action


class GameEncoder(json.JSONEncoder):
    def default(self, obj):
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, tuple):
            return obj
        if isinstance(obj, Game):
            nodes = {}
            edges = {}
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
                for direction, edge in tile.edges.items():
                    color = obj.state.board.roads.get(edge, None)
                    edge_id = tuple(sorted(edge))
                    edges[edge_id] = {
                        "id": edge_id,
                        "tile_coordinate": coordinate,
                        "direction": self.default(direction),
                        "color": self.default(color),
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
                "bot_colors": list(
                    map(
                        lambda p: p.color, filter(lambda p: p.is_bot, obj.state.players)
                    )
                ),
                "is_initial_build_phase": obj.state.is_initial_build_phase,
                "robber_coordinate": obj.state.board.robber_coordinate,
                "current_color": obj.state.current_color(),
                "current_prompt": obj.state.current_prompt,
                "current_playable_actions": obj.state.playable_actions,
                "longest_roads_by_player": longest_roads_by_player(obj.state),
                "played_knights_by_player": played_knights_by_player(obj.state),
                "winning_color": obj.winning_color(),
                "state_index": get_state_index(obj.state),
            }
        if isinstance(obj, Water):
            return {"type": "WATER"}
        if isinstance(obj, Port):
            # Map resource to emoji for frontend display
            resource_emoji_map = {
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
