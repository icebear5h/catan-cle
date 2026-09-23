"""Engine-backed public board contracts and deterministic QA generation."""

from __future__ import annotations

from typing import List, Optional

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import CITY, SETTLEMENT
from cle.game_engine.models.map import NUM_NODES
from cle.game_engine.state_functions import (
    get_largest_army,
    get_longest_road_color,
    get_longest_road_length,
    get_played_dev_cards,
    get_visible_victory_points,
)
from evals.catan_board_bench.builder.questions_board import (
    add_achievement_questions,
    add_robber_questions,
    add_tile_questions,
)
from evals.catan_board_bench.builder.questions_graph import (
    add_node_edge_questions,
    add_port_questions,
)
from evals.catan_board_bench.builder.questions_players import (
    add_player_questions,
    add_topology_questions,
)
from evals.catan_board_bench.builder.selection import _find_by_id
from evals.catan_board_bench.builder.shapes import json_list, member, rows, whole_number
from evals.catan_board_bench.builder.topology import (
    _node_ports,
    _owned_road_count,
    _playable_edges,
    _port_coordinates,
    _port_nodes_by_id,
    _tile_coordinates,
)
from evals.catan_board_bench.tokens import (
    building_token,
    canonical_edge,
    color_token,
    edge_token,
    node_token,
    object_token,
    port_token,
    resource_token,
    tile_token,
)
from evals.json_types import JsonDict, JsonList, as_dict


class CatanObservationSuite:
    """Engine-backed public board contracts and QA generation."""

    schema = "catan_public_board_contract/v0"

    def public_board_contract(
        self,
        game: GameEngine,
        *,
        sample: Optional[JsonDict] = None,
        source: Optional[JsonDict] = None,
    ) -> JsonDict:
        state = game.state
        board = state.board
        catan_map = board.map

        tile_coordinates = _tile_coordinates(catan_map)
        port_coordinates = _port_coordinates(catan_map)
        port_nodes = _port_nodes_by_id(catan_map)
        node_ports = _node_ports(catan_map)
        playable_edges = _playable_edges(catan_map)
        robber_tile = catan_map.land_tiles[board.robber_coordinate]

        players: JsonList = []
        for color in state.colors:
            players.append(
                {
                    "color": color.value,
                    "color_token": color_token(color),
                    "visible_victory_points": get_visible_victory_points(state, color),
                    "settlement_count": len(state.buildings_by_color[color][SETTLEMENT]),
                    "city_count": len(state.buildings_by_color[color][CITY]),
                    "road_count": _owned_road_count(board.roads, color),
                    "longest_road_length": get_longest_road_length(state, color),
                    "played_knights": get_played_dev_cards(state, color, "KNIGHT"),
                }
            )

        tiles: JsonList = []
        for tile_id, tile in sorted(catan_map.tiles_by_id.items()):
            resource = tile.resource
            tiles.append(
                {
                    "id": tile_id,
                    "token": tile_token(tile_id),
                    "coord": list(tile_coordinates[tile_id]),
                    "resource": resource,
                    "resource_token": resource_token(resource),
                    "number": tile.number,
                    "has_robber": tile.id == robber_tile.id,
                    "nodes": json_list(sorted(tile.nodes.values())),
                    "node_tokens": [node_token(node_id) for node_id in sorted(tile.nodes.values())],
                    "edges": [
                        list(edge)
                        for edge in sorted(canonical_edge(e) for e in tile.edges.values())
                    ],
                    "edge_tokens": [
                        edge_token(e)
                        for e in sorted(canonical_edge(e) for e in tile.edges.values())
                    ],
                }
            )

        nodes: JsonList = []
        for node_id in range(NUM_NODES):
            building = board.buildings.get(node_id)
            building_color = building[0] if building else None
            building_type = building[1] if building else None
            adjacent_tiles = sorted(tile.id for tile in catan_map.adjacent_tiles.get(node_id, []))
            adjacent_edges = sorted(edge for edge in playable_edges if node_id in edge)
            attached_ports = sorted(node_ports.get(node_id, []))
            nodes.append(
                {
                    "id": node_id,
                    "token": node_token(node_id),
                    "building": building_type,
                    "building_token": building_token(building_type) if building_type else None,
                    "color": building_color.value if building_color else None,
                    "color_token": color_token(building_color) if building_color else None,
                    "adjacent_tiles": json_list(adjacent_tiles),
                    "adjacent_tile_tokens": [tile_token(tile_id) for tile_id in adjacent_tiles],
                    "adjacent_edges": [list(edge) for edge in adjacent_edges],
                    "adjacent_edge_tokens": [edge_token(edge) for edge in adjacent_edges],
                    "port_ids": json_list(attached_ports),
                    "port_tokens": [port_token(port_id) for port_id in attached_ports],
                }
            )

        edges: JsonList = []
        for edge in playable_edges:
            road_color = board.roads.get(edge) or board.roads.get((edge[1], edge[0]))
            edges.append(
                {
                    "id": list(edge),
                    "token": edge_token(edge),
                    "nodes": list(edge),
                    "node_tokens": [node_token(node_id) for node_id in edge],
                    "road_color": road_color.value if road_color else None,
                    "road_color_token": color_token(road_color) if road_color else None,
                }
            )

        ports: JsonList = []
        for port_id, port in sorted(catan_map.ports_by_id.items()):
            resource = port.resource
            attached_nodes = port_nodes[port_id]
            ports.append(
                {
                    "id": port_id,
                    "token": port_token(port_id),
                    "coord": list(port_coordinates[port_id]),
                    "direction": port.direction.value,
                    "kind": "generic" if resource is None else "resource",
                    "ratio": "3:1" if resource is None else "2:1",
                    "resource": resource,
                    "resource_token": resource_token(resource) if resource is not None else None,
                    "attached_nodes": json_list(attached_nodes),
                    "attached_node_tokens": [node_token(node_id) for node_id in attached_nodes],
                }
            )

        road_holder = get_longest_road_color(state)
        largest_army_color, largest_army_size = get_largest_army(state)

        return {
            "schema": self.schema,
            "sample": sample or {},
            "source": source or {},
            "current": {
                "current_color": state.current_color().value,
                "current_color_token": color_token(state.current_color()),
                "current_prompt": state.current_prompt.value,
                "turn_index": state.current_turn_index,
                "player_index": state.current_player_index,
                "num_completed_turns": state.num_turns,
                "is_initial_build_phase": state.is_initial_build_phase,
            },
            "players": players,
            "tiles": tiles,
            "nodes": nodes,
            "edges": edges,
            "ports": ports,
            "robber": {
                "object_token": object_token("ROBBER"),
                "tile_id": robber_tile.id,
                "tile_token": tile_token(robber_tile.id),
                "coord": list(board.robber_coordinate),
            },
            "achievements": {
                "longest_road": {
                    "holder": road_holder.value if road_holder else None,
                    "holder_token": color_token(road_holder) if road_holder else None,
                    "length": get_longest_road_length(state, road_holder) if road_holder else 0,
                },
                "largest_army": {
                    "holder": largest_army_color.value if largest_army_color else None,
                    "holder_token": color_token(largest_army_color) if largest_army_color else None,
                    "size": largest_army_size or 0,
                },
            },
        }

    def qa_pairs(self, contract: JsonDict, questions_per_sample: int = 32) -> List[JsonDict]:
        """Build deterministic engine-scored QA items for one contract."""

        sample = as_dict(contract.get("sample", {}), "contract sample")
        sample_id = sample.get("id", "sample_unknown")
        sample_index = whole_number(sample.get("index", 0), "sample index")
        image_path = sample.get("image_path")
        contract_path = sample.get("contract_path")

        qas: List[JsonDict] = []

        def add(
            category: str, question: str, answer: str, target: JsonDict, scoring: str = "exact"
        ) -> None:
            qa_id = f"{sample_id}_q{len(qas):02d}_{category}"
            qas.append(
                {
                    "id": qa_id,
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "contract_path": contract_path,
                    "category": category,
                    "question": question,
                    "answer": answer,
                    "target": target,
                    "scoring": scoring,
                }
            )

        robber = member(contract, "robber")
        robber_tile = _find_by_id(rows(contract, "tiles"), robber["tile_id"])
        if robber_tile is None:
            raise ValueError(f"robber tile missing from contract: {robber}")

        add_robber_questions(contract, robber, robber_tile, add)
        add_achievement_questions(contract, add)
        add_tile_questions(contract, sample_index, robber_tile, add)
        add_node_edge_questions(contract, sample_index, add)
        add_port_questions(contract, sample_index, add)
        add_player_questions(contract, sample_index, add)
        add_topology_questions(contract, sample_index, add)

        return qas[:questions_per_sample]
