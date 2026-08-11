"""Catan board observation contracts and benchmark generation.

The core idea is that the engine remains the oracle. This module turns a live
``Game`` into a public board contract, deterministic question/answer items, and
optionally 512x512 board images rendered through the existing frontend.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageOps

from engine.game import Game
from engine.models.board import get_edges
from engine.models.enums import CITY, ROAD, SETTLEMENT, RESOURCES
from engine.models.map import (
    NUM_EDGES,
    NUM_NODES,
    NUM_TILES,
    PORT_DIRECTION_TO_NODEREFS,
    CatanMap,
    LandTile,
    Port,
)
from engine.models.player import Color, SimplePlayer
from engine.state_functions import (
    get_largest_army,
    get_longest_road_color,
    get_longest_road_length,
    get_played_dev_cards,
    get_visible_victory_points,
)
from playground.game_viewer.colonist.coordinates import create_map_from_colonist
from playground.game_viewer.colonist.event_parser import parse_colonist_events_to_actions
from playground.game_viewer.replay.step_executor import replay_step_logic
from playground.game_viewer.state import ServerState

from data_pipeline.catanbench.tokens import (
    base_edges,
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "data_pipeline" / "bootstrapping" / "data" / "raw_replays"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_pipeline" / "catanbench" / "datasets" / "catanbench_100"
DEFAULT_QUESTION_DIR = DEFAULT_OUTPUT_DIR / "questions"


COLONIST_COLOR_NAMES = {
    1: "red",
    2: "blue",
    3: "orange",
    4: "green",
    5: "black",
    6: "bronze",
    7: "silver",
    8: "gold",
    9: "white",
    10: "pink",
    11: "mystic_blue",
}

COLONIST_TO_ENGINE_COLOR = {
    1: Color.RED,
    2: Color.BLUE,
    3: Color.ORANGE,
    4: Color.GREEN,
    5: Color.BLACK,
    6: Color.BRONZE,
    7: Color.SILVER,
    8: Color.GOLD,
    9: Color.WHITE,
    10: Color.PINK,
    11: Color.MYSTIC_BLUE,
}

FALLBACK_ENGINE_COLORS = [
    Color.ORANGE,
    Color.BRONZE,
    Color.SILVER,
    Color.GOLD,
    Color.PINK,
    Color.MYSTIC_BLUE,
]


JsonDict = Dict[str, Any]
EdgeId = Tuple[int, int]


@dataclass(frozen=True)
class BenchmarkBuildResult:
    """Summary of a completed benchmark build."""

    output_dir: Path
    sample_count: int
    qa_count: int
    rendered_images: int
    replay_count: int
    metadata_path: Path
    manifest_path: Path
    questions_path: Path
    answer_key_path: Path


class CatanObservationSuite:
    """Engine-backed public board contracts and QA generation."""

    schema = "catan_public_board_contract/v0"

    def public_board_contract(
        self,
        game: Game,
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

        players = []
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

        tiles = []
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
                    "nodes": sorted(tile.nodes.values()),
                    "node_tokens": [node_token(node_id) for node_id in sorted(tile.nodes.values())],
                    "edges": [list(edge) for edge in sorted(canonical_edge(e) for e in tile.edges.values())],
                    "edge_tokens": [edge_token(e) for e in sorted(canonical_edge(e) for e in tile.edges.values())],
                }
            )

        nodes = []
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
                    "adjacent_tiles": adjacent_tiles,
                    "adjacent_tile_tokens": [tile_token(tile_id) for tile_id in adjacent_tiles],
                    "adjacent_edges": [list(edge) for edge in adjacent_edges],
                    "adjacent_edge_tokens": [edge_token(edge) for edge in adjacent_edges],
                    "port_ids": attached_ports,
                    "port_tokens": [port_token(port_id) for port_id in attached_ports],
                }
            )

        edges = []
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

        ports = []
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
                    "attached_nodes": attached_nodes,
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

        sample = contract.get("sample", {})
        sample_id = sample.get("id", "sample_unknown")
        sample_index = int(sample.get("index", 0))
        image_path = sample.get("image_path")
        contract_path = sample.get("contract_path")

        qas: List[JsonDict] = []

        def add(category: str, question: str, answer: str, target: JsonDict, scoring: str = "exact") -> None:
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

        robber = contract["robber"]
        robber_tile = _find_by_id(contract["tiles"], robber["tile_id"])
        if robber_tile is None:
            raise ValueError(f"robber tile missing from contract: {robber}")

        add(
            "robber_tile",
            "Which tile is the robber on?",
            robber["tile_token"],
            {"tile_id": robber["tile_id"], "tile_token": robber["tile_token"], "coord": robber["coord"]},
        )

        add(
            "robber_resource_number",
            "Which tile is the robber on, and what resource and dice number does that tile show?",
            f"{robber['tile_token']} {_resource_number_answer(robber_tile)}",
            {
                "tile_id": robber_tile["id"],
                "tile_token": robber_tile["token"],
                "resource": robber_tile["resource"],
                "resource_token": robber_tile["resource_token"],
                "number": robber_tile["number"],
            },
        )

        add(
            "robber_adjacent_buildings",
            "Which buildings are on the six nodes touching the robber's tile?",
            _tile_occupied_nodes_answer(contract, robber_tile),
            {
                "tile_id": robber_tile["id"],
                "tile_token": robber_tile["token"],
                "occupied_nodes": _tile_occupied_nodes_target(contract, robber_tile),
            },
        )

        longest_road = contract["achievements"]["longest_road"]
        if longest_road["holder_token"]:
            add(
                "longest_road_holder",
                "Who currently holds the Longest Road award?",
                longest_road["holder_token"],
                longest_road,
            )

        largest_army = contract["achievements"]["largest_army"]
        if largest_army["holder_token"]:
            add(
                "largest_army_holder",
                "Who currently holds the Largest Army award?",
                largest_army["holder_token"],
                largest_army,
            )

        current = contract["current"]
        add(
            "current_player",
            "Which color is currently prompted to act?",
            current["current_color_token"],
            {
                "color": current["current_color"],
                "color_token": current["current_color_token"],
                "prompt": current["current_prompt"],
            },
        )

        for tile in _pick(contract["tiles"], 3, sample_index, stride=5):
            add(
                "tile_resource_number",
                f"What resource and dice number are on {tile['token']}?",
                _resource_number_answer(tile),
                {
                    "tile_id": tile["id"],
                    "tile_token": tile["token"],
                    "resource": tile["resource"],
                    "resource_token": tile["resource_token"],
                    "number": tile["number"],
                },
            )

        non_robber_tile = _pick(
            [tile for tile in contract["tiles"] if tile["id"] != robber_tile["id"]],
            1,
            sample_index + 2,
            stride=7,
        )[0]
        for tile in [robber_tile, non_robber_tile]:
            has_robber = tile["id"] == robber_tile["id"]
            add(
                "tile_has_robber",
                f"Is the robber on {tile['token']}?",
                "YES" if has_robber else "NO",
                {
                    "tile_id": tile["id"],
                    "tile_token": tile["token"],
                    "has_robber": has_robber,
                },
            )

        node_candidates = _positive_first(
            contract["nodes"],
            4,
            sample_index + 1,
            stride=7,
            predicate=lambda node: node["building"] is not None,
        )
        for node in node_candidates:
            answer = (
                "EMPTY"
                if node["building"] is None
                else f"{node['color_token']} {node['building_token']}"
            )
            add(
                "node_occupancy",
                f"What building, if any, is on {node['token']}?",
                answer,
                {
                    "node_id": node["id"],
                    "node_token": node["token"],
                    "building": node["building"],
                    "building_token": node["building_token"],
                    "color": node["color"],
                    "color_token": node["color_token"],
                },
            )

        edge_candidates = _positive_first(
            contract["edges"],
            4,
            sample_index + 2,
            stride=11,
            predicate=lambda edge: edge["road_color"] is not None,
        )
        for edge in edge_candidates:
            add(
                "edge_road_owner",
                f"Who owns the road on {edge['token']}?",
                edge["road_color_token"] or "EMPTY",
                {
                    "edge": edge["id"],
                    "edge_token": edge["token"],
                    "road_color": edge["road_color"],
                    "road_color_token": edge["road_color_token"],
                },
            )

        for port in _pick(contract["ports"], 2, sample_index + 3, stride=2):
            resource_answer = "GENERIC" if port["kind"] == "generic" else port["resource_token"]
            add(
                "port_trade_type",
                f"What trade type is shown on {port['token']}?",
                f"{resource_answer} {port['ratio']}",
                {
                    "port_id": port["id"],
                    "port_token": port["token"],
                    "kind": port["kind"],
                    "ratio": port["ratio"],
                    "resource": port["resource"],
                    "resource_token": port["resource_token"],
                },
            )
            add(
                "port_type_nodes",
                f"What trade port is {port['token']}, and which nodes touch it?",
                f"{resource_answer} {port['ratio']} {' '.join(port['attached_node_tokens'])}",
                {
                    "port_id": port["id"],
                    "port_token": port["token"],
                    "kind": port["kind"],
                    "ratio": port["ratio"],
                    "resource": port["resource"],
                    "resource_token": port["resource_token"],
                    "attached_nodes": port["attached_nodes"],
                    "attached_node_tokens": port["attached_node_tokens"],
                },
            )
            add(
                "port_occupancy",
                f"Who, if anyone, has a building on {port['token']}?",
                _port_occupancy_answer(contract, port),
                {
                    "port_id": port["id"],
                    "port_token": port["token"],
                    "attached_nodes": port["attached_nodes"],
                    "attached_node_tokens": port["attached_node_tokens"],
                    "occupied_nodes": _port_occupied_nodes_target(contract, port),
                },
            )

        count_player = _pick(contract["players"], 1, sample_index + 5, stride=3)[0]
        add(
            "color_building_counts",
            f"How many settlements and cities does {count_player['color_token']} have on the board?",
            f"{count_player['color_token']} SETTLEMENTS {count_player['settlement_count']} CITIES {count_player['city_count']}",
            {
                "color": count_player["color"],
                "color_token": count_player["color_token"],
                "settlement_count": count_player["settlement_count"],
                "city_count": count_player["city_count"],
            },
        )
        add(
            "color_road_count",
            f"How many roads does {count_player['color_token']} have on the board?",
            f"{count_player['color_token']} ROADS {count_player['road_count']}",
            {
                "color": count_player["color"],
                "color_token": count_player["color_token"],
                "road_count": count_player["road_count"],
            },
        )
        road_edges = _road_edges_for_color(contract, count_player["color"])
        road_edge_tokens = [edge["token"] for edge in road_edges]
        add(
            "color_road_locations",
            f"Where are {count_player['color_token']}'s roads? List the edge tokens.",
            "NONE" if not road_edge_tokens else " ".join(road_edge_tokens),
            {
                "color": count_player["color"],
                "color_token": count_player["color_token"],
                "road_edges": [edge["id"] for edge in road_edges],
                "road_edge_tokens": road_edge_tokens,
            },
            scoring="set_exact",
        )

        occupied_tile = _pick(contract["tiles"], 1, sample_index + 6, stride=5)[0]
        add(
            "tile_occupied_nodes",
            f"Which buildings touch {occupied_tile['token']}?",
            _tile_occupied_nodes_answer(contract, occupied_tile),
            {
                "tile_id": occupied_tile["id"],
                "tile_token": occupied_tile["token"],
                "occupied_nodes": _tile_occupied_nodes_target(contract, occupied_tile),
            },
        )

        topology_node = _pick(contract["nodes"], 1, sample_index + 4, stride=13)[0]
        add(
            "node_adjacent_tiles",
            f"Which land tiles touch {topology_node['token']}?",
            " ".join(topology_node["adjacent_tile_tokens"]),
            {
                "node_id": topology_node["id"],
                "node_token": topology_node["token"],
                "adjacent_tiles": topology_node["adjacent_tiles"],
                "adjacent_tile_tokens": topology_node["adjacent_tile_tokens"],
            },
        )

        connected_edge = _pick(contract["edges"], 1, sample_index + 7, stride=17)[0]
        disconnected_pair = _disconnected_node_pair(contract, sample_index + 8)
        if sample_index % 2 == 0:
            node_pair = connected_edge["nodes"]
            node_pair_tokens = connected_edge["node_tokens"]
            connected = True
            connecting_edge_token = connected_edge["token"]
        else:
            node_pair = list(disconnected_pair)
            node_pair_tokens = [node_token(node_id) for node_id in node_pair]
            connected = False
            connecting_edge_token = None
        add(
            "nodes_connected",
            f"Are {node_pair_tokens[0]} and {node_pair_tokens[1]} connected by a board edge?",
            "YES" if connected else "NO",
            {
                "nodes": node_pair,
                "node_tokens": node_pair_tokens,
                "connected": connected,
                "edge_token": connecting_edge_token,
            },
        )

        edge_probe = _pick(contract["edges"], 1, sample_index + 9, stride=19)[0]
        if sample_index % 2 == 0:
            edge_node_pair = edge_probe["nodes"]
            edge_node_pair_tokens = edge_probe["node_tokens"]
            edge_connected = True
        else:
            edge_node_pair = list(_disconnected_node_pair(contract, sample_index + 10))
            edge_node_pair_tokens = [node_token(node_id) for node_id in edge_node_pair]
            edge_connected = False
        add(
            "edge_connects_nodes",
            f"Does {edge_probe['token']} connect {edge_node_pair_tokens[0]} and {edge_node_pair_tokens[1]}?",
            "YES" if edge_connected else "NO",
            {
                "edge": edge_probe["id"],
                "edge_token": edge_probe["token"],
                "nodes": edge_node_pair,
                "node_tokens": edge_node_pair_tokens,
                "connected": edge_connected,
            },
        )

        return qas[:questions_per_sample]


class CatanBenchBuilder:
    """Build CatanBench records from Colonist replay files."""

    def __init__(
        self,
        *,
        replay_dir: Path = DEFAULT_REPLAY_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        question_dir: Optional[Path] = None,
        samples: int = 100,
        min_step: int = 8,
        render_images: bool = False,
        image_size: int = 512,
        quiet_replay: bool = True,
    ):
        self.replay_dir = Path(replay_dir)
        self.output_dir = Path(output_dir)
        self.question_dir = Path(question_dir) if question_dir is not None else self.output_dir
        self.samples = samples
        self.min_step = min_step
        self.render_images = render_images
        self.image_size = image_size
        self.quiet_replay = quiet_replay
        self.suite = CatanObservationSuite()

    async def build(self) -> BenchmarkBuildResult:
        self._prepare_output_dirs()
        replay_files = sorted(self.replay_dir.glob("*.json"))
        if not replay_files:
            raise FileNotFoundError(f"no replay JSON files found in {self.replay_dir}")

        per_replay_limit = max(1, math.ceil(self.samples / len(replay_files)))
        counters: Counter[str] = Counter()
        used_sources: set[Tuple[str, int]] = set()

        metadata_path = self.output_dir / "metadata.json"
        manifest_path = self.output_dir / "manifest.jsonl"
        questions_path = self.question_dir / "questions.jsonl"
        answer_key_path = self.question_dir / "answer_key.jsonl"
        qa_path = self.question_dir / "qa.jsonl"

        self._write_readme()
        self._write_question_readme()

        sample_count = 0
        qa_count = 0
        rendered_images = 0

        screenshotter = None
        if self.render_images:
            from playground.screenshot_board import FrontendScreenshotter

            screenshotter = FrontendScreenshotter(
                headless=True,
                board_width=1200,
                board_height=1200,
                crop_pct=0.08,
                vertical_offset_pct=0.0,
            )
            await screenshotter.start()

        try:
            with (
                manifest_path.open("w") as manifest_f,
                questions_path.open("w") as questions_f,
                answer_key_path.open("w") as answer_key_f,
                qa_path.open("w") as qa_f,
            ):
                for replay_file in replay_files:
                    if sample_count >= self.samples:
                        break

                    state = load_colonist_replay(replay_file, quiet=self.quiet_replay)
                    total_steps = len(state.replay_data.get("parsed_actions", []))
                    target_steps = set(_spaced_steps(total_steps, per_replay_limit, self.min_step))
                    replay_sample_count = 0

                    while state.replay_index < total_steps and sample_count < self.samples:
                        result = step_replay(state, quiet=self.quiet_replay)
                        if isinstance(result, tuple) or result.get("error"):
                            counters["replay_step_errors"] += 1
                            break
                        if result.get("finished"):
                            break

                        step_index = state.replay_index
                        source_key = (replay_file.name, step_index)
                        if step_index not in target_steps or source_key in used_sources:
                            continue

                        sample_id = f"sample_{sample_count:03d}"
                        contract_rel = Path("contracts") / f"{sample_id}.json"
                        image_rel: Optional[Path] = None

                        if screenshotter is not None:
                            image_rel = Path("images") / f"{sample_id}.png"
                            png = await screenshotter.screenshot(state.current_game, settle_ms=250)
                            _write_square_png(
                                png,
                                self.output_dir / image_rel,
                                size=self.image_size,
                            )
                            rendered_images += 1

                        source = {
                            "kind": "colonist_replay",
                            "replay_file": str(replay_file.relative_to(PROJECT_ROOT)),
                            "game_id": state.replay_data.get("game_id"),
                            "replay_step": step_index,
                            "total_replay_steps": total_steps,
                            "engine_action_count": len(state.current_game.state.actions),
                            "replay_semantic_issue_count": len(getattr(state, "replay_semantic_issues", [])),
                        }
                        sample_meta = {
                            "id": sample_id,
                            "index": sample_count,
                            "contract_path": str(contract_rel),
                            "image_path": str(image_rel) if image_rel else None,
                            "image_size": [self.image_size, self.image_size] if image_rel else None,
                        }

                        contract = self.suite.public_board_contract(
                            state.current_game,
                            sample=sample_meta,
                            source=source,
                        )
                        contract_path = self.output_dir / contract_rel
                        _write_json(contract_path, contract)

                        qas = self.suite.qa_pairs(contract)
                        for qa in qas:
                            _write_jsonl(qa_f, qa)
                            _write_jsonl(questions_f, _question_view(qa))
                            _write_jsonl(answer_key_f, _answer_view(qa))
                        qa_count += len(qas)

                        manifest_record = {
                            "sample_id": sample_id,
                            "contract_path": str(contract_rel),
                            "image_path": str(image_rel) if image_rel else None,
                            "question_count": len(qas),
                            "source": source,
                        }
                        _write_jsonl(manifest_f, manifest_record)

                        used_sources.add(source_key)
                        counters[f"replay:{replay_file.name}"] += 1
                        sample_count += 1
                        replay_sample_count += 1

                        if replay_sample_count >= per_replay_limit:
                            break

            metadata = {
                "name": "CatanBench-100",
                "schema": "catan_benchmark/v0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sample_count": sample_count,
                "qa_count": qa_count,
                "render_images": self.render_images,
                "rendered_images": rendered_images,
                "image_size": [self.image_size, self.image_size],
                "replay_dir": _project_relative_path(self.replay_dir),
                "source_replay_count": len(replay_files),
                "sampling": {
                    "target_samples": self.samples,
                    "per_replay_limit": per_replay_limit,
                    "min_step": self.min_step,
                },
                "counts": dict(counters),
                "files": {
                    "manifest": manifest_path.name,
                    "question_dir": _project_relative_path(self.question_dir),
                    "questions": questions_path.name,
                    "answer_key": answer_key_path.name,
                    "qa": qa_path.name,
                    "contracts_dir": "contracts",
                    "images_dir": "images" if self.render_images else None,
                },
            }
            _write_json(metadata_path, metadata)
        finally:
            if screenshotter is not None:
                await screenshotter.stop()

        if sample_count < self.samples:
            raise RuntimeError(
                f"only built {sample_count}/{self.samples} samples from {len(replay_files)} replays"
            )

        return BenchmarkBuildResult(
            output_dir=self.output_dir,
            sample_count=sample_count,
            qa_count=qa_count,
            rendered_images=rendered_images,
            replay_count=len(replay_files),
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            questions_path=questions_path,
            answer_key_path=answer_key_path,
        )

    def _prepare_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.question_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "contracts").mkdir(parents=True, exist_ok=True)
        if self.render_images:
            (self.output_dir / "images").mkdir(parents=True, exist_ok=True)

    def _write_readme(self) -> None:
        readme = self.output_dir / "README.md"
        readme.write_text(
            "\n".join(
                [
                    "# CatanBench-100",
                    "",
                    "Small engine-oracle benchmark for Catan board recognition.",
                    "",
                    "Files:",
                    "- `manifest.jsonl`: one row per board sample.",
                    "- `contracts/`: full public board contracts derived from the engine.",
                    "- `images/`: optional 512x512 board renders when built with `--render-images`.",
                    "",
                    "Question-suite files live in `questions/`.",
                    "That directory contains `questions.jsonl`, `answer_key.jsonl`, and `qa.jsonl`.",
                    "",
                    "The contract intentionally excludes hidden hands and hidden dev cards.",
                    "It does include public board state, visible points, played knights,",
                    "current prompt, ports, robber location, and Longest Road/Largest Army.",
                    "",
                ]
            )
        )

    def _write_question_readme(self) -> None:
        readme = self.question_dir / "README.md"
        readme.write_text(
            "\n".join(
                [
                    "# CatanBench-100 Questions",
                    "",
                    "Engine-scored public-board QA rows for CatanBench-100.",
                    "",
                    "Files:",
                    "- `questions.jsonl`: promptable questions without answers.",
                    "- `answer_key.jsonl`: deterministic answer targets for scoring.",
                    "- `qa.jsonl`: questions plus answers and target metadata for local analysis.",
                    "",
                    "The referenced contracts and board images live in",
                    "`data_pipeline/catanbench/datasets/catanbench_100/`.",
                    "",
                ]
            )
        )


def build_catanbench_sync(**kwargs: Any) -> BenchmarkBuildResult:
    """Synchronous wrapper for scripts/tests."""

    return asyncio.run(CatanBenchBuilder(**kwargs).build())


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
            initial_state = (
                raw_data["data"].get("eventHistory", {}).get("initialState")
                or raw_data["data"].get("initialState")
            )
        elif "eventHistory" in raw_data:
            initial_state = raw_data["eventHistory"].get("initialState") or raw_data.get("initialState")

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
                    "color": COLONIST_COLOR_NAMES.get(color_id, f"color_{color_id}"),
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

    state.current_game = Game(players, catan_map=catan_map, shuffle_players=False)
    state.current_players = players
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


def step_replay(state: ServerState, *, quiet: bool = True) -> JsonDict:
    with _quiet_context(quiet):
        result = replay_step_logic(state, lambda: None)
    return result


def _tile_coordinates(catan_map: CatanMap) -> Dict[int, Tuple[int, int, int]]:
    return {
        tile.id: coordinate
        for coordinate, tile in catan_map.land_tiles.items()
        if isinstance(tile, LandTile)
    }


def _port_coordinates(catan_map: CatanMap) -> Dict[int, Tuple[int, int, int]]:
    return {
        port.id: coordinate
        for coordinate, port in catan_map.tiles.items()
        if isinstance(port, Port)
    }


def _port_nodes_by_id(catan_map: CatanMap) -> Dict[int, List[int]]:
    port_nodes = {}
    for port_id, port in sorted(catan_map.ports_by_id.items()):
        node_refs = PORT_DIRECTION_TO_NODEREFS[port.direction]
        port_nodes[port_id] = [port.nodes[node_ref] for node_ref in node_refs]
    return port_nodes


def _node_ports(catan_map: CatanMap) -> Dict[int, List[int]]:
    node_ports: Dict[int, List[int]] = {node_id: [] for node_id in range(NUM_NODES)}
    for port_id, nodes in _port_nodes_by_id(catan_map).items():
        for node_id in nodes:
            node_ports[node_id].append(port_id)
    return node_ports


def _playable_edges(catan_map: CatanMap) -> List[EdgeId]:
    edges = sorted(canonical_edge(edge) for edge in get_edges(catan_map.land_nodes))
    if len(edges) != NUM_EDGES:
        fallback_edges = base_edges()
        if len(fallback_edges) != NUM_EDGES:
            raise RuntimeError(f"expected {NUM_EDGES} playable edges, found {len(edges)}")
        return fallback_edges
    return edges


def _owned_road_count(roads: Dict[EdgeId, Color], color: Color) -> int:
    return len({canonical_edge(edge) for edge, owner in roads.items() if owner == color})


def _pick(items: Sequence[JsonDict], count: int, seed: int, *, stride: int) -> List[JsonDict]:
    if not items:
        return []
    result = []
    seen = set()
    index = (seed * stride) % len(items)
    attempts = 0
    while len(result) < min(count, len(items)) and attempts < len(items) * 2:
        if index not in seen:
            result.append(items[index])
            seen.add(index)
        index = (index + stride) % len(items)
        attempts += 1
    return result


def _positive_first(
    items: Sequence[JsonDict],
    count: int,
    seed: int,
    *,
    stride: int,
    predicate: Any,
) -> List[JsonDict]:
    positives = [item for item in items if predicate(item)]
    negatives = [item for item in items if not predicate(item)]
    selected = _pick(positives, min(count, len(positives)), seed, stride=stride)
    if len(selected) < count:
        selected.extend(_pick(negatives, count - len(selected), seed + 17, stride=stride))
    return selected[:count]


def _resource_number_answer(tile: JsonDict) -> str:
    number_answer = "NO_NUMBER" if tile["number"] is None else str(tile["number"])
    return f"{tile['resource_token']} {number_answer}"


def _find_by_id(items: Sequence[JsonDict], item_id: int) -> Optional[JsonDict]:
    for item in items:
        if item.get("id") == item_id:
            return item
    return None


def _tile_occupied_nodes_target(contract: JsonDict, tile: JsonDict) -> List[JsonDict]:
    nodes_by_id = {node["id"]: node for node in contract["nodes"]}
    occupied = []
    for node_id in tile["nodes"]:
        node = nodes_by_id[node_id]
        if node["building"] is None:
            continue
        occupied.append(
            {
                "node_id": node["id"],
                "node_token": node["token"],
                "building": node["building"],
                "building_token": node["building_token"],
                "color": node["color"],
                "color_token": node["color_token"],
            }
        )
    return occupied


def _tile_occupied_nodes_answer(contract: JsonDict, tile: JsonDict) -> str:
    occupied = _tile_occupied_nodes_target(contract, tile)
    if not occupied:
        return "NONE"
    return " ".join(
        f"{node['node_token']} {node['color_token']} {node['building_token']}"
        for node in occupied
    )


def _port_occupied_nodes_target(contract: JsonDict, port: JsonDict) -> List[JsonDict]:
    nodes_by_id = {node["id"]: node for node in contract["nodes"]}
    occupied = []
    for node_id in port["attached_nodes"]:
        node = nodes_by_id[node_id]
        if node["building"] is None:
            continue
        occupied.append(
            {
                "node_id": node["id"],
                "node_token": node["token"],
                "building": node["building"],
                "building_token": node["building_token"],
                "color": node["color"],
                "color_token": node["color_token"],
            }
        )
    return occupied


def _port_occupancy_answer(contract: JsonDict, port: JsonDict) -> str:
    occupied = _port_occupied_nodes_target(contract, port)
    if not occupied:
        return "NONE"
    return " ".join(
        f"{node['color_token']} {node['building_token']} {node['node_token']}"
        for node in occupied
    )


def _road_edges_for_color(contract: JsonDict, color: str) -> List[JsonDict]:
    return sorted(
        [edge for edge in contract["edges"] if edge["road_color"] == color],
        key=lambda edge: edge["token"],
    )


def _disconnected_node_pair(contract: JsonDict, seed: int) -> Tuple[int, int]:
    node_ids = [node["id"] for node in contract["nodes"]]
    connected_edges = {tuple(edge["id"]) for edge in contract["edges"]}
    start = (seed * 13) % len(node_ids)
    for offset in range(len(node_ids) * len(node_ids)):
        a = node_ids[(start + offset) % len(node_ids)]
        b = node_ids[(start + 5 + offset * 7) % len(node_ids)]
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        if pair not in connected_edges:
            return pair
    raise ValueError("could not find disconnected node pair")


def _spaced_steps(total_steps: int, count: int, min_step: int) -> List[int]:
    if total_steps <= min_step or count <= 0:
        return []
    start = min_step
    end = max(start, total_steps - 1)
    if count == 1:
        return [round((start + end) / 2)]
    steps = [round(start + i * (end - start) / (count - 1)) for i in range(count)]
    return sorted(set(steps))


def _question_view(qa: JsonDict) -> JsonDict:
    return {
        "id": qa["id"],
        "sample_id": qa["sample_id"],
        "image_path": qa["image_path"],
        "contract_path": qa["contract_path"],
        "category": qa["category"],
        "question": qa["question"],
    }


def _answer_view(qa: JsonDict) -> JsonDict:
    return {
        "id": qa["id"],
        "sample_id": qa["sample_id"],
        "category": qa["category"],
        "answer": qa["answer"],
        "target": qa["target"],
        "scoring": qa["scoring"],
    }


def _write_square_png(png: bytes, path: Path, *, size: int) -> None:
    image = Image.open(io.BytesIO(png)).convert("RGB")
    image = ImageOps.contain(image, (size, size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), color=(13, 111, 165))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def _write_json(path: Path, value: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(handle: Any, value: JsonDict) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")


@contextlib.contextmanager
def _quiet_context(enabled: bool) -> Iterable[None]:
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield
