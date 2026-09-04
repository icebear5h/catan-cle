"""Build post-atlas node visual-grounding data.

The purpose of this dataset is not to imitate real games or teach the atlas from
scratch. It assumes the fixed Catan atlas already exists, then trains visual
readout around stable atlas handles:

    stable token: <N41>
    transient visual state: EMPTY / <RED> <SETTLEMENT> / <BLUE> <CITY> / ...
    varied context: shuffled tile resources, numbers, ports, robber, and roads

This guards against the bad shortcut where a model learns that a node token
permanently "has" a piece. Each atlas node appears under many contradictory
transient states.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.catan_board_bench.annotations import annotation_payload_for_contract
from evals.catan_board_bench.tokens import (
    atlas_metadata,
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
from cle.game_engine.models.enums import CITY, SETTLEMENT
from cle.game_engine.models.player import Color
from sft.paths import GENERATED_SFT_ROOT


DEFAULT_COLORS = ("RED", "BLUE", "ORANGE", "WHITE", "BLACK")
RESOURCE_DECK = (
    ["WOOD"] * 4
    + ["BRICK"] * 3
    + ["SHEEP"] * 4
    + ["WHEAT"] * 4
    + ["ORE"] * 3
    + [None]
)
NUMBER_DECK = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
PORT_DECK = [None, None, None, None, "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
PROMPT_PREFIX = "Answer exactly using Catan atlas tokens. Use compact JSON when requested. Do not explain."
DATASET_NAME = "post_atlas_node_visual_grounding"
DATASET_SCHEMA = "catan_post_atlas_node_visual_grounding/v1"
CURRICULUM_STAGE = "phase_1_post_atlas_visual_grounding"
REQUIRES_STAGE = "phase_0_text_atlas_topology"
DATASET_ROLE = "bind_stable_node_tokens_to_transient_visual_facts"
CATEGORY_GROUPS = {
    "board_atlas_bboxes": "full_board_atlas_bbox_map",
    "node_bbox": "atlas_coordinate_grounding",
    "node_occupancy": "transient_node_state_readout",
    "node_adjacent_tile_resource_numbers": "local_tile_readout",
    "node_incident_road_owners": "local_edge_readout",
    "node_local_state_json": "local_state_composition",
}


def category_group(category: str) -> str:
    return CATEGORY_GROUPS.get(category, "uncategorized")


def build_indices(atlas: dict[str, Any]) -> dict[str, Any]:
    tiles = {tile["id"]: tile for tile in atlas["tiles"]}
    nodes = {node["id"]: node for node in atlas["nodes"]}
    edges = {tuple(edge["id"]): edge for edge in atlas["edges"]}
    ports = {port["id"]: port for port in atlas["ports"]}

    node_tiles: dict[int, set[int]] = defaultdict(set)
    node_edges: dict[int, set[tuple[int, int]]] = defaultdict(set)
    edge_tiles: dict[tuple[int, int], set[int]] = defaultdict(set)
    node_ports: dict[int, set[int]] = defaultdict(set)
    node_neighbors: dict[int, set[int]] = defaultdict(set)

    for tile in tiles.values():
        for node_id in tile["nodes"].values():
            node_tiles[node_id].add(tile["id"])
        for edge in tile["edges"].values():
            edge_tuple = tuple(canonical_edge(tuple(edge)))
            edge_tiles[edge_tuple].add(tile["id"])
            a, b = edge_tuple
            node_edges[a].add(edge_tuple)
            node_edges[b].add(edge_tuple)
            node_neighbors[a].add(b)
            node_neighbors[b].add(a)

    for port in ports.values():
        for node_id in port["attached_nodes"]:
            node_ports[node_id].add(port["id"])

    return {
        "tiles": tiles,
        "nodes": nodes,
        "edges": edges,
        "ports": ports,
        "node_tiles": node_tiles,
        "node_edges": node_edges,
        "edge_tiles": edge_tiles,
        "node_ports": node_ports,
        "node_neighbors": node_neighbors,
    }


def validate_colors(colors: list[str]) -> list[str]:
    valid = {color.value for color in Color}
    unknown = sorted(set(colors) - valid)
    if unknown:
        raise ValueError(f"unknown colors: {unknown}; valid={sorted(valid)}")
    return colors


def occupancy_cases(colors: list[str]) -> list[dict[str, str | None]]:
    cases: list[dict[str, str | None]] = [{"name": "empty", "color": None, "building": None}]
    for color in colors:
        cases.append({"name": f"{color.lower()}_settlement", "color": color, "building": SETTLEMENT})
        cases.append({"name": f"{color.lower()}_city", "color": color, "building": CITY})
    return cases


def build_contract(
    *,
    atlas: dict[str, Any],
    indices: dict[str, Any],
    sample_id: str,
    sample_index: int,
    target_node_id: int,
    occupancy: dict[str, str | None],
    colors: list[str],
    seed: int,
    image_size: int,
    distractor_buildings: int,
    distractor_roads: int,
    assume_rendered_images: bool,
) -> dict[str, Any]:
    rng = random.Random(seed)
    resources = list(RESOURCE_DECK)
    numbers = list(NUMBER_DECK)
    port_resources = list(PORT_DECK)
    rng.shuffle(resources)
    rng.shuffle(numbers)
    rng.shuffle(port_resources)

    tile_resource_by_id: dict[int, str | None] = {}
    tile_number_by_id: dict[int, int | None] = {}
    for tile_id in sorted(indices["tiles"]):
        resource = resources.pop()
        tile_resource_by_id[tile_id] = resource
        tile_number_by_id[tile_id] = None if resource is None else numbers.pop()

    desert_tile_ids = [tile_id for tile_id, resource in tile_resource_by_id.items() if resource is None]
    adjacent_tile_ids = sorted(indices["node_tiles"].get(target_node_id, set()))
    if adjacent_tile_ids and sample_index % 3 == 0:
        robber_tile_id = adjacent_tile_ids[sample_index % len(adjacent_tile_ids)]
    elif desert_tile_ids and sample_index % 3 == 1:
        robber_tile_id = desert_tile_ids[0]
    else:
        robber_tile_id = rng.choice(sorted(indices["tiles"]))

    target_buildings: dict[int, tuple[str, str]] = {}
    if occupancy["color"] and occupancy["building"]:
        target_buildings[target_node_id] = (str(occupancy["color"]), str(occupancy["building"]))

    buildings = {
        **target_buildings,
        **choose_distractor_buildings(
            indices=indices,
            rng=rng,
            target_node_id=target_node_id,
            colors=colors,
            count=distractor_buildings,
            existing=target_buildings,
        ),
    }
    roads = choose_roads(
        indices=indices,
        rng=rng,
        target_node_id=target_node_id,
        target_color=str(occupancy["color"]) if occupancy["color"] else None,
        colors=colors,
        variant=sample_index,
        distractor_count=distractor_roads,
    )

    contract_rel = Path("contracts") / f"{sample_id}.json"
    image_rel = Path("images") / f"{sample_id}.png" if assume_rendered_images else None
    sample = {
        "id": sample_id,
        "index": sample_index,
        "contract_path": str(contract_rel),
        "image_path": str(image_rel) if image_rel else None,
        "image_size": [image_size, image_size] if image_rel else None,
        "target_node": node_token(target_node_id),
    }
    source = {
        "kind": DATASET_NAME,
        "generator": "sft/scripts/build_node_factor_dataset.py",
        "seed": seed,
        "render_status": "pending_frontend_render" if image_rel else "contract_only",
        "leakage_note": "Synthetic atlas coverage; not sampled from CatanBoardBench-100 game IDs.",
        "curriculum_stage": CURRICULUM_STAGE,
        "requires_stage": REQUIRES_STAGE,
        "dataset_role": DATASET_ROLE,
    }

    tiles = []
    for tile_id, tile in sorted(indices["tiles"].items()):
        resource = tile_resource_by_id[tile_id]
        number = tile_number_by_id[tile_id]
        edges = sorted(tuple(canonical_edge(tuple(edge))) for edge in tile["edges"].values())
        nodes = sorted(tile["nodes"].values())
        tiles.append(
            {
                "id": tile_id,
                "token": tile_token(tile_id),
                "coord": tile["coord"],
                "resource": resource,
                "resource_token": resource_token(resource),
                "number": number,
                "has_robber": tile_id == robber_tile_id,
                "nodes": nodes,
                "node_tokens": [node_token(node_id) for node_id in nodes],
                "edges": [list(edge) for edge in edges],
                "edge_tokens": [edge_token(edge) for edge in edges],
            }
        )

    nodes = []
    for node_id in sorted(indices["nodes"]):
        building = buildings.get(node_id)
        color = building[0] if building else None
        building_type = building[1] if building else None
        adjacent_tiles = sorted(indices["node_tiles"].get(node_id, set()))
        adjacent_edges = sorted(indices["node_edges"].get(node_id, set()))
        attached_ports = sorted(indices["node_ports"].get(node_id, set()))
        nodes.append(
            {
                "id": node_id,
                "token": node_token(node_id),
                "building": building_type,
                "building_token": building_token(building_type) if building_type else None,
                "color": color,
                "color_token": color_token(color) if color else None,
                "adjacent_tiles": adjacent_tiles,
                "adjacent_tile_tokens": [tile_token(tile_id) for tile_id in adjacent_tiles],
                "adjacent_edges": [list(edge) for edge in adjacent_edges],
                "adjacent_edge_tokens": [edge_token(edge) for edge in adjacent_edges],
                "port_ids": attached_ports,
                "port_tokens": [port_token(port_id) for port_id in attached_ports],
            }
        )

    edges = []
    for edge in sorted(indices["edges"]):
        road_color = roads.get(edge)
        edges.append(
            {
                "id": list(edge),
                "token": edge_token(edge),
                "nodes": list(edge),
                "node_tokens": [node_token(node_id) for node_id in edge],
                "road_color": road_color,
                "road_color_token": color_token(road_color) if road_color else None,
            }
        )

    ports = []
    for port_id, port in sorted(indices["ports"].items()):
        resource = port_resources.pop()
        ports.append(
            {
                "id": port_id,
                "token": port_token(port_id),
                "coord": port["coord"],
                "direction": port["direction"],
                "kind": "generic" if resource is None else "resource",
                "ratio": "3:1" if resource is None else "2:1",
                "resource": resource,
                "resource_token": resource_token(resource) if resource is not None else None,
                "attached_nodes": sorted(port["attached_nodes"]),
                "attached_node_tokens": [node_token(node_id) for node_id in sorted(port["attached_nodes"])],
            }
        )

    players = player_summaries(colors=colors, buildings=buildings, roads=roads)
    robber_tile = indices["tiles"][robber_tile_id]
    return {
        "schema": "catan_public_board_contract/v0",
        "sample": sample,
        "source": source,
        "current": {
            "current_color": colors[sample_index % len(colors)],
            "current_color_token": color_token(colors[sample_index % len(colors)]),
            "current_prompt": "SYNTHETIC_NODE_FACTOR",
            "turn_index": sample_index,
            "player_index": sample_index % len(colors),
            "num_completed_turns": sample_index,
            "is_initial_build_phase": False,
        },
        "players": players,
        "tiles": tiles,
        "nodes": nodes,
        "edges": edges,
        "ports": ports,
        "robber": {
            "object_token": object_token("ROBBER"),
            "tile_id": robber_tile_id,
            "tile_token": tile_token(robber_tile_id),
            "coord": robber_tile["coord"],
        },
        "achievements": {
            "longest_road": {"holder": None, "holder_token": None, "length": 0},
            "largest_army": {"holder": None, "holder_token": None, "size": 0},
        },
    }


def choose_distractor_buildings(
    *,
    indices: dict[str, Any],
    rng: random.Random,
    target_node_id: int,
    colors: list[str],
    count: int,
    existing: dict[int, tuple[str, str]],
) -> dict[int, tuple[str, str]]:
    buildings: dict[int, tuple[str, str]] = {}
    occupied = set(existing)
    forbidden = {target_node_id, *occupied}
    for node_id in occupied:
        forbidden.update(indices["node_neighbors"].get(node_id, set()))

    candidates = [node_id for node_id in sorted(indices["nodes"]) if node_id not in forbidden]
    attempts = 0
    while candidates and len(buildings) < count and attempts < 500:
        attempts += 1
        node_id = rng.choice(candidates)
        color = colors[(node_id + attempts) % len(colors)]
        building = CITY if (node_id + attempts) % 5 == 0 else SETTLEMENT
        buildings[node_id] = (color, building)

        forbidden.add(node_id)
        forbidden.update(indices["node_neighbors"].get(node_id, set()))
        candidates = [candidate for candidate in candidates if candidate not in forbidden]

    return buildings


def choose_roads(
    *,
    indices: dict[str, Any],
    rng: random.Random,
    target_node_id: int,
    target_color: str | None,
    colors: list[str],
    variant: int,
    distractor_count: int,
) -> dict[tuple[int, int], str]:
    roads: dict[tuple[int, int], str] = {}
    incident_edges = sorted(indices["node_edges"].get(target_node_id, set()))
    if incident_edges:
        mode = variant % 4
        if target_color and mode in {1, 2}:
            roads[incident_edges[0]] = target_color
            if mode == 2 and len(incident_edges) > 1:
                roads[incident_edges[1]] = target_color
        elif mode in {1, 3}:
            roads[incident_edges[-1]] = colors[(variant + 1) % len(colors)]

    available_edges = [edge for edge in sorted(indices["edges"]) if edge not in roads]
    rng.shuffle(available_edges)
    for edge in available_edges[:distractor_count]:
        roads[edge] = colors[(edge[0] + edge[1] + variant) % len(colors)]
    return roads


def player_summaries(
    *,
    colors: list[str],
    buildings: dict[int, tuple[str, str]],
    roads: dict[tuple[int, int], str],
) -> list[dict[str, Any]]:
    summaries = []
    building_counts: Counter[tuple[str, str]] = Counter(buildings.values())
    road_counts: Counter[str] = Counter(roads.values())
    for color in colors:
        settlement_count = building_counts[(color, SETTLEMENT)]
        city_count = building_counts[(color, CITY)]
        summaries.append(
            {
                "color": color,
                "color_token": color_token(color),
                "visible_victory_points": settlement_count + 2 * city_count,
                "settlement_count": settlement_count,
                "city_count": city_count,
                "road_count": road_counts[color],
                "longest_road_length": 0,
                "played_knights": 0,
            }
        )
    return summaries


def build_qas(contract: dict[str, Any], target_node_id: int, annotations: dict[str, Any]) -> list[dict[str, Any]]:
    sample = contract["sample"]
    sample_id = sample["id"]
    node = contract["nodes"][target_node_id]
    target_annotation = find_annotation(annotations, kind="node", token=node["token"])
    if target_annotation is None:
        raise RuntimeError(f"missing node annotation for {node['token']}")

    qas: list[dict[str, Any]] = []

    def add(category: str, question: str, answer: str, target: dict[str, Any], scoring: str = "exact") -> None:
        qas.append(
            {
                "id": f"{sample_id}_q{len(qas):02d}_{category}",
                "sample_id": sample_id,
                "image_path": sample.get("image_path"),
                "contract_path": sample.get("contract_path"),
                "category": category,
                "category_group": category_group(category),
                "curriculum_stage": CURRICULUM_STAGE,
                "requires_stage": REQUIRES_STAGE,
                "question": question,
                "answer": answer,
                "target": target,
                "scoring": scoring,
            }
        )

    occupancy = occupancy_answer(node)
    bbox = target_annotation["bbox_2d"]
    add(
        "node_bbox",
        f"Locate {node['token']} and output the bbox coordinates in JSON format.",
        compact_json({"bbox_2d": bbox}),
        {
            "node_id": node["id"],
            "node_token": node["token"],
            "bbox_2d": bbox,
            "center": target_annotation["center"],
        },
        scoring="json_exact",
    )
    board_bboxes = board_atlas_bbox_payload(annotations)
    add(
        "board_atlas_bboxes",
        "Return the Catan atlas bboxes for every visible tile, tile number, node, edge, and port as compact JSON.",
        compact_json(board_bboxes),
        board_bboxes,
        scoring="json_exact",
    )
    add(
        "node_occupancy",
        f"What building, if any, is on {node['token']}?",
        occupancy,
        {
            "node_id": node["id"],
            "node_token": node["token"],
            "building": node["building"],
            "building_token": node["building_token"],
            "color": node["color"],
            "color_token": node["color_token"],
        },
    )
    add(
        "node_adjacent_tile_resource_numbers",
        f"What resource and dice number are on the land tiles touching {node['token']}?",
        adjacent_tile_answer(contract, node),
        {
            "node_id": node["id"],
            "node_token": node["token"],
            "adjacent_tiles": local_tile_payloads(contract, node),
        },
    )
    add(
        "node_incident_road_owners",
        f"Who owns the roads touching {node['token']}?",
        incident_road_answer(contract, node),
        {
            "node_id": node["id"],
            "node_token": node["token"],
            "incident_edges": local_edge_payloads(contract, node),
        },
    )
    local_state = local_state_payload(contract, node, bbox=bbox)
    add(
        "node_local_state_json",
        f"Return the visible local state around {node['token']} as compact JSON.",
        compact_json(local_state),
        local_state,
        scoring="json_exact",
    )
    return qas


def board_atlas_bbox_payload(annotations: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "tiles": [],
        "tile_numbers": [],
        "nodes": [],
        "edges": [],
        "ports": [],
    }
    kind_to_group = {
        "tile": "tiles",
        "tile_number": "tile_numbers",
        "node": "nodes",
        "edge": "edges",
        "port": "ports",
    }
    for annotation in sorted(
        annotations["annotations"],
        key=lambda item: (item["kind"], item.get("label") or item["token"]),
    ):
        group = kind_to_group.get(annotation["kind"])
        if group is None:
            continue
        groups[group].append(
            {
                "label": annotation.get("label") or annotation["token"],
                "bbox_2d": annotation["bbox_2d"],
            }
        )
    return groups


def find_annotation(
    annotations: dict[str, Any],
    *,
    kind: str,
    token: str,
) -> dict[str, Any] | None:
    for annotation in annotations["annotations"]:
        if annotation["kind"] == kind and annotation["token"] == token:
            return annotation
    return None


def occupancy_answer(node: dict[str, Any]) -> str:
    if node["building"] is None:
        return "EMPTY"
    return f"{node['color_token']} {node['building_token']}"


def adjacent_tile_answer(contract: dict[str, Any], node: dict[str, Any]) -> str:
    return " ".join(tile_phrase(tile) for tile in local_tiles(contract, node))


def incident_road_answer(contract: dict[str, Any], node: dict[str, Any]) -> str:
    payloads = local_edge_payloads(contract, node)
    return " ".join(f"{edge['edge']}:{edge['road']}" for edge in payloads) or "NONE"


def local_state_payload(contract: dict[str, Any], node: dict[str, Any], *, bbox: list[int]) -> dict[str, Any]:
    return {
        "node": node["token"],
        "bbox_2d": bbox,
        "occupancy": occupancy_answer(node),
        "adjacent_tiles": local_tile_payloads(contract, node),
        "incident_edges": local_edge_payloads(contract, node),
        "ports": local_port_payloads(contract, node),
    }


def local_tiles(contract: dict[str, Any], node: dict[str, Any]) -> list[dict[str, Any]]:
    tiles_by_id = {tile["id"]: tile for tile in contract["tiles"]}
    return [tiles_by_id[tile_id] for tile_id in node["adjacent_tiles"]]


def local_tile_payloads(contract: dict[str, Any], node: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "tile": tile["token"],
            "resource": tile["resource_token"],
            "number": tile["number"],
            "robber": tile["has_robber"],
        }
        for tile in local_tiles(contract, node)
    ]


def local_edge_payloads(contract: dict[str, Any], node: dict[str, Any]) -> list[dict[str, str]]:
    edges_by_token = {edge["token"]: edge for edge in contract["edges"]}
    payloads = []
    for edge_token_value in node["adjacent_edge_tokens"]:
        edge = edges_by_token[edge_token_value]
        payloads.append({"edge": edge["token"], "road": edge["road_color_token"] or "EMPTY"})
    return payloads


def local_port_payloads(contract: dict[str, Any], node: dict[str, Any]) -> list[dict[str, str | None]]:
    ports_by_id = {port["id"]: port for port in contract["ports"]}
    payloads = []
    for port_id in node["port_ids"]:
        port = ports_by_id[port_id]
        payloads.append(
            {
                "port": port["token"],
                "trade": "GENERIC" if port["kind"] == "generic" else port["resource_token"],
                "ratio": port["ratio"],
            }
        )
    return payloads


def tile_phrase(tile: dict[str, Any]) -> str:
    if tile["resource"] is None:
        return f"{tile['token']} <DESERT>"
    return f"{tile['token']} {tile['resource_token']} {tile['number']}"


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def question_view(qa: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": qa["id"],
        "sample_id": qa["sample_id"],
        "image_path": qa["image_path"],
        "contract_path": qa["contract_path"],
        "category": qa["category"],
        "category_group": qa["category_group"],
        "curriculum_stage": qa["curriculum_stage"],
        "requires_stage": qa["requires_stage"],
        "question": qa["question"],
    }


def answer_view(qa: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": qa["id"],
        "sample_id": qa["sample_id"],
        "category": qa["category"],
        "category_group": qa["category_group"],
        "curriculum_stage": qa["curriculum_stage"],
        "requires_stage": qa["requires_stage"],
        "answer": qa["answer"],
        "target": qa["target"],
        "scoring": qa["scoring"],
    }


def message_view(qa: dict[str, Any]) -> dict[str, Any]:
    content: list[dict[str, str]] = []
    if qa["image_path"]:
        content.append({"type": "image"})
    content.append({"type": "text", "text": f"{PROMPT_PREFIX}\n\nQuestion: {qa['question']}"})
    return {
        "id": qa["id"],
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": qa["answer"]}]},
        ],
        "metadata": {
            "phase": CURRICULUM_STAGE,
            "requires_stage": REQUIRES_STAGE,
            "dataset_role": DATASET_ROLE,
            "category": qa["category"],
            "category_group": qa["category_group"],
            "sample_id": qa["sample_id"],
            "contract_path": qa["contract_path"],
            "image_path": qa["image_path"],
            "target": qa["target"],
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")


def write_readme(output_dir: Path, *, sample_count: int, qa_count: int, assume_rendered_images: bool) -> None:
    render_note = (
        "QA rows contain image paths under `images/`, but this script does not render those images."
        if assume_rendered_images
        else "This is contract-only output. QA rows have `image_path: null` until a frontend render step fills images."
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Post-Atlas Node Visual Grounding Dataset",
                "",
                "Controlled post-atlas visual QA for node grounding. This dataset assumes",
                "Phase 0 text atlas topology has already established stable tokens like",
                "`<N11>`, then asks the model to bind those stable node handles to visible",
                "coordinates and transient local board facts.",
                "",
                render_note,
                "",
                "Curriculum:",
                f"- Stage: `{CURRICULUM_STAGE}`",
                f"- Requires: `{REQUIRES_STAGE}`",
                f"- Role: `{DATASET_ROLE}`",
                "",
                "Category groups:",
                "- `atlas_coordinate_grounding`: locate the stable node handle.",
                "- `full_board_atlas_bbox_map`: return all tile/node/edge/port bboxes.",
                "- `transient_node_state_readout`: read occupancy at that handle.",
                "- `local_tile_readout`: read neighboring tile resource/number facts.",
                "- `local_edge_readout`: read road ownership around the node.",
                "- `local_state_composition`: compose the local node state as JSON.",
                "",
                "Files:",
                "- `contracts/`: public board contracts.",
                "- `annotations.jsonl`: frontend-aligned tile/node/edge/port bboxes for each contract.",
                "- `manifest.jsonl`: one row per synthetic sample.",
                "- `questions/qa.jsonl`: QA rows with answers and targets.",
                "- `questions/questions.jsonl`: promptable questions without answers.",
                "- `questions/answer_key.jsonl`: deterministic answer targets.",
                "- `messages.jsonl`: chat-style rows for text-only or pending-image smoke tests.",
                "",
                f"Samples: {sample_count}",
                f"QA rows: {qa_count}",
                "",
            ]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GENERATED_SFT_ROOT / "node_factors",
        help="Directory for generated contracts, annotations, and QA rows.",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        default=list(DEFAULT_COLORS),
        help="Color names to permute over. Defaults to common 4p colors plus BLACK.",
    )
    parser.add_argument(
        "--all-colors",
        action="store_true",
        help="Use every engine Color enum value.",
    )
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument(
        "--variants-per-case",
        type=int,
        default=1,
        help="Number of independent board shuffles for each node/occupancy pair.",
    )
    parser.add_argument("--distractor-buildings", type=int, default=2)
    parser.add_argument("--distractor-roads", type=int, default=4)
    parser.add_argument(
        "--assume-rendered-images",
        action="store_true",
        help="Populate image_path as images/<sample>.png for a later render step.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    colors = [color.value for color in Color] if args.all_colors else validate_colors(list(args.colors))

    atlas = atlas_metadata()
    indices = build_indices(atlas)
    cases = occupancy_cases(colors)
    output_dir = args.output_dir
    questions_dir = output_dir / "questions"
    contracts_dir = output_dir / "contracts"
    output_dir.mkdir(parents=True, exist_ok=True)
    questions_dir.mkdir(parents=True, exist_ok=True)
    contracts_dir.mkdir(parents=True, exist_ok=True)
    if args.assume_rendered_images:
        (output_dir / "images").mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    annotations_path = output_dir / "annotations.jsonl"
    qa_path = questions_dir / "qa.jsonl"
    questions_path = questions_dir / "questions.jsonl"
    answer_key_path = questions_dir / "answer_key.jsonl"
    messages_path = output_dir / "messages.jsonl"

    sample_count = 0
    qa_count = 0
    category_counts: Counter[str] = Counter()
    with (
        manifest_path.open("w") as manifest_f,
        annotations_path.open("w") as annotations_f,
        qa_path.open("w") as qa_f,
        questions_path.open("w") as questions_f,
        answer_key_path.open("w") as answer_key_f,
        messages_path.open("w") as messages_f,
    ):
        for node_id in sorted(indices["nodes"]):
            for case in cases:
                if args.limit_samples is not None and sample_count >= args.limit_samples:
                    break
                for variant_index in range(args.variants_per_case):
                    if args.limit_samples is not None and sample_count >= args.limit_samples:
                        break
                    sample_id = f"node_factor_n{node_id:02d}_{case['name']}_v{variant_index:02d}"
                    contract = build_contract(
                        atlas=atlas,
                        indices=indices,
                        sample_id=sample_id,
                        sample_index=sample_count,
                        target_node_id=node_id,
                        occupancy=case,
                        colors=colors,
                        seed=args.seed + sample_count * 1009 + node_id + variant_index * 9173,
                        image_size=args.image_size,
                        distractor_buildings=args.distractor_buildings,
                        distractor_roads=args.distractor_roads,
                        assume_rendered_images=args.assume_rendered_images,
                    )
                    annotations = annotation_payload_for_contract(contract, image_size=args.image_size)
                    qas = build_qas(contract, node_id, annotations)

                    write_json(contracts_dir / f"{sample_id}.json", contract)
                    write_jsonl(annotations_f, annotations)
                    for qa in qas:
                        write_jsonl(qa_f, qa)
                        write_jsonl(questions_f, question_view(qa))
                        write_jsonl(answer_key_f, answer_view(qa))
                        write_jsonl(messages_f, message_view(qa))
                        category_counts[qa["category"]] += 1
                    qa_count += len(qas)

                    write_jsonl(
                        manifest_f,
                        {
                            "sample_id": sample_id,
                            "contract_path": contract["sample"]["contract_path"],
                            "image_path": contract["sample"]["image_path"],
                            "annotation_file": "annotations.jsonl",
                            "question_count": len(qas),
                            "variant_index": variant_index,
                            "target": {
                                "node_id": node_id,
                                "node_token": node_token(node_id),
                                "occupancy": "EMPTY"
                                if case["building"] is None
                                else f"{color_token(str(case['color']))} {building_token(str(case['building']))}",
                            },
                            "source": contract["source"],
                            "curriculum_stage": CURRICULUM_STAGE,
                            "requires_stage": REQUIRES_STAGE,
                            "dataset_role": DATASET_ROLE,
                        },
                    )
                    sample_count += 1
            if args.limit_samples is not None and sample_count >= args.limit_samples:
                break

    metadata = {
        "name": DATASET_NAME,
        "schema": DATASET_SCHEMA,
        "curriculum_stage": CURRICULUM_STAGE,
        "requires_stage": REQUIRES_STAGE,
        "dataset_role": DATASET_ROLE,
        "category_groups": CATEGORY_GROUPS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": sample_count,
        "qa_count": qa_count,
        "image_size": [args.image_size, args.image_size],
        "colors": colors,
        "occupancy_cases": [case["name"] for case in cases],
        "variants_per_case": args.variants_per_case,
        "distractor_buildings": args.distractor_buildings,
        "distractor_roads": args.distractor_roads,
        "category_counts": dict(category_counts),
        "files": {
            "manifest": manifest_path.name,
            "annotations": annotations_path.name,
            "messages": messages_path.name,
            "questions_dir": "questions",
            "qa": "questions/qa.jsonl",
            "questions": "questions/questions.jsonl",
            "answer_key": "questions/answer_key.jsonl",
            "contracts_dir": "contracts",
            "images_dir": "images" if args.assume_rendered_images else None,
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    write_readme(
        output_dir,
        sample_count=sample_count,
        qa_count=qa_count,
        assume_rendered_images=args.assume_rendered_images,
    )

    print(f"wrote_samples={sample_count}")
    print(f"wrote_qa={qa_count}")
    print(f"colors={','.join(colors)}")
    print(f"categories={','.join(sorted(category_counts))}")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
