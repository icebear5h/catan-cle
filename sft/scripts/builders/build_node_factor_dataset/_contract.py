from __future__ import annotations

import random
from pathlib import Path

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
from sft.json_types import JsonDict, as_dict, as_int, as_list, json_list

from ._sources import (
    CURRICULUM_STAGE,
    DATASET_NAME,
    DATASET_ROLE,
    NUMBER_DECK,
    PORT_DECK,
    REQUIRES_STAGE,
    RESOURCE_DECK,
    AtlasIndices,
    edge_pair,
)
from ._variants import choose_distractor_buildings, choose_roads, player_summaries


def build_contract(
    *,
    atlas: JsonDict,
    indices: AtlasIndices,
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
) -> JsonDict:
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
    sample: JsonDict = {
        "id": sample_id,
        "index": sample_index,
        "contract_path": str(contract_rel),
        "image_path": str(image_rel) if image_rel else None,
        "image_size": [image_size, image_size] if image_rel else None,
        "target_node": node_token(target_node_id),
    }
    source: JsonDict = {
        "kind": DATASET_NAME,
        "generator": "sft/scripts/builders/build_node_factor_dataset.py",
        "seed": seed,
        "render_status": "pending_frontend_render" if image_rel else "contract_only",
        "leakage_note": "Synthetic atlas coverage; not sampled from CatanBoardBench-100 game IDs.",
        "curriculum_stage": CURRICULUM_STAGE,
        "requires_stage": REQUIRES_STAGE,
        "dataset_role": DATASET_ROLE,
    }

    tiles: list[JsonDict] = []
    for tile_id, tile in sorted(indices["tiles"].items()):
        resource = tile_resource_by_id[tile_id]
        number = tile_number_by_id[tile_id]
        tile_edges = sorted(canonical_edge(edge_pair(edge)) for edge in as_dict(tile["edges"]).values())
        tile_nodes = sorted(as_int(node_id) for node_id in as_dict(tile["nodes"]).values())
        tiles.append(
            {
                "id": tile_id,
                "token": tile_token(tile_id),
                "coord": tile["coord"],
                "resource": resource,
                "resource_token": resource_token(resource),
                "number": number,
                "has_robber": tile_id == robber_tile_id,
                "nodes": json_list(tile_nodes),
                "node_tokens": [node_token(node_id) for node_id in tile_nodes],
                "edges": [list(edge) for edge in tile_edges],
                "edge_tokens": [edge_token(edge) for edge in tile_edges],
            }
        )

    nodes: list[JsonDict] = []
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
                "adjacent_tiles": json_list(adjacent_tiles),
                "adjacent_tile_tokens": [tile_token(tile_id) for tile_id in adjacent_tiles],
                "adjacent_edges": [list(edge) for edge in adjacent_edges],
                "adjacent_edge_tokens": [edge_token(edge) for edge in adjacent_edges],
                "port_ids": json_list(attached_ports),
                "port_tokens": [port_token(port_id) for port_id in attached_ports],
            }
        )

    edges: list[JsonDict] = []
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

    ports: list[JsonDict] = []
    for port_id, port in sorted(indices["ports"].items()):
        resource = port_resources.pop()
        attached_nodes = sorted(as_int(node_id) for node_id in as_list(port["attached_nodes"]))
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
                "attached_nodes": json_list(attached_nodes),
                "attached_node_tokens": [node_token(node_id) for node_id in attached_nodes],
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
        "players": json_list(players),
        "tiles": json_list(tiles),
        "nodes": json_list(nodes),
        "edges": json_list(edges),
        "ports": json_list(ports),
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
