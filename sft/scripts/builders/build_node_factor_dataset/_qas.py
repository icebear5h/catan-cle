from __future__ import annotations

import json

from sft.json_types import (
    JsonDict,
    JsonList,
    JsonValue,
    as_dict,
    as_int,
    as_list,
    as_str,
    json_dict,
    json_list,
)

from ._sources import CURRICULUM_STAGE, REQUIRES_STAGE, category_group, json_objects


def build_qas(contract: JsonDict, target_node_id: int, annotations: JsonDict) -> list[JsonDict]:
    sample = as_dict(contract["sample"])
    sample_id = sample["id"]
    node = as_dict(as_list(contract["nodes"])[target_node_id])
    target_annotation = find_annotation(annotations, kind="node", token=as_str(node["token"]))
    if target_annotation is None:
        raise RuntimeError(f"missing node annotation for {node['token']}")

    qas: list[JsonDict] = []

    def add(category: str, question: str, answer: str, target: JsonDict, scoring: str = "exact") -> None:
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
            "adjacent_tiles": json_list(local_tile_payloads(contract, node)),
        },
    )
    add(
        "node_incident_road_owners",
        f"Who owns the roads touching {node['token']}?",
        incident_road_answer(contract, node),
        {
            "node_id": node["id"],
            "node_token": node["token"],
            "incident_edges": json_list(local_edge_payloads(contract, node)),
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


def board_atlas_bbox_payload(annotations: JsonDict) -> JsonDict:
    groups: dict[str, JsonList] = {
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
        json_objects(annotations["annotations"]),
        key=lambda item: (as_str(item["kind"]), as_str(item.get("label") or item["token"])),
    ):
        group = kind_to_group.get(as_str(annotation["kind"]))
        if group is None:
            continue
        groups[group].append(
            {
                "label": annotation.get("label") or annotation["token"],
                "bbox_2d": annotation["bbox_2d"],
            }
        )
    return json_dict(groups)


def find_annotation(
    annotations: JsonDict,
    *,
    kind: str,
    token: str,
) -> JsonDict | None:
    for annotation in json_objects(annotations["annotations"]):
        if annotation["kind"] == kind and annotation["token"] == token:
            return annotation
    return None


def occupancy_answer(node: JsonDict) -> str:
    if node["building"] is None:
        return "EMPTY"
    return f"{node['color_token']} {node['building_token']}"


def adjacent_tile_answer(contract: JsonDict, node: JsonDict) -> str:
    return " ".join(tile_phrase(tile) for tile in local_tiles(contract, node))


def incident_road_answer(contract: JsonDict, node: JsonDict) -> str:
    payloads = local_edge_payloads(contract, node)
    return " ".join(f"{edge['edge']}:{edge['road']}" for edge in payloads) or "NONE"


def local_state_payload(contract: JsonDict, node: JsonDict, *, bbox: JsonValue) -> JsonDict:
    return {
        "node": node["token"],
        "bbox_2d": bbox,
        "occupancy": occupancy_answer(node),
        "adjacent_tiles": json_list(local_tile_payloads(contract, node)),
        "incident_edges": json_list(local_edge_payloads(contract, node)),
        "ports": json_list(local_port_payloads(contract, node)),
    }


def local_tiles(contract: JsonDict, node: JsonDict) -> list[JsonDict]:
    tiles_by_id = {as_int(tile["id"]): tile for tile in json_objects(contract["tiles"])}
    return [tiles_by_id[as_int(tile_id)] for tile_id in as_list(node["adjacent_tiles"])]


def local_tile_payloads(contract: JsonDict, node: JsonDict) -> list[JsonDict]:
    return [
        {
            "tile": tile["token"],
            "resource": tile["resource_token"],
            "number": tile["number"],
            "robber": tile["has_robber"],
        }
        for tile in local_tiles(contract, node)
    ]


def local_edge_payloads(contract: JsonDict, node: JsonDict) -> list[JsonDict]:
    edges_by_token = {as_str(edge["token"]): edge for edge in json_objects(contract["edges"])}
    payloads: list[JsonDict] = []
    for edge_token_value in as_list(node["adjacent_edge_tokens"]):
        edge = edges_by_token[as_str(edge_token_value)]
        payloads.append({"edge": as_str(edge["token"]), "road": as_str(edge["road_color_token"] or "EMPTY")})
    return payloads


def local_port_payloads(contract: JsonDict, node: JsonDict) -> list[JsonDict]:
    ports_by_id = {as_int(port["id"]): port for port in json_objects(contract["ports"])}
    payloads: list[JsonDict] = []
    for port_id in as_list(node["port_ids"]):
        port = ports_by_id[as_int(port_id)]
        payloads.append(
            {
                "port": port["token"],
                "trade": "GENERIC" if port["kind"] == "generic" else port["resource_token"],
                "ratio": port["ratio"],
            }
        )
    return payloads


def tile_phrase(tile: JsonDict) -> str:
    if tile["resource"] is None:
        return f"{tile['token']} <DESERT>"
    return f"{tile['token']} {tile['resource_token']} {tile['number']}"


def compact_json(value: JsonDict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
