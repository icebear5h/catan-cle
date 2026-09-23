"""Deterministic candidate selection and answer derivation helpers."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from evals.catan_board_bench.builder.shapes import rows, text
from evals.json_types import JsonDict, JsonValue, as_int, as_list


def _pick(items: Sequence[JsonDict], count: int, seed: int, *, stride: int) -> List[JsonDict]:
    if not items:
        return []
    result: List[JsonDict] = []
    seen: set[int] = set()
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
    predicate: Callable[[JsonDict], bool],
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


def _find_by_id(items: Sequence[JsonDict], item_id: JsonValue) -> Optional[JsonDict]:
    for item in items:
        if item.get("id") == item_id:
            return item
    return None


def _tile_occupied_nodes_target(contract: JsonDict, tile: JsonDict) -> List[JsonDict]:
    nodes_by_id = {node["id"]: node for node in rows(contract, "nodes")}
    occupied: List[JsonDict] = []
    for node_id in as_list(tile["nodes"], "tile nodes"):
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
        f"{node['node_token']} {node['color_token']} {node['building_token']}" for node in occupied
    )


def _port_occupied_nodes_target(contract: JsonDict, port: JsonDict) -> List[JsonDict]:
    nodes_by_id = {node["id"]: node for node in rows(contract, "nodes")}
    occupied: List[JsonDict] = []
    for node_id in as_list(port["attached_nodes"], "port attached_nodes"):
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
        f"{node['color_token']} {node['building_token']} {node['node_token']}" for node in occupied
    )


def _road_edges_for_color(contract: JsonDict, color: str) -> List[JsonDict]:
    return sorted(
        [edge for edge in rows(contract, "edges") if edge["road_color"] == color],
        key=lambda edge: text(edge["token"], "edge token"),
    )


def _disconnected_node_pair(contract: JsonDict, seed: int) -> Tuple[int, int]:
    node_ids = [as_int(node["id"], "node id") for node in rows(contract, "nodes")]
    connected_edges = {tuple(as_list(edge["id"], "edge id")) for edge in rows(contract, "edges")}
    start = (seed * 13) % len(node_ids)
    for offset in range(len(node_ids) * len(node_ids)):
        a = node_ids[(start + offset) % len(node_ids)]
        b = node_ids[(start + 5 + offset * 7) % len(node_ids)]
        if a == b:
            continue
        low, high = sorted((a, b))
        if (low, high) not in connected_edges:
            return low, high
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

