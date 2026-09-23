"""Canonical atlas context and the screenshot prompt."""

from __future__ import annotations

from collections import defaultdict

from evals.catan_board_bench.tokens import atlas_metadata
from scripts.board_bench.shapes import JsonDict, integer, obj, objs, strings, text, values

__all__ = [
    "build_prompt",
    "canonical_atlas_context",
    "edge_anchor_text",
    "node_anchor_text",
    "tile_rows_text",
]


def build_prompt(contract: JsonDict, qa: JsonDict) -> str:
    context = canonical_atlas_context(contract, qa)
    return (
        f"Fixed engine-atlas context:\n{context}\n\n"
        f"Question: {qa['question']}\n"
        f"Required JSON shape: {qa['output_schema']}\n"
        "Return exactly one JSON object."
    )


def canonical_atlas_context(contract: JsonDict, qa: JsonDict) -> str:
    lines = [tile_rows_text(contract)]
    category = text(qa["category"], "category")
    target = obj(qa["target"], "question target")
    if category in {"node_state", "node_adjacent_tiles"}:
        lines.append(node_anchor_text(text(target["node"], "target node")))
    elif category == "nodes_connected":
        lines.extend(node_anchor_text(node) for node in strings(target["nodes"], "target nodes"))
    elif category == "edge_state":
        lines.append(edge_anchor_text(text(target["edge"], "target edge")))
    elif category == "port_occupancy":
        port_id = int(text(target["port"], "target port")[1:])
        port = next(
            row for row in objs(contract["ports"], "contract ports") if row["id"] == port_id
        )
        endpoint_text = " ".join(
            f"N{integer(node_id, 'attached node'):02d}"
            for node_id in values(port["attached_nodes"], "attached_nodes")
        )
        lines.append(
            f"P{port_id:02d} is the port at cube {port['coord']} facing "
            f"{port['direction']} and has endpoints {endpoint_text}."
        )
    return "\n".join(lines)


def tile_rows_text(contract: JsonDict) -> str:
    rows: dict[int, list[JsonDict]] = defaultdict(list)
    for tile in objs(contract["tiles"], "contract tiles"):
        coord = values(tile["coord"], "tile coord")
        rows[integer(coord[2], "tile coord z")].append(tile)
    parts = []
    for row_index, z in enumerate(sorted(rows)):
        tiles = sorted(
            rows[z],
            key=lambda tile: integer(values(tile["coord"], "tile coord")[0], "tile coord x"),
        )
        parts.append(
            f"row {row_index} left-to-right: "
            + " ".join(f"T{integer(tile['id'], 'tile id'):02d}" for tile in tiles)
        )
    return "Tile rows top-to-bottom: " + "; ".join(parts) + "."


def node_anchor_text(node_token: str) -> str:
    node_id = int(node_token[1:])
    candidates = []
    for tile in atlas_metadata()["tiles"]:
        for direction, candidate in tile["nodes"].items():
            if candidate == node_id:
                candidates.append((tile["id"], direction))
    if not candidates:
        raise ValueError(f"unknown canonical node: {node_token}")
    tile_id, direction = min(candidates)
    return f"{node_token} is the {direction} corner of T{tile_id:02d}."


def edge_anchor_text(edge_token: str) -> str:
    node_text = edge_token[1:]
    node_a, node_b = (int(value) for value in node_text.split("_"))
    target = tuple(sorted((node_a, node_b)))
    candidates = []
    for tile in atlas_metadata()["tiles"]:
        for direction, edge in tile["edges"].items():
            if tuple(sorted(edge)) == target:
                candidates.append((tile["id"], direction))
    if not candidates:
        raise ValueError(f"unknown canonical edge: {edge_token}")
    tile_id, direction = min(candidates)
    return (
        f"{edge_token} connects N{node_a:02d} and N{node_b:02d}; it is the "
        f"{direction} side of T{tile_id:02d}."
    )
