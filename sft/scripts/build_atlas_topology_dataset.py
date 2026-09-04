"""Build Phase 0 atlas/token topology SFT examples.

This is text-only SFT. It teaches stable Catan atlas semantics before visual
grounding:

    <N11> -> fixed vertex topology
    <T09> -> fixed tile topology
    <E18_40> -> fixed edge topology
    <P06> -> fixed port slot topology
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.catan_board_bench.tokens import atlas_metadata


DEFAULT_PROMPT_PREFIX = "Answer exactly using Catan atlas tokens. Do not explain."


def message_row(row_id: str, category: str, question: str, answer: str, target: dict[str, Any]):
    return {
        "id": row_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{DEFAULT_PROMPT_PREFIX}\n\nQuestion: {question}",
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
        "metadata": {
            "phase": "atlas_topology",
            "category": category,
            "target": target,
        },
    }


def tokens(items: list[dict[str, Any]], key: str = "token") -> list[str]:
    return [str(item[key]) for item in items]


def build_indices(atlas: dict[str, Any]) -> dict[str, Any]:
    tiles = {tile["id"]: tile for tile in atlas["tiles"]}
    nodes = {node["id"]: node for node in atlas["nodes"]}
    edges = {tuple(edge["id"]): edge for edge in atlas["edges"]}
    ports = {port["id"]: port for port in atlas["ports"]}

    node_tiles: dict[int, set[int]] = defaultdict(set)
    node_edges: dict[int, set[tuple[int, int]]] = defaultdict(set)
    edge_tiles: dict[tuple[int, int], set[int]] = defaultdict(set)
    node_ports: dict[int, set[int]] = defaultdict(set)

    for tile in tiles.values():
        for node_id in tile["nodes"].values():
            node_tiles[node_id].add(tile["id"])
        for edge in tile["edges"].values():
            edge_tuple = tuple(edge)
            edge_tiles[edge_tuple].add(tile["id"])
            a, b = edge_tuple
            node_edges[a].add(edge_tuple)
            node_edges[b].add(edge_tuple)

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
    }


def build_examples() -> list[dict[str, Any]]:
    atlas = atlas_metadata()
    indices = build_indices(atlas)
    rows: list[dict[str, Any]] = []

    for tile_id, tile in sorted(indices["tiles"].items()):
        tile_tok = tile["token"]
        node_ids = sorted(tile["nodes"].values())
        edge_ids = sorted(tuple(edge) for edge in tile["edges"].values())
        coord = tile["coord"]
        rows.extend(
            [
                message_row(
                    f"atlas_tile_{tile_id:02d}_nodes",
                    "tile_nodes",
                    f"Which nodes touch {tile_tok}?",
                    " ".join(f"<N{node_id:02d}>" for node_id in node_ids),
                    {"tile": tile_tok, "nodes": node_ids},
                ),
                message_row(
                    f"atlas_tile_{tile_id:02d}_edges",
                    "tile_edges",
                    f"Which edges border {tile_tok}?",
                    " ".join(f"<E{a:02d}_{b:02d}>" for a, b in edge_ids),
                    {"tile": tile_tok, "edges": [list(edge) for edge in edge_ids]},
                ),
                message_row(
                    f"atlas_tile_{tile_id:02d}_coord",
                    "tile_coordinate",
                    f"What cube coordinate does {tile_tok} occupy?",
                    f"Q {coord[0]} R {coord[1]} S {coord[2]}",
                    {"tile": tile_tok, "coord": coord},
                ),
            ]
        )

    for node_id, node in sorted(indices["nodes"].items()):
        node_tok = node["token"]
        tile_ids = sorted(indices["node_tiles"].get(node_id, set()))
        edge_ids = sorted(indices["node_edges"].get(node_id, set()))
        port_ids = sorted(indices["node_ports"].get(node_id, set()))
        status = "PORT_ADJACENT" if port_ids else ("INTERIOR" if len(tile_ids) == 3 else "COAST")
        rows.extend(
            [
                message_row(
                    f"atlas_node_{node_id:02d}_tiles",
                    "node_tiles",
                    f"Which tiles touch {node_tok}?",
                    " ".join(f"<T{tile_id:02d}>" for tile_id in tile_ids) or "NONE",
                    {"node": node_tok, "tiles": tile_ids},
                ),
                message_row(
                    f"atlas_node_{node_id:02d}_edges",
                    "node_edges",
                    f"Which edges touch {node_tok}?",
                    " ".join(f"<E{a:02d}_{b:02d}>" for a, b in edge_ids) or "NONE",
                    {"node": node_tok, "edges": [list(edge) for edge in edge_ids]},
                ),
                message_row(
                    f"atlas_node_{node_id:02d}_status",
                    "node_status",
                    f"Is {node_tok} INTERIOR, COAST, or PORT_ADJACENT?",
                    status,
                    {"node": node_tok, "status": status, "ports": port_ids},
                ),
            ]
        )

    for edge, edge_info in sorted(indices["edges"].items()):
        a, b = edge
        edge_tok = edge_info["token"]
        tile_ids = sorted(indices["edge_tiles"].get(edge, set()))
        rows.extend(
            [
                message_row(
                    f"atlas_edge_{a:02d}_{b:02d}_nodes",
                    "edge_nodes",
                    f"Which nodes does {edge_tok} connect?",
                    f"<N{a:02d}> <N{b:02d}>",
                    {"edge": edge_tok, "nodes": [a, b]},
                ),
                message_row(
                    f"atlas_edge_{a:02d}_{b:02d}_tiles",
                    "edge_tiles",
                    f"Which land tiles touch {edge_tok}?",
                    " ".join(f"<T{tile_id:02d}>" for tile_id in tile_ids) or "NONE",
                    {"edge": edge_tok, "tiles": tile_ids},
                ),
            ]
        )

    for port_id, port in sorted(indices["ports"].items()):
        port_tok = port["token"]
        node_ids = sorted(port["attached_nodes"])
        coord = port["coord"]
        rows.extend(
            [
                message_row(
                    f"atlas_port_{port_id:02d}_nodes",
                    "port_nodes",
                    f"Which nodes touch {port_tok}?",
                    " ".join(f"<N{node_id:02d}>" for node_id in node_ids),
                    {"port": port_tok, "nodes": node_ids},
                ),
                message_row(
                    f"atlas_port_{port_id:02d}_coord",
                    "port_coordinate",
                    f"What cube coordinate does {port_tok} occupy?",
                    f"Q {coord[0]} R {coord[1]} S {coord[2]}",
                    {"port": port_tok, "coord": coord},
                ),
                message_row(
                    f"atlas_port_{port_id:02d}_direction",
                    "port_direction",
                    f"Which side direction is {port_tok} on?",
                    str(port["direction"]),
                    {"port": port_tok, "direction": port["direction"]},
                ),
            ]
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = build_examples()
    if args.limit:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    categories = sorted({row["metadata"]["category"] for row in rows})
    print(f"wrote_rows={len(rows)}")
    print(f"categories={','.join(categories)}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
