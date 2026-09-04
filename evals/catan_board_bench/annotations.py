"""Frontend-aligned Catan annotation geometry.

This mirrors the constants and coordinate transform used by
``playground/frontend/src/components/HexBoard.tsx`` so benchmark contracts can
emit machine-readable bbox and point annotations for data generation.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from evals.catan_board_bench.paths import RELATIVE_DATASETS_DIR
from evals.catan_board_bench.tokens import atlas_metadata, canonical_edge


HEX_SIZE = 50.0
HEX_SPACING = 0.0
TILE_BLEED = 1.5
TILE_WIDTH = math.sqrt(3) * HEX_SIZE + TILE_BLEED
TILE_HEIGHT = 2 * HEX_SIZE + TILE_BLEED
NUMBER_TOKEN_SIZE = HEX_SIZE * 0.7
NUMBER_TOKEN_Y_OFFSET = HEX_SIZE * 0.30
PORT_SHIP_SIZE = HEX_SIZE * 1.05
PORT_SHIP_X_OFFSET = -PORT_SHIP_SIZE * 0.1
PORT_SHIP_Y_OFFSET = -PORT_SHIP_SIZE * 0.25
NODE_BOX_SIZE = 16.0
EDGE_BOX_PAD = 8.0


def contract_to_render_state(contract: dict[str, Any]) -> dict[str, Any]:
    """Convert a public board contract into the minimal HexBoard GameState."""

    refs = _atlas_render_refs()
    placed_tiles: list[dict[str, Any]] = []

    for tile in contract.get("tiles", []):
        resource = tile.get("resource")
        if resource is None:
            tile_payload = {"id": tile["id"], "type": "DESERT"}
        else:
            tile_payload = {
                "id": tile["id"],
                "type": "RESOURCE_TILE",
                "resource": resource,
                "number": tile.get("number"),
            }
        placed_tiles.append({"coordinate": tile["coord"], "tile": tile_payload})

    for port in contract.get("ports", []):
        resource = port.get("resource")
        placed_tiles.append(
            {
                "coordinate": port["coord"],
                "tile": {
                    "id": port["id"],
                    "type": "PORT",
                    "direction": port["direction"],
                    "resource": resource,
                    "emoji": "3:1" if resource is None else resource,
                    "port_nodes": port["attached_nodes"],
                },
            }
        )

    nodes: dict[str, dict[str, Any]] = {}
    for node in contract.get("nodes", []):
        ref = refs["nodes"].get(node["id"])
        if not ref:
            continue
        nodes[str(node["id"])] = {
            "id": node["id"],
            "tile_coordinate": ref["tile_coordinate"],
            "direction": ref["direction"],
            "building": node.get("building"),
            "color": node.get("color"),
        }

    edges: list[dict[str, Any]] = []
    for edge in contract.get("edges", []):
        edge_id = tuple(canonical_edge(tuple(edge["id"])))
        ref = refs["edges"].get(edge_id)
        if not ref:
            continue
        edges.append(
            {
                "id": list(edge_id),
                "tile_coordinate": ref["tile_coordinate"],
                "direction": ref["direction"],
                "color": edge.get("road_color"),
            }
        )

    players = [player["color"] for player in contract.get("players", [])]
    current = contract.get("current", {})

    return {
        "tiles": placed_tiles,
        "nodes": nodes,
        "edges": edges,
        "robber_coordinate": contract.get("robber", {}).get("coord"),
        "adjacent_tiles": {},
        "colors": players,
        "current_color": current.get("current_color") or (players[0] if players else "RED"),
        "bot_colors": [],
        "player_state": {},
        "longest_roads_by_player": {},
        "played_knights_by_player": {},
        "is_initial_build_phase": current.get("is_initial_build_phase", False),
        "current_prompt": current.get("current_prompt", ""),
        "current_playable_actions": [],
        "actions": [],
        "winning_color": None,
        "state_index": current.get("num_completed_turns", 0),
        "achievements": contract.get("achievements", {}),
    }


def annotation_payload_for_contract(
    contract: dict[str, Any],
    *,
    image_size: int = 512,
) -> dict[str, Any]:
    render_state = contract_to_render_state(contract)
    annotations = annotations_for_render_state(render_state, image_size=image_size)
    counts: dict[str, int] = {}
    for annotation in annotations:
        counts[annotation["kind"]] = counts.get(annotation["kind"], 0) + 1

    return {
        "schema": "catan_frontend_annotations/v0",
        "image_size": [image_size, image_size],
        "sample": contract.get("sample", {}),
        "source": contract.get("source", {}),
        "view_box": _view_box_payload(render_state, image_size),
        "counts": counts,
        "annotations": annotations,
    }


def annotations_for_render_state(
    render_state: dict[str, Any],
    *,
    image_size: int = 512,
) -> list[dict[str, Any]]:
    node_positions = _node_positions(render_state)
    transform = _pixel_transform(_view_box(render_state), image_size)
    annotations: list[dict[str, Any]] = []

    for placed in render_state.get("tiles", []):
        tile = placed["tile"]
        if tile["type"] == "PORT":
            continue

        center = _hex_to_pixel(placed["coordinate"])
        token = _object_token("T", tile["id"])
        semantic = {
            "tile_id": tile["id"],
            "type": tile["type"],
            "resource": tile.get("resource"),
            "number": tile.get("number"),
        }
        if tile["type"] == "DESERT":
            semantic["resource"] = "DESERT"

        annotations.append(
            _annotation(
                kind="tile",
                token=token,
                bbox_svg=_rect(
                    center["x"] - TILE_WIDTH / 2,
                    center["y"] - TILE_HEIGHT / 2,
                    TILE_WIDTH,
                    TILE_HEIGHT,
                ),
                point_svg=center,
                transform=transform,
                semantic=semantic,
            )
        )

        if tile["type"] == "RESOURCE_TILE":
            number_point = {"x": center["x"], "y": center["y"] + NUMBER_TOKEN_Y_OFFSET}
            annotations.append(
                _annotation(
                    kind="tile_number",
                    token=token,
                    label=f"{token}:number",
                    bbox_svg=_rect(
                        center["x"] - NUMBER_TOKEN_SIZE / 2,
                        center["y"] - NUMBER_TOKEN_SIZE / 2 + NUMBER_TOKEN_Y_OFFSET,
                        NUMBER_TOKEN_SIZE,
                        NUMBER_TOKEN_SIZE,
                    ),
                    point_svg=number_point,
                    transform=transform,
                    semantic={"tile_id": tile["id"], "number": tile.get("number")},
                )
            )

    for edge in render_state.get("edges", []):
        node_a, node_b = edge["id"]
        pos_a = node_positions.get(node_a)
        pos_b = node_positions.get(node_b)
        if not pos_a or not pos_b:
            continue

        token = _edge_token(edge["id"])
        bbox = {
            "x1": min(pos_a["x"], pos_b["x"]) - EDGE_BOX_PAD,
            "y1": min(pos_a["y"], pos_b["y"]) - EDGE_BOX_PAD,
            "x2": max(pos_a["x"], pos_b["x"]) + EDGE_BOX_PAD,
            "y2": max(pos_a["y"], pos_b["y"]) + EDGE_BOX_PAD,
        }
        annotations.append(
            _annotation(
                kind="edge",
                token=token,
                bbox_svg=bbox,
                point_svg={
                    "x": (pos_a["x"] + pos_b["x"]) / 2,
                    "y": (pos_a["y"] + pos_b["y"]) / 2,
                },
                transform=transform,
                semantic={
                    "edge": edge["id"],
                    "road_color": edge.get("color"),
                    "node_tokens": [_object_token("N", node_a), _object_token("N", node_b)],
                },
            )
        )

    for placed in render_state.get("tiles", []):
        tile = placed["tile"]
        if tile["type"] != "PORT":
            continue

        center = _hex_to_pixel(placed["coordinate"])
        token = _object_token("P", tile["id"])
        annotations.append(
            _annotation(
                kind="port",
                token=token,
                bbox_svg=_rect(
                    center["x"] - PORT_SHIP_SIZE / 2 + PORT_SHIP_X_OFFSET,
                    center["y"] - PORT_SHIP_SIZE / 2 + PORT_SHIP_Y_OFFSET,
                    PORT_SHIP_SIZE,
                    PORT_SHIP_SIZE,
                ),
                point_svg=center,
                transform=transform,
                semantic={
                    "port_id": tile["id"],
                    "resource": tile.get("resource"),
                    "ratio": "3:1" if tile.get("resource") is None else "2:1",
                    "attached_nodes": tile.get("port_nodes", []),
                    "attached_node_tokens": [
                        _object_token("N", node_id) for node_id in tile.get("port_nodes", [])
                    ],
                },
            )
        )

    for node in render_state.get("nodes", {}).values():
        pos = node_positions.get(node["id"])
        if not pos:
            continue

        token = _object_token("N", node["id"])
        annotations.append(
            _annotation(
                kind="node",
                token=token,
                bbox_svg=_rect(
                    pos["x"] - NODE_BOX_SIZE / 2,
                    pos["y"] - NODE_BOX_SIZE / 2,
                    NODE_BOX_SIZE,
                    NODE_BOX_SIZE,
                ),
                point_svg=pos,
                transform=transform,
                semantic={
                    "node_id": node["id"],
                    "building": node.get("building"),
                    "color": node.get("color"),
                },
            )
        )

    return annotations


@lru_cache(maxsize=1)
def _atlas_render_refs() -> dict[str, Any]:
    atlas = atlas_metadata()

    node_refs: dict[int, dict[str, Any]] = {}
    edge_refs: dict[tuple[int, int], dict[str, Any]] = {}
    for tile in atlas["tiles"]:
        coord = tile["coord"]
        for direction, node_id in tile["nodes"].items():
            node_refs.setdefault(node_id, {"tile_coordinate": coord, "direction": direction})
        for direction, edge in tile["edges"].items():
            edge_refs.setdefault(tuple(edge), {"tile_coordinate": coord, "direction": direction})

    return {"nodes": node_refs, "edges": edge_refs}


def _hex_to_pixel(coord: list[int] | tuple[int, int, int]) -> dict[str, float]:
    cube_x, _cube_y, cube_z = coord
    size = HEX_SIZE + HEX_SPACING
    q = cube_x
    r = cube_z
    return {
        "x": size * math.sqrt(3) * (q + r / 2),
        "y": size * (3 / 2) * r,
    }


def _node_offset(direction: str) -> dict[str, float]:
    s = HEX_SIZE + HEX_SPACING
    w = s * math.sqrt(3) / 2
    offsets = {
        "NORTH": {"x": 0.0, "y": -s},
        "NORTHEAST": {"x": w, "y": -s / 2},
        "SOUTHEAST": {"x": w, "y": s / 2},
        "SOUTH": {"x": 0.0, "y": s},
        "SOUTHWEST": {"x": -w, "y": s / 2},
        "NORTHWEST": {"x": -w, "y": -s / 2},
    }
    return offsets.get(direction, {"x": 0.0, "y": 0.0})


def _node_positions(render_state: dict[str, Any]) -> dict[int, dict[str, float]]:
    positions = {}
    for node in render_state.get("nodes", {}).values():
        center = _hex_to_pixel(node["tile_coordinate"])
        offset = _node_offset(node["direction"])
        positions[node["id"]] = {
            "x": center["x"] + offset["x"],
            "y": center["y"] + offset["y"],
        }
    return positions


def _view_box(render_state: dict[str, Any]) -> dict[str, float]:
    positions = [_hex_to_pixel(placed["coordinate"]) for placed in render_state.get("tiles", [])]
    min_x = min(point["x"] for point in positions) - HEX_SIZE * 2.5
    max_x = max(point["x"] for point in positions) + HEX_SIZE * 2.5
    min_y = min(point["y"] for point in positions) - HEX_SIZE * 2.5
    max_y = max(point["y"] for point in positions) + HEX_SIZE * 2.5
    return {"min_x": min_x, "min_y": min_y, "width": max_x - min_x, "height": max_y - min_y}


def _view_box_payload(render_state: dict[str, Any], image_size: int) -> dict[str, float]:
    view_box = _view_box(render_state)
    transform = _pixel_transform(view_box, image_size)
    return {**view_box, **transform}


def _pixel_transform(view_box: dict[str, float], image_size: int) -> dict[str, float]:
    scale = min(image_size / view_box["width"], image_size / view_box["height"])
    return {
        "scale": scale,
        "offset_x": (image_size - view_box["width"] * scale) / 2,
        "offset_y": (image_size - view_box["height"] * scale) / 2,
        "image_size": float(image_size),
        "min_x": view_box["min_x"],
        "min_y": view_box["min_y"],
    }


def _annotation(
    *,
    kind: str,
    token: str,
    bbox_svg: dict[str, float],
    point_svg: dict[str, float],
    transform: dict[str, float],
    semantic: dict[str, Any],
    label: str | None = None,
) -> dict[str, Any]:
    bbox = _bbox_to_pixels(bbox_svg, transform)
    center = _point_to_pixels(point_svg, transform)
    annotation = {
        "id": f"{kind}:{label or token}",
        "kind": kind,
        "token": token,
        "bbox": bbox,
        "bbox_2d": bbox,
        "center": center,
        "point": center,
        "bbox_svg": _round_bbox(bbox_svg),
        "point_svg": _round_point(point_svg),
        "semantic": semantic,
    }
    if label:
        annotation["label"] = label
    return annotation


def _rect(x: float, y: float, width: float, height: float) -> dict[str, float]:
    return {"x1": x, "y1": y, "x2": x + width, "y2": y + height}


def _bbox_to_pixels(bbox: dict[str, float], transform: dict[str, float]) -> list[int]:
    p1 = _point_to_pixels({"x": bbox["x1"], "y": bbox["y1"]}, transform)
    p2 = _point_to_pixels({"x": bbox["x2"], "y": bbox["y2"]}, transform)
    image_size = int(transform["image_size"])
    return [
        _clamp_int(min(p1[0], p2[0]), 0, image_size),
        _clamp_int(min(p1[1], p2[1]), 0, image_size),
        _clamp_int(max(p1[0], p2[0]), 0, image_size),
        _clamp_int(max(p1[1], p2[1]), 0, image_size),
    ]


def _point_to_pixels(point: dict[str, float], transform: dict[str, float]) -> list[int]:
    x = (point["x"] - transform["min_x"]) * transform["scale"] + transform["offset_x"]
    y = (point["y"] - transform["min_y"]) * transform["scale"] + transform["offset_y"]
    image_size = int(transform["image_size"])
    return [_clamp_int(x, 0, image_size), _clamp_int(y, 0, image_size)]


def _clamp_int(value: float, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(round(value))))


def _round_bbox(bbox: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 3) for key, value in bbox.items()}


def _round_point(point: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 3) for key, value in point.items()}


def _object_token(prefix: str, object_id: int) -> str:
    return f"<{prefix}{object_id:02d}>"


def _edge_token(edge: list[int] | tuple[int, int]) -> str:
    a, b = canonical_edge(tuple(edge))
    return f"<E{a:02d}_{b:02d}>"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=RELATIVE_DATASETS_DIR / "catan_board_bench_100",
        help="Benchmark directory containing contracts/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("-"),
        help="JSONL output path, or '-' for stdout.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--sample-id",
        help="Optional sample id such as sample_000. Defaults to all contracts.",
    )
    args = parser.parse_args(argv)

    contract_dir = args.bench_dir / "contracts"
    if args.sample_id:
        contract_paths = [contract_dir / f"{args.sample_id}.json"]
    else:
        contract_paths = sorted(contract_dir.glob("sample_*.json"))

    rows = []
    for contract_path in contract_paths:
        contract = json.loads(contract_path.read_text())
        rows.append(annotation_payload_for_contract(contract, image_size=args.image_size))

    lines = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    if str(args.output) == "-":
        print(lines, end="")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(lines)
    print(f"wrote {len(rows)} annotation rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
