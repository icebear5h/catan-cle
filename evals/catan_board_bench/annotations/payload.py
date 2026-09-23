"""Machine-readable bbox and point annotations for one render state."""

from __future__ import annotations

from evals.catan_board_bench.annotations.constants import (
    EDGE_BOX_PAD,
    NODE_BOX_SIZE,
    NUMBER_TOKEN_SIZE,
    NUMBER_TOKEN_Y_OFFSET,
    PORT_SHIP_SIZE,
    PORT_SHIP_X_OFFSET,
    PORT_SHIP_Y_OFFSET,
    TILE_HEIGHT,
    TILE_WIDTH,
)
from evals.catan_board_bench.annotations.geometry import (
    _annotation,
    _edge_token,
    _hex_to_pixel,
    _node_positions,
    _object_token,
    _pixel_transform,
    _rect,
    _view_box,
    _view_box_payload,
)
from evals.catan_board_bench.annotations.render_state import RenderState, contract_to_render_state
from evals.json_types import JsonDict, as_str


def annotation_payload_for_contract(
    contract: JsonDict,
    *,
    image_size: int = 512,
) -> JsonDict:
    render_state = contract_to_render_state(contract)
    annotations = annotations_for_render_state(render_state, image_size=image_size)
    counts: dict[str, int] = {}
    for annotation in annotations:
        kind = as_str(annotation["kind"], "annotation kind")
        counts[kind] = counts.get(kind, 0) + 1

    return {
        "schema": "catan_frontend_annotations/v0",
        "image_size": [image_size, image_size],
        "sample": contract.get("sample", {}),
        "source": contract.get("source", {}),
        "view_box": dict(_view_box_payload(render_state, image_size)),
        "counts": dict(counts),
        "annotations": list(annotations),
    }


def annotations_for_render_state(
    render_state: RenderState,
    *,
    image_size: int = 512,
) -> list[JsonDict]:
    node_positions = _node_positions(render_state)
    transform = _pixel_transform(_view_box(render_state), image_size)
    annotations: list[JsonDict] = []

    for placed in render_state.get("tiles", []):
        tile = placed["tile"]
        if tile["type"] == "PORT":
            continue

        center = _hex_to_pixel(placed["coordinate"])
        token = _object_token("T", tile["id"])
        semantic: JsonDict = {
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
                    "edge": list(edge["id"]),
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
                    "attached_nodes": list(tile.get("port_nodes", [])),
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

