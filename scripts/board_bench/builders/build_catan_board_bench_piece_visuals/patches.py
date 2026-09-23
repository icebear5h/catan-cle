"""Local board crops taken from the approved dummy contract."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from evals.catan_board_bench.annotations import annotation_payload_for_contract
from evals.catan_board_bench.render import DEFAULT_RENDER_STYLE, render_contract_image
from evals.catan_board_bench.tokens import (
    color_token,
    edge_token,
    node_token,
    resource_token,
    tile_token,
)
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.constants import (
    PROJECT_ROOT,
)
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.images import crop_square
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.writer import DatasetWriter
from scripts.board_bench.shapes import JsonDict, integer, objs, read_json_object, text, values

__all__ = ["add_crop_sample", "add_local_patch_examples"]

NODE_IDS = [1, 10, 14, 17, 22, 25, 33, 0, 52]
EDGE_IDS = [(0, 1), (22, 49), (24, 25), (1, 2), (48, 49), (10, 29)]
TILE_IDS = [15, 16, 1, 5, 7]
PORT_IDS = [0, 1, 3, 4, 7]
ROBBER_TILES = [(1, "YES"), (15, "NO")]


def _center(annotation: JsonDict) -> list[int]:
    return [integer(item, "annotation center") for item in values(annotation["center"], "center")]


def _by_kind_token(annotations: list[JsonDict]) -> dict[tuple[str, str], JsonDict]:
    return {
        (text(ann["kind"], "annotation kind"), text(ann["token"], "annotation token")): ann
        for ann in annotations
    }


def add_local_patch_examples(writer: DatasetWriter, contract_path: Path) -> None:
    contract = read_json_object(contract_path)
    full_image_size = 1024
    full_image = render_contract_image(
        contract, image_size=full_image_size, style=DEFAULT_RENDER_STYLE
    )
    annotation_payload = annotation_payload_for_contract(contract, image_size=full_image_size)
    annotations = objs(annotation_payload["annotations"], "annotations")
    by_kind_token = _by_kind_token(annotations)
    relative_contract = str(contract_path.relative_to(PROJECT_ROOT))

    nodes_by_id = {
        integer(node["id"], "node id"): node for node in objs(contract["nodes"], "contract nodes")
    }
    for node_id in NODE_IDS:
        token = node_token(node_id)
        ann = by_kind_token.get(("node", token))
        if ann is None:
            continue
        node = nodes_by_id[node_id]
        answer = "EMPTY"
        if node.get("building") and node.get("color"):
            answer = f"{color_token(text(node['color'], 'color'))} <{node['building']}>"
        add_crop_sample(
            writer,
            full_image,
            _center(ann),
            crop_size=180,
            sample_id=f"local_patch_node_{node_id:02d}",
            category="local_patch_node_occupancy",
            question="What building, if any, is centered in this crop?",
            answer=answer,
            target={
                "node": token,
                "color": node.get("color"),
                "building": node.get("building"),
                "source_center": list(_center(ann)),
            },
            contract_path=relative_contract,
        )

    edges_by_id = {
        (integer(values(edge["id"], "edge id")[0], "edge id"),
         integer(values(edge["id"], "edge id")[1], "edge id")): edge
        for edge in objs(contract["edges"], "contract edges")
    }
    for edge in EDGE_IDS:
        token = edge_token(edge)
        ann = by_kind_token.get(("edge", token))
        if ann is None:
            continue
        edge_payload = edges_by_id[edge]
        color = edge_payload.get("road_color")
        add_crop_sample(
            writer,
            full_image,
            _center(ann),
            crop_size=170,
            sample_id=f"local_patch_edge_{edge[0]:02d}_{edge[1]:02d}",
            category="local_patch_edge_road_owner",
            question="Who owns the centered road segment?",
            answer="EMPTY" if color is None else color_token(text(color, "road_color")),
            target={
                "edge": token,
                "road_color": color,
                "source_center": list(_center(ann)),
            },
            contract_path=relative_contract,
        )

    tiles_by_id = {
        integer(tile["id"], "tile id"): tile for tile in objs(contract["tiles"], "contract tiles")
    }
    for tile_id in TILE_IDS:
        token = tile_token(tile_id)
        ann = by_kind_token.get(("tile", token))
        if ann is None:
            continue
        tile = tiles_by_id[tile_id]
        if tile["resource"] is None:
            answer = "<DESERT>"
        else:
            answer = f"{resource_token(text(tile['resource'], 'resource'))} {tile['number']}"
        add_crop_sample(
            writer,
            full_image,
            _center(ann),
            crop_size=190,
            sample_id=f"local_patch_tile_{tile_id:02d}",
            category="local_patch_tile_resource_number",
            question="What resource and dice number are shown on the centered tile?",
            answer=answer,
            target={
                "tile": token,
                "resource": tile["resource"],
                "number": tile["number"],
                "source_center": list(_center(ann)),
            },
            contract_path=relative_contract,
        )

    ports_by_id = {
        integer(port["id"], "port id"): port for port in objs(contract["ports"], "contract ports")
    }
    for port_id in PORT_IDS:
        token = f"<P{port_id:02d}>"
        ann = by_kind_token.get(("port", token))
        if ann is None:
            continue
        port = ports_by_id[port_id]
        resource = port.get("resource")
        answer = (
            "GENERIC 3:1"
            if resource is None
            else f"{resource_token(text(resource, 'resource'))} 2:1"
        )
        add_crop_sample(
            writer,
            full_image,
            _center(ann),
            crop_size=180,
            sample_id=f"local_patch_port_{port_id:02d}",
            category="local_patch_port_trade_type",
            question="What trade port is visible in this crop?",
            answer=answer,
            target={
                "port": token,
                "resource": resource,
                "ratio": "3:1" if resource is None else "2:1",
                "source_center": list(_center(ann)),
            },
            contract_path=relative_contract,
        )

    for tile_id, robber_answer in ROBBER_TILES:
        token = tile_token(tile_id)
        ann = by_kind_token.get(("tile", token))
        if ann is None:
            continue
        add_crop_sample(
            writer,
            full_image,
            _center(ann),
            crop_size=210,
            sample_id=f"local_patch_robber_{tile_id:02d}",
            category="local_patch_robber_presence",
            question="Is the robber visible in this crop?",
            answer=robber_answer,
            target={
                "tile": token,
                "robber": robber_answer == "YES",
                "source_center": list(_center(ann)),
            },
            contract_path=relative_contract,
        )


def add_crop_sample(
    writer: DatasetWriter,
    full_image: Image.Image,
    center: list[int],
    *,
    crop_size: int,
    sample_id: str,
    category: str,
    question: str,
    answer: str,
    target: JsonDict,
    contract_path: str,
) -> None:
    crop = crop_square(full_image, center[0], center[1], crop_size, writer.image_size)
    writer.add_sample(
        sample_id=sample_id,
        category=category,
        image=crop,
        question=question,
        answer=answer,
        target=target,
        source={
            "kind": "local_board_crop",
            "full_image_size": list(full_image.size),
            "crop_size": crop_size,
        },
        contract_path=contract_path,
    )
