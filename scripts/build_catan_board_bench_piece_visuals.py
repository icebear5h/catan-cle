"""Build isolated and cropped Catan visual-primitive QA examples.

This is criterion-1 data: can the model see Catan primitives before we ask it to
bind them to atlas ids. It composes examples from the same frontend assets used
by the Python board renderer, plus local board crops from the approved dummy
fixture.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from evals.catan_board_bench.annotations import annotation_payload_for_contract
from evals.catan_board_bench.render import (
    DEFAULT_RENDER_STYLE,
    WATER_RGB,
    _asset_path,
    _raster_asset,
    render_contract_image,
)
from evals.catan_board_bench.tokens import color_token, edge_token, node_token, resource_token, tile_token


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "generated" / "catan_board_bench" / "piece_recognition"
)
DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "artifacts"
    / "fixtures"
    / "sft"
    / "render_contracts"
    / "colonist_dummy_setup.json"
)
DEFAULT_PROMPT_PREFIX = "Answer exactly using Catan visual tokens. Do not explain."

RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
NUMBERS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
COLORS = ["RED", "BLUE", "ORANGE", "WHITE", "BLACK"]
PORTS = [None, "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]


class DatasetWriter:
    def __init__(self, output_dir: Path, image_size: int, prompt_prefix: str) -> None:
        self.output_dir = output_dir
        self.image_dir = output_dir / "images"
        self.question_dir = output_dir / "questions"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.prompt_prefix = prompt_prefix
        self.manifest_rows: list[dict[str, Any]] = []
        self.qa_rows: list[dict[str, Any]] = []

    def add_sample(
        self,
        *,
        sample_id: str,
        category: str,
        image: Image.Image,
        question: str,
        answer: str,
        target: dict[str, Any],
        source: dict[str, Any] | None = None,
        contract_path: str | None = None,
    ) -> None:
        image_rel = Path("images") / f"{sample_id}.png"
        image.save(self.output_dir / image_rel)
        qa_id = f"{sample_id}_q00_{category}"
        self.manifest_rows.append(
            {
                "sample_id": sample_id,
                "category": category,
                "image_path": str(image_rel),
                "source": source or {"kind": "frontend_asset_composition"},
                "target": target,
            }
        )
        self.qa_rows.append(
            {
                "id": qa_id,
                "sample_id": sample_id,
                "category": category,
                "image_path": str(image_rel),
                "contract_path": contract_path,
                "question": question,
                "answer": answer,
                "target": target,
                "scoring": "exact",
            }
        )

    def write(self, metadata: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.question_dir.mkdir(parents=True, exist_ok=True)

        write_jsonl(self.output_dir / "manifest.jsonl", self.manifest_rows)
        write_jsonl(self.question_dir / "qa.jsonl", self.qa_rows)
        write_jsonl(
            self.question_dir / "questions.jsonl",
            [
                {
                    "id": row["id"],
                    "sample_id": row["sample_id"],
                    "image_path": row["image_path"],
                    "contract_path": row["contract_path"],
                    "category": row["category"],
                    "question": row["question"],
                }
                for row in self.qa_rows
            ],
        )
        write_jsonl(
            self.question_dir / "answer_key.jsonl",
            [
                {
                    "id": row["id"],
                    "sample_id": row["sample_id"],
                    "category": row["category"],
                    "answer": row["answer"],
                    "target": row["target"],
                    "scoring": row["scoring"],
                }
                for row in self.qa_rows
            ],
        )
        write_jsonl(
            self.output_dir / "messages.jsonl",
            [self._message_row(row) for row in self.qa_rows],
        )

        (self.output_dir / "README.md").write_text(
            "# Isolated Catan Visuals\n\n"
            "Primitive visual-grounding examples composed from frontend assets and "
            "local board crops.\n\n"
            "Pillow is used for composition only. SVG/PNG art is loaded from "
            "`playground/frontend/public/assets`.\n\n"
            f"Samples: {len(self.manifest_rows)}\n"
            f"QA rows: {len(self.qa_rows)}\n",
        )
        write_json(self.output_dir / "metadata.json", metadata)

    def _message_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "image": row["image_path"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": f"{self.prompt_prefix}\n\nQuestion: {row['question']}",
                        },
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]},
            ],
            "metadata": {
                "phase": "isolated_visual_grounding",
                "category": row["category"],
                "sample_id": row["sample_id"],
                "target": row["target"],
            },
        }


def build_dataset(
    *,
    output_dir: Path,
    contract_path: Path,
    image_size: int,
    variants: int,
    prompt_prefix: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = DatasetWriter(output_dir, image_size, prompt_prefix)

    for variant in range(variants):
        add_tile_examples(writer, variant)
        add_road_examples(writer, variant)
        add_node_examples(writer, variant)
        add_port_examples(writer, variant)
        add_robber_examples(writer, variant)
    add_local_patch_examples(writer, contract_path)

    counts = Counter(row["category"] for row in writer.qa_rows)
    metadata = {
        "schema": "catan_isolated_visual_grounding/v1",
        "name": "isolated_visuals",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset_root": "playground/frontend/public/assets",
        "image_size": [image_size, image_size],
        "variants": variants,
        "sample_count": len(writer.manifest_rows),
        "qa_count": len(writer.qa_rows),
        "category_counts": dict(sorted(counts.items())),
        "colors": COLORS,
        "files": {
            "manifest": "manifest.jsonl",
            "qa": "questions/qa.jsonl",
            "questions": "questions/questions.jsonl",
            "answer_key": "questions/answer_key.jsonl",
            "messages": "messages.jsonl",
            "images_dir": "images",
        },
    }
    writer.write(metadata)
    return metadata


def add_tile_examples(writer: DatasetWriter, variant: int) -> None:
    for resource in RESOURCES:
        for number in NUMBERS:
            sample_id = f"isolated_tile_{resource.lower()}_{number}_v{variant:02d}"
            image, tile_bbox, number_bbox = render_tile_image(
                resource, number, writer.image_size, variant
            )
            writer.add_sample(
                sample_id=sample_id,
                category="isolated_tile_resource_number",
                image=image,
                question="What resource and dice number are shown on this tile?",
                answer=f"{resource_token(resource)} {number}",
                target={
                    "resource": resource,
                    "resource_token": resource_token(resource),
                    "number": number,
                    "tile_bbox": tile_bbox,
                    "number_bbox": number_bbox,
                },
            )

    sample_id = f"isolated_tile_desert_v{variant:02d}"
    image, tile_bbox, _ = render_tile_image(None, None, writer.image_size, variant)
    writer.add_sample(
        sample_id=sample_id,
        category="isolated_tile_resource_number",
        image=image,
        question="What resource and dice number are shown on this tile?",
        answer="<DESERT>",
        target={
            "resource": None,
            "resource_token": "<DESERT>",
            "number": None,
            "tile_bbox": tile_bbox,
        },
    )


def add_road_examples(writer: DatasetWriter, variant: int) -> None:
    for color in [None, *COLORS]:
        color_slug = "empty" if color is None else color.lower()
        sample_id = f"isolated_road_{color_slug}_v{variant:02d}"
        image, road_bbox, angle = render_road_image(color, writer.image_size, variant)
        writer.add_sample(
            sample_id=sample_id,
            category="isolated_road_owner",
            image=image,
            question="Who owns the road shown?",
            answer="EMPTY" if color is None else color_token(color),
            target={
                "color": color,
                "color_token": color_token(color) if color else None,
                "angle": angle,
                "road_bbox": road_bbox,
            },
        )


def add_node_examples(writer: DatasetWriter, variant: int) -> None:
    sample_id = f"isolated_node_empty_v{variant:02d}"
    image, marker_bbox = render_node_image(None, None, writer.image_size, variant)
    writer.add_sample(
        sample_id=sample_id,
        category="isolated_node_occupancy",
        image=image,
        question="What building, if any, is shown on this node?",
        answer="EMPTY",
        target={"color": None, "building": None, "marker_bbox": marker_bbox},
    )
    for color in COLORS:
        for building in ["SETTLEMENT", "CITY"]:
            sample_id = f"isolated_node_{color.lower()}_{building.lower()}_v{variant:02d}"
            image, building_bbox = render_node_image(color, building, writer.image_size, variant)
            writer.add_sample(
                sample_id=sample_id,
                category="isolated_node_occupancy",
                image=image,
                question="What building, if any, is shown on this node?",
                answer=f"{color_token(color)} <{building}>",
                target={
                    "color": color,
                    "color_token": color_token(color),
                    "building": building,
                    "building_token": f"<{building}>",
                    "building_bbox": building_bbox,
                },
            )


def add_port_examples(writer: DatasetWriter, variant: int) -> None:
    for resource in PORTS:
        slug = "generic" if resource is None else resource.lower()
        sample_id = f"isolated_port_{slug}_v{variant:02d}"
        image, port_bbox = render_port_image(resource, writer.image_size, variant)
        writer.add_sample(
            sample_id=sample_id,
            category="isolated_port_trade_type",
            image=image,
            question="What trade port is shown?",
            answer="GENERIC 3:1" if resource is None else f"{resource_token(resource)} 2:1",
            target={
                "resource": resource,
                "resource_token": resource_token(resource) if resource else None,
                "ratio": "3:1" if resource is None else "2:1",
                "port_bbox": port_bbox,
            },
        )


def add_robber_examples(writer: DatasetWriter, variant: int) -> None:
    for resource in [None, *RESOURCES]:
        slug = "desert" if resource is None else resource.lower()
        for robber in [True, False]:
            sample_id = f"isolated_robber_{'on' if robber else 'absent'}_{slug}_v{variant:02d}"
            image, tile_bbox, robber_bbox = render_robber_image(
                resource, robber, writer.image_size, variant
            )
            writer.add_sample(
                sample_id=sample_id,
                category="isolated_robber_presence",
                image=image,
                question="Is the robber shown on this tile?",
                answer="YES" if robber else "NO",
                target={
                    "resource": resource,
                    "resource_token": resource_token(resource),
                    "robber": robber,
                    "tile_bbox": tile_bbox,
                    "robber_bbox": robber_bbox,
                },
            )


def add_local_patch_examples(writer: DatasetWriter, contract_path: Path) -> None:
    contract = json.loads(contract_path.read_text())
    full_image_size = 1024
    full_image = render_contract_image(
        contract, image_size=full_image_size, style=DEFAULT_RENDER_STYLE
    )
    annotation_payload = annotation_payload_for_contract(contract, image_size=full_image_size)
    annotations = annotation_payload["annotations"]
    by_kind_token = {(ann["kind"], ann["token"]): ann for ann in annotations}

    nodes_by_id = {node["id"]: node for node in contract["nodes"]}
    for node_id in [1, 10, 14, 17, 22, 25, 33, 0, 52]:
        token = node_token(node_id)
        ann = by_kind_token.get(("node", token))
        if ann is None:
            continue
        node = nodes_by_id[node_id]
        answer = "EMPTY"
        if node.get("building") and node.get("color"):
            answer = f"{color_token(node['color'])} <{node['building']}>"
        add_crop_sample(
            writer,
            full_image,
            ann["center"],
            crop_size=180,
            sample_id=f"local_patch_node_{node_id:02d}",
            category="local_patch_node_occupancy",
            question="What building, if any, is centered in this crop?",
            answer=answer,
            target={
                "node": token,
                "color": node.get("color"),
                "building": node.get("building"),
                "source_center": ann["center"],
            },
            contract_path=str(contract_path.relative_to(PROJECT_ROOT)),
        )

    edges_by_id = {tuple(edge["id"]): edge for edge in contract["edges"]}
    for edge in [(0, 1), (22, 49), (24, 25), (1, 2), (48, 49), (10, 29)]:
        token = edge_token(edge)
        ann = by_kind_token.get(("edge", token))
        if ann is None:
            continue
        edge_payload = edges_by_id[tuple(edge)]
        color = edge_payload.get("road_color")
        add_crop_sample(
            writer,
            full_image,
            ann["center"],
            crop_size=170,
            sample_id=f"local_patch_edge_{edge[0]:02d}_{edge[1]:02d}",
            category="local_patch_edge_road_owner",
            question="Who owns the centered road segment?",
            answer="EMPTY" if color is None else color_token(color),
            target={
                "edge": token,
                "road_color": color,
                "source_center": ann["center"],
            },
            contract_path=str(contract_path.relative_to(PROJECT_ROOT)),
        )

    tiles_by_id = {tile["id"]: tile for tile in contract["tiles"]}
    for tile_id in [15, 16, 1, 5, 7]:
        token = tile_token(tile_id)
        ann = by_kind_token.get(("tile", token))
        if ann is None:
            continue
        tile = tiles_by_id[tile_id]
        if tile["resource"] is None:
            answer = "<DESERT>"
        else:
            answer = f"{resource_token(tile['resource'])} {tile['number']}"
        add_crop_sample(
            writer,
            full_image,
            ann["center"],
            crop_size=190,
            sample_id=f"local_patch_tile_{tile_id:02d}",
            category="local_patch_tile_resource_number",
            question="What resource and dice number are shown on the centered tile?",
            answer=answer,
            target={
                "tile": token,
                "resource": tile["resource"],
                "number": tile["number"],
                "source_center": ann["center"],
            },
            contract_path=str(contract_path.relative_to(PROJECT_ROOT)),
        )

    ports_by_id = {port["id"]: port for port in contract["ports"]}
    for port_id in [0, 1, 3, 4, 7]:
        token = f"<P{port_id:02d}>"
        ann = by_kind_token.get(("port", token))
        if ann is None:
            continue
        port = ports_by_id[port_id]
        resource = port.get("resource")
        answer = "GENERIC 3:1" if resource is None else f"{resource_token(resource)} 2:1"
        add_crop_sample(
            writer,
            full_image,
            ann["center"],
            crop_size=180,
            sample_id=f"local_patch_port_{port_id:02d}",
            category="local_patch_port_trade_type",
            question="What trade port is visible in this crop?",
            answer=answer,
            target={
                "port": token,
                "resource": resource,
                "ratio": "3:1" if resource is None else "2:1",
                "source_center": ann["center"],
            },
            contract_path=str(contract_path.relative_to(PROJECT_ROOT)),
        )

    for tile_id, answer in [(1, "YES"), (15, "NO")]:
        token = tile_token(tile_id)
        ann = by_kind_token.get(("tile", token))
        if ann is None:
            continue
        add_crop_sample(
            writer,
            full_image,
            ann["center"],
            crop_size=210,
            sample_id=f"local_patch_robber_{tile_id:02d}",
            category="local_patch_robber_presence",
            question="Is the robber visible in this crop?",
            answer=answer,
            target={
                "tile": token,
                "robber": answer == "YES",
                "source_center": ann["center"],
            },
            contract_path=str(contract_path.relative_to(PROJECT_ROOT)),
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
    target: dict[str, Any],
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
            "full_image_size": full_image.size,
            "crop_size": crop_size,
        },
        contract_path=contract_path,
    )


def render_tile_image(
    resource: str | None,
    number: int | None,
    image_size: int,
    variant: int,
) -> tuple[Image.Image, list[int], list[int] | None]:
    canvas = base_canvas(image_size)
    tile_size = int(image_size * 0.72)
    tile_x = int((image_size - tile_size) / 2 + variant_offset(variant)[0])
    tile_y = int((image_size - tile_size) / 2 + variant_offset(variant)[1])
    asset_name = "desert" if resource is None else resource.lower()
    tile_bbox = paste_asset(
        canvas, _asset_path(f"/assets/tiles/{asset_name}.svg"), tile_x, tile_y, tile_size, tile_size
    )
    number_bbox = None
    if number is not None:
        number_size = int(image_size * 0.23)
        number_x = int(image_size / 2 - number_size / 2)
        number_y = int(image_size / 2 - number_size / 2 + image_size * 0.06)
        number_bbox = paste_asset(
            canvas,
            _asset_path(f"/assets/numbers/{number}.svg"),
            number_x,
            number_y,
            number_size,
            number_size,
        )
    return canvas.convert("RGB"), tile_bbox, number_bbox


def render_road_image(
    color: str | None, image_size: int, variant: int
) -> tuple[Image.Image, list[int] | None, int]:
    canvas = base_canvas(image_size)
    draw = ImageDraw.Draw(canvas, "RGBA")
    angle = 120 if variant % 2 == 0 else 60
    if color is None:
        draw.line(
            [(image_size * 0.25, image_size * 0.55), (image_size * 0.75, image_size * 0.45)],
            fill=(230, 230, 220, 85),
            width=max(4, image_size // 28),
        )
        return canvas.convert("RGB"), None, angle

    width = int(image_size * 0.28)
    height = int(image_size * 0.62)
    asset = _raster_asset(_asset_path(f"/assets/pieces/road_{color.lower()}.svg"), width, height)
    rotated = asset.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)
    x = int(image_size / 2 - rotated.width / 2)
    y = int(image_size / 2 - rotated.height / 2)
    canvas.alpha_composite(rotated, (x, y))
    return canvas.convert("RGB"), [x, y, x + rotated.width, y + rotated.height], angle


def render_node_image(
    color: str | None,
    building: str | None,
    image_size: int,
    variant: int,
) -> tuple[Image.Image, list[int]]:
    canvas = base_canvas(image_size)
    draw = ImageDraw.Draw(canvas, "RGBA")
    center = (
        image_size // 2 + variant_offset(variant)[0],
        image_size // 2 + variant_offset(variant)[1],
    )
    marker_radius = int(image_size * 0.12)
    marker_bbox = [
        center[0] - marker_radius,
        center[1] - marker_radius,
        center[0] + marker_radius,
        center[1] + marker_radius,
    ]
    draw.ellipse(marker_bbox, fill=(240, 224, 115, 95), outline=(92, 86, 30, 180), width=3)
    if not color or not building:
        return canvas.convert("RGB"), marker_bbox

    size = int(image_size * (0.36 if building == "SETTLEMENT" else 0.40))
    width = int(size * (1.12 if building == "CITY" else 1.0))
    x = int(center[0] - width / 2)
    y = int(center[1] - size * 0.72)
    bbox = paste_asset(
        canvas,
        _asset_path(f"/assets/pieces/{building.lower()}_{color.lower()}.svg"),
        x,
        y,
        width,
        size,
    )
    return canvas.convert("RGB"), bbox


def render_port_image(
    resource: str | None, image_size: int, variant: int
) -> tuple[Image.Image, list[int]]:
    canvas = base_canvas(image_size)
    slug = "generic" if resource is None else resource.lower()
    size = int(image_size * 0.52)
    x = int(image_size / 2 - size / 2 + variant_offset(variant)[0])
    y = int(image_size / 2 - size / 2 + variant_offset(variant)[1])
    bbox = paste_asset(canvas, _asset_path(f"/assets/tiles/port_{slug}.svg"), x, y, size, size)
    return canvas.convert("RGB"), bbox


def render_robber_image(
    resource: str | None,
    robber: bool,
    image_size: int,
    variant: int,
) -> tuple[Image.Image, list[int], list[int] | None]:
    canvas, tile_bbox, _ = render_tile_image(
        resource, None if resource is None else 8, image_size, variant
    )
    canvas = canvas.convert("RGBA")
    robber_bbox = None
    if robber:
        size = int(image_size * 0.28)
        x = int(image_size * 0.52 + variant_offset(variant)[0])
        y = int(image_size * 0.52 + variant_offset(variant)[1])
        robber_bbox = paste_asset(
            canvas, _asset_path("/assets/pieces/robber.svg"), x, y, size, size
        )
    return canvas.convert("RGB"), tile_bbox, robber_bbox


def base_canvas(image_size: int) -> Image.Image:
    canvas = Image.new("RGBA", (image_size, image_size), (*WATER_RGB, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    step = max(16, image_size // 8)
    for pos in range(0, image_size + 1, step):
        draw.line([(pos, 0), (pos, image_size)], fill=(255, 255, 255, 26), width=1)
        draw.line([(0, pos), (image_size, pos)], fill=(255, 255, 255, 26), width=1)
    return canvas


def paste_asset(
    canvas: Image.Image, path: Path, x: int, y: int, width: int, height: int
) -> list[int]:
    asset = _raster_asset(path, width, height)
    canvas.alpha_composite(asset, (x, y))
    return [x, y, x + width, y + height]


def crop_square(
    image: Image.Image, center_x: int, center_y: int, crop_size: int, output_size: int
) -> Image.Image:
    half = crop_size // 2
    left = center_x - half
    top = center_y - half
    right = left + crop_size
    bottom = top + crop_size
    crop = Image.new("RGB", (crop_size, crop_size), WATER_RGB)
    source_left = max(left, 0)
    source_top = max(top, 0)
    source_right = min(right, image.width)
    source_bottom = min(bottom, image.height)
    paste_x = source_left - left
    paste_y = source_top - top
    crop.paste(
        image.crop((source_left, source_top, source_right, source_bottom)), (paste_x, paste_y)
    )
    return crop.resize((output_size, output_size), Image.Resampling.LANCZOS)


def variant_offset(variant: int) -> tuple[int, int]:
    offsets = [(-2, 3), (3, -2), (0, 0), (-4, -1)]
    return offsets[variant % len(offsets)]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--prompt-prefix", default=DEFAULT_PROMPT_PREFIX)
    args = parser.parse_args()

    metadata = build_dataset(
        output_dir=args.output_dir.resolve(),
        contract_path=args.contract.resolve(),
        image_size=args.image_size,
        variants=args.variants,
        prompt_prefix=args.prompt_prefix,
    )
    print(f"wrote_samples={metadata['sample_count']}")
    print(f"wrote_qa={metadata['qa_count']}")
    print("categories=" + ",".join(f"{k}:{v}" for k, v in metadata["category_counts"].items()))
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
