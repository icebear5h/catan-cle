"""Generate a balanced visual probe for one-hop hex-grid directions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from game_engine.models.coordinate_system import Direction, UNIT_VECTORS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_pipeline/catan_board_bench/datasets/hex_direction_probe"
IMAGE_SIZE = 512
LAYOUT_COUNT = 6
ANCHOR_LABEL = "X"
CANDIDATE_LABELS = ("1", "2", "3", "4", "5", "6")

SCREEN_DIRECTIONS = {
    Direction.WEST: "LEFT",
    Direction.EAST: "RIGHT",
    Direction.NORTHWEST: "UP-LEFT",
    Direction.NORTHEAST: "UP-RIGHT",
    Direction.SOUTHWEST: "DOWN-LEFT",
    Direction.SOUTHEAST: "DOWN-RIGHT",
}
DIRECTION_ORDER = (
    Direction.WEST,
    Direction.EAST,
    Direction.NORTHWEST,
    Direction.NORTHEAST,
    Direction.SOUTHWEST,
    Direction.SOUTHEAST,
)

_PALETTES = (
    ((29, 78, 216), (234, 88, 12), (22, 163, 74), (147, 51, 234), (202, 138, 4), (219, 39, 119)),
    ((3, 105, 161), (190, 24, 93), (101, 163, 13), (194, 65, 12), (124, 58, 237), (13, 148, 136)),
)
_OFFSETS = ((0, 0), (8, -5), (-7, 6), (5, 7), (-6, -7), (3, -2))


def cube_to_pixel(
    coordinate: tuple[int, int, int],
    *,
    center: tuple[float, float],
    radius: float,
) -> tuple[float, float]:
    """Project engine cube coordinates with the frontend pointy-top convention."""
    q = coordinate[0]
    r = coordinate[2]
    return (
        center[0] + math.sqrt(3) * radius * (q + r / 2),
        center[1] + 1.5 * radius * r,
    )


def label_assignment(layout_index: int) -> dict[Direction, str]:
    """Use a Latin-square shift so every label occupies every direction once."""
    return {
        direction: CANDIDATE_LABELS[(direction_index + layout_index) % 6]
        for direction_index, direction in enumerate(DIRECTION_ORDER)
    }


def build_probe(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    image_size: int = IMAGE_SIZE,
    layout_count: int = LAYOUT_COUNT,
) -> dict[str, Any]:
    if layout_count != len(CANDIDATE_LABELS):
        raise ValueError("Exactly six layouts are required for Latin-square balance")

    image_dir = output_dir / "images"
    question_dir = output_dir / "questions"
    image_dir.mkdir(parents=True, exist_ok=True)
    question_dir.mkdir(parents=True, exist_ok=True)

    qa_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    direction_answers: Counter[str] = Counter()
    label_answers: Counter[str] = Counter()
    label_positions: Counter[tuple[str, str]] = Counter()

    for layout_index in range(layout_count):
        assignment = label_assignment(layout_index)
        sample_id = f"hex_layout_{layout_index:02d}"
        image, centers = render_layout(
            assignment,
            layout_index=layout_index,
            image_size=image_size,
        )
        image_path = Path("images") / f"{sample_id}.png"
        image.save(output_dir / image_path)

        manifest_rows.append(
            {
                "sample_id": sample_id,
                "category": "hex_direction_probe",
                "image_path": str(image_path),
                "source": {
                    "kind": "synthetic_engine_cube_projection",
                    "layout_index": layout_index,
                },
                "anchor": {
                    "label": ANCHOR_LABEL,
                    "cube_coordinate": [0, 0, 0],
                    "pixel_center": centers["anchor"],
                },
                "neighbors": [
                    {
                        "label": assignment[direction],
                        "direction": SCREEN_DIRECTIONS[direction],
                        "engine_direction": direction.value,
                        "cube_delta": list(UNIT_VECTORS[direction]),
                        "pixel_center": centers[direction.value],
                    }
                    for direction in DIRECTION_ORDER
                ],
            }
        )

        for direction_index, direction in enumerate(DIRECTION_ORDER):
            screen_direction = SCREEN_DIRECTIONS[direction]
            label = assignment[direction]
            common_target = {
                "layout_index": layout_index,
                "anchor_label": ANCHOR_LABEL,
                "neighbor_label": label,
                "direction": screen_direction,
                "engine_direction": direction.value,
                "cube_delta": list(UNIT_VECTORS[direction]),
                "anchor_pixel_center": centers["anchor"],
                "neighbor_pixel_center": centers[direction.value],
            }
            qa_rows.append(
                {
                    "id": f"{sample_id}_q{direction_index:02d}_direction_to_label",
                    "sample_id": sample_id,
                    "category": "isolated_hex_direction_to_label",
                    "image_path": str(image_path),
                    "contract_path": None,
                    "question": (
                        f"Which numbered hex is directly {screen_direction} of the "
                        f"center hex {ANCHOR_LABEL}? Answer only its number."
                    ),
                    "answer": label,
                    "target": common_target,
                    "scoring": "hex_direction",
                }
            )
            qa_rows.append(
                {
                    "id": f"{sample_id}_q{direction_index + 6:02d}_label_to_direction",
                    "sample_id": sample_id,
                    "category": "isolated_hex_label_to_direction",
                    "image_path": str(image_path),
                    "contract_path": None,
                    "question": (
                        f"Where is hex {label} relative to the center hex "
                        f"{ANCHOR_LABEL}? Answer exactly one of: LEFT, RIGHT, "
                        "UP-LEFT, UP-RIGHT, DOWN-LEFT, DOWN-RIGHT."
                    ),
                    "answer": screen_direction,
                    "target": common_target,
                    "scoring": "hex_direction",
                }
            )
            direction_answers[screen_direction] += 1
            label_answers[label] += 1
            label_positions[(label, screen_direction)] += 1

    _validate_balance(
        qa_rows,
        direction_answers=direction_answers,
        label_answers=label_answers,
        label_positions=label_positions,
    )

    write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    write_jsonl(question_dir / "qa.jsonl", qa_rows)
    write_jsonl(
        question_dir / "questions.jsonl",
        [
            {
                key: row[key]
                for key in (
                    "id",
                    "sample_id",
                    "category",
                    "image_path",
                    "contract_path",
                    "question",
                )
            }
            for row in qa_rows
        ],
    )
    write_jsonl(
        question_dir / "answer_key.jsonl",
        [
            {
                key: row[key]
                for key in ("id", "sample_id", "category", "answer", "target", "scoring")
            }
            for row in qa_rows
        ],
    )

    metadata = {
        "schema": "catan_hex_direction_probe/v1",
        "name": "hex_direction_probe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "image_size": [image_size, image_size],
        "layout_count": layout_count,
        "image_count": len(manifest_rows),
        "qa_count": len(qa_rows),
        "categories": dict(Counter(row["category"] for row in qa_rows)),
        "directions": [SCREEN_DIRECTIONS[direction] for direction in DIRECTION_ORDER],
        "chance_accuracy": 1 / 6,
        "atlas_prompt": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output_dir / "README.md").write_text(
        "# Hex Direction Probe\n\n"
        "A balanced visual diagnostic for one-hop pointy-top hex directions. "
        "Every candidate label occupies every direction exactly once. The two "
        "question types test direction-to-label and label-to-direction mapping. "
        "No board atlas or contract is supplied.\n\n"
        "For a full balanced probe, run the evaluator with `--suite probe`, both "
        "hex categories, and `--limit-samples 0`; positive limits are applied per "
        "category and may select an unbalanced prefix.\n"
    )
    return metadata


def render_layout(
    assignment: dict[Direction, str],
    *,
    layout_index: int,
    image_size: int,
) -> tuple[Image.Image, dict[str, list[int]]]:
    canvas = Image.new("RGB", (image_size, image_size), (8, 31, 52))
    draw = ImageDraw.Draw(canvas)
    offset = _OFFSETS[layout_index % len(_OFFSETS)]
    center = (image_size / 2 + offset[0], image_size / 2 + offset[1])
    radius = image_size * (0.142 + 0.003 * (layout_index % 3))
    label_font = load_font(max(28, int(radius * 0.58)))
    caption_font = load_font(max(15, int(radius * 0.25)))
    palette = _PALETTES[layout_index % len(_PALETTES)]
    centers: dict[str, list[int]] = {"anchor": [round(center[0]), round(center[1])]}

    draw.text(
        (image_size / 2, 18),
        "ONE-HOP HEX NEIGHBORS",
        font=caption_font,
        fill=(196, 225, 243),
        anchor="ma",
    )

    for direction_index, direction in enumerate(DIRECTION_ORDER):
        pixel_center = cube_to_pixel(
            UNIT_VECTORS[direction],
            center=center,
            radius=radius,
        )
        centers[direction.value] = [round(pixel_center[0]), round(pixel_center[1])]
        fill = palette[(direction_index * 5 + layout_index * 2) % len(palette)]
        draw_hex(
            draw,
            center=pixel_center,
            radius=radius,
            fill=fill,
            outline=(230, 244, 251),
            width=5,
        )
        draw_label(
            draw,
            pixel_center,
            assignment[direction],
            font=label_font,
            fill=(255, 255, 255),
        )

    draw_hex(
        draw,
        center=center,
        radius=radius,
        fill=(13, 148, 136),
        outline=(255, 255, 255),
        width=7,
    )
    draw_label(
        draw,
        center,
        ANCHOR_LABEL,
        font=label_font,
        fill=(255, 255, 255),
    )
    return canvas, centers


def draw_hex(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    radius: float,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    width: int,
) -> None:
    points = [
        (
            center[0] + radius * math.cos(math.radians(-90 + 60 * vertex)),
            center[1] + radius * math.sin(math.radians(-90 + 60 * vertex)),
        )
        for vertex in range(6)
    ]
    draw.polygon(points, fill=fill, outline=outline, width=width)


def draw_label(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    label: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bbox = draw.textbbox(center, label, font=font, anchor="mm", stroke_width=2)
    pad = 7
    draw.rounded_rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        radius=8,
        fill=(4, 18, 31, 215),
        outline=(255, 255, 255, 190),
        width=2,
    )
    draw.text(
        center,
        label,
        font=font,
        fill=fill,
        anchor="mm",
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _validate_balance(
    qa_rows: list[dict[str, Any]],
    *,
    direction_answers: Counter[str],
    label_answers: Counter[str],
    label_positions: Counter[tuple[str, str]],
) -> None:
    expected_directions = set(SCREEN_DIRECTIONS.values())
    assert len(qa_rows) == 72
    assert set(direction_answers) == expected_directions
    assert set(direction_answers.values()) == {6}
    assert set(label_answers) == set(CANDIDATE_LABELS)
    assert set(label_answers.values()) == {6}
    assert len(label_positions) == 36
    assert set(label_positions.values()) == {1}
    assert all(row["contract_path"] is None for row in qa_rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()
    metadata = build_probe(args.output_dir, image_size=args.image_size)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
