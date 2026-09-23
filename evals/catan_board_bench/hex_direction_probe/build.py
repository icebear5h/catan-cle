"""Probe dataset assembly and the per-direction balance check."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from cle.game_engine.models.coordinate_system import UNIT_VECTORS
from evals.catan_board_bench.hex_direction_probe.drawing import render_layout
from evals.catan_board_bench.hex_direction_probe.layout import (
    ANCHOR_LABEL,
    CANDIDATE_LABELS,
    DEFAULT_OUTPUT_DIR,
    DIRECTION_ORDER,
    IMAGE_SIZE,
    LAYOUT_COUNT,
    SCREEN_DIRECTIONS,
    label_assignment,
)
from evals.json_types import JsonDict, as_str


def build_probe(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    image_size: int = IMAGE_SIZE,
    layout_count: int = LAYOUT_COUNT,
) -> JsonDict:
    if layout_count != len(CANDIDATE_LABELS):
        raise ValueError("Exactly six layouts are required for Latin-square balance")

    image_dir = output_dir / "images"
    question_dir = output_dir / "questions"
    image_dir.mkdir(parents=True, exist_ok=True)
    question_dir.mkdir(parents=True, exist_ok=True)

    qa_rows: list[JsonDict] = []
    manifest_rows: list[JsonDict] = []
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
                    "pixel_center": list(centers["anchor"]),
                },
                "neighbors": [
                    {
                        "label": assignment[direction],
                        "direction": SCREEN_DIRECTIONS[direction],
                        "engine_direction": direction.value,
                        "cube_delta": list(UNIT_VECTORS[direction]),
                        "pixel_center": list(centers[direction.value]),
                    }
                    for direction in DIRECTION_ORDER
                ],
            }
        )

        for direction_index, direction in enumerate(DIRECTION_ORDER):
            screen_direction = SCREEN_DIRECTIONS[direction]
            label = assignment[direction]
            common_target: JsonDict = {
                "layout_index": layout_index,
                "anchor_label": ANCHOR_LABEL,
                "neighbor_label": label,
                "direction": screen_direction,
                "engine_direction": direction.value,
                "cube_delta": list(UNIT_VECTORS[direction]),
                "anchor_pixel_center": list(centers["anchor"]),
                "neighbor_pixel_center": list(centers[direction.value]),
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

    metadata: JsonDict = {
        "schema": "catan_hex_direction_probe/v1",
        "name": "hex_direction_probe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "image_size": [image_size, image_size],
        "layout_count": layout_count,
        "image_count": len(manifest_rows),
        "qa_count": len(qa_rows),
        "categories": dict(Counter(as_str(row["category"], "category") for row in qa_rows)),
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



def _validate_balance(
    qa_rows: list[JsonDict],
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


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

