from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

from sft.json_types import JsonDict, JsonLike, JsonList

from ._sources import CURRICULUM_STAGE, DATASET_ROLE, PROMPT_PREFIX, REQUIRES_STAGE


def question_view(qa: JsonDict) -> JsonDict:
    return {
        "id": qa["id"],
        "sample_id": qa["sample_id"],
        "image_path": qa["image_path"],
        "contract_path": qa["contract_path"],
        "category": qa["category"],
        "category_group": qa["category_group"],
        "curriculum_stage": qa["curriculum_stage"],
        "requires_stage": qa["requires_stage"],
        "question": qa["question"],
    }


def answer_view(qa: JsonDict) -> JsonDict:
    return {
        "id": qa["id"],
        "sample_id": qa["sample_id"],
        "category": qa["category"],
        "category_group": qa["category_group"],
        "curriculum_stage": qa["curriculum_stage"],
        "requires_stage": qa["requires_stage"],
        "answer": qa["answer"],
        "target": qa["target"],
        "scoring": qa["scoring"],
    }


def message_view(qa: JsonDict) -> JsonDict:
    content: JsonList = []
    if qa["image_path"]:
        content.append({"type": "image"})
    content.append({"type": "text", "text": f"{PROMPT_PREFIX}\n\nQuestion: {qa['question']}"})
    return {
        "id": qa["id"],
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": qa["answer"]}]},
        ],
        "metadata": {
            "phase": CURRICULUM_STAGE,
            "requires_stage": REQUIRES_STAGE,
            "dataset_role": DATASET_ROLE,
            "category": qa["category"],
            "category_group": qa["category_group"],
            "sample_id": qa["sample_id"],
            "contract_path": qa["contract_path"],
            "image_path": qa["image_path"],
            "target": qa["target"],
        },
    }


def write_json(path: Path, value: Mapping[str, JsonLike]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(handle: TextIO, value: Mapping[str, JsonLike]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")


def write_readme(output_dir: Path, *, sample_count: int, qa_count: int, assume_rendered_images: bool) -> None:
    render_note = (
        "QA rows contain image paths under `images/`, but this script does not render those images."
        if assume_rendered_images
        else "This is contract-only output. QA rows have `image_path: null` until a frontend render step fills images."
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Post-Atlas Node Visual Grounding Dataset",
                "",
                "Controlled post-atlas visual QA for node grounding. This dataset assumes",
                "Phase 0 text atlas topology has already established stable tokens like",
                "`<N11>`, then asks the model to bind those stable node handles to visible",
                "coordinates and transient local board facts.",
                "",
                render_note,
                "",
                "Curriculum:",
                f"- Stage: `{CURRICULUM_STAGE}`",
                f"- Requires: `{REQUIRES_STAGE}`",
                f"- Role: `{DATASET_ROLE}`",
                "",
                "Category groups:",
                "- `atlas_coordinate_grounding`: locate the stable node handle.",
                "- `full_board_atlas_bbox_map`: return all tile/node/edge/port bboxes.",
                "- `transient_node_state_readout`: read occupancy at that handle.",
                "- `local_tile_readout`: read neighboring tile resource/number facts.",
                "- `local_edge_readout`: read road ownership around the node.",
                "- `local_state_composition`: compose the local node state as JSON.",
                "",
                "Files:",
                "- `contracts/`: public board contracts.",
                "- `annotations.jsonl`: frontend-aligned tile/node/edge/port bboxes for each contract.",
                "- `manifest.jsonl`: one row per synthetic sample.",
                "- `questions/qa.jsonl`: QA rows with answers and targets.",
                "- `questions/questions.jsonl`: promptable questions without answers.",
                "- `questions/answer_key.jsonl`: deterministic answer targets.",
                "- `messages.jsonl`: chat-style rows for text-only or pending-image smoke tests.",
                "",
                f"Samples: {sample_count}",
                f"QA rows: {qa_count}",
                "",
            ]
        )
    )
