"""Render Catan contract datasets with the Python HexBoard-compatible renderer.

This is the fast path after contracts exist. It does not use Playwright; it
uses ``evals.catan_board_bench.render`` to mirror frontend geometry/assets.

For safety, it writes image-backed copies of QA/message files instead of
mutating the original contract-only rows.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TextIO, cast

from evals.catan_board_bench.render import RenderStyle, render_contract_image
from sft.json_types import JsonDict, JsonLike, JsonLikeDict, JsonValue, as_dict, as_str
from sft.paths import GENERATED_SFT_ROOT, RENDERER_STYLE_CONFIG, repository_relative_path

CURRICULUM_STAGE = "phase_1_post_atlas_visual_grounding"
REQUIRES_STAGE = "phase_0_text_atlas_topology"
DATASET_ROLE = "bind_stable_node_tokens_to_transient_visual_facts"
CATEGORY_GROUPS = {
    "board_atlas_bboxes": "full_board_atlas_bbox_map",
    "node_bbox": "atlas_coordinate_grounding",
    "node_occupancy": "transient_node_state_readout",
    "node_adjacent_tile_resource_numbers": "local_tile_readout",
    "node_incident_road_owners": "local_edge_readout",
    "node_local_state_json": "local_state_composition",
}


def category_group(category: str) -> str:
    return CATEGORY_GROUPS.get(category, "uncategorized")


def iter_jsonl(path: Path) -> Iterator[tuple[int, JsonValue]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def write_jsonl(handle: TextIO, value: Mapping[str, JsonLike]) -> None:
    handle.write(json.dumps(value, sort_keys=True) + "\n")


def question_view(qa: JsonDict) -> JsonLikeDict:
    return {
        "id": qa["id"],
        "sample_id": qa["sample_id"],
        "image_path": qa["image_path"],
        "contract_path": qa["contract_path"],
        "category": qa["category"],
        "category_group": qa.get("category_group", category_group(as_str(qa["category"]))),
        "curriculum_stage": qa.get("curriculum_stage", CURRICULUM_STAGE),
        "requires_stage": qa.get("requires_stage", REQUIRES_STAGE),
        "question": qa["question"],
    }


def message_view(qa: JsonDict, *, prompt_prefix: str) -> JsonLikeDict:
    return {
        "id": qa["id"],
        "image": qa["image_path"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{prompt_prefix}\n\nQuestion: {qa['question']}"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": qa["answer"]}]},
        ],
        "metadata": {
            "phase": qa.get("curriculum_stage", CURRICULUM_STAGE),
            "requires_stage": qa.get("requires_stage", REQUIRES_STAGE),
            "dataset_role": DATASET_ROLE,
            "category": qa["category"],
            "category_group": qa.get("category_group",
                                     category_group(as_str(qa["category"]))),
            "sample_id": qa["sample_id"],
            "contract_path": qa["contract_path"],
            "image_path": qa["image_path"],
            "target": qa.get("target"),
        },
    }


def load_json(path: Path) -> JsonDict:
    return as_dict(json.loads(path.read_text()))


def write_json(path: Path, value: Mapping[str, JsonLike]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_render_style(path: Path | None) -> RenderStyle | None:
    if path is None:
        return None
    payload = load_json(path)
    style_payload = as_dict(payload.get("style", payload))
    style_keys = RenderStyle.__dataclass_fields__.keys()
    return RenderStyle(**{key: cast("float", style_payload[key])
                          for key in style_keys if key in style_payload})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=GENERATED_SFT_ROOT / "node_factors",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument(
        "--style-config",
        type=Path,
        default=RENDERER_STYLE_CONFIG,
        help="Renderer style JSON saved by the tuning UI. If missing, built-in defaults are used.",
    )
    parser.add_argument(
        "--prompt-prefix",
        default="Answer exactly using Catan atlas tokens. Do not explain.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    manifest_path = dataset_dir / "manifest.jsonl"
    qa_path = dataset_dir / "questions/qa.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not qa_path.exists():
        raise FileNotFoundError(qa_path)

    style = load_render_style(args.style_config if args.style_config.exists() else None)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rendered: dict[str, str] = {}
    image_manifest_path = dataset_dir / "manifest_with_images.jsonl"
    with image_manifest_path.open("w") as manifest_out:
        for index, (_, entry) in enumerate(iter_jsonl(manifest_path)):
            if args.limit_samples is not None and index >= args.limit_samples:
                break
            row = as_dict(entry)
            sample_id = as_str(row["sample_id"])
            contract_rel = Path(as_str(row["contract_path"]))
            contract = load_json(dataset_dir / contract_rel)
            image_rel = Path("images") / f"{sample_id}.png"
            image = render_contract_image(contract, image_size=args.image_size, style=style)
            image.save(dataset_dir / image_rel)
            rendered[sample_id] = str(image_rel)
            row_with_image: JsonDict = dict(row)
            row_with_image["image_path"] = str(image_rel)
            if isinstance(row_with_image.get("source"), dict):
                source = dict(as_dict(row_with_image["source"]))
                source["render_status"] = "rendered_python_hexboard"
                row_with_image["source"] = source
            row_with_image["render"] = {
                "kind": "python_catan_hexboard_renderer",
                "image_size": [args.image_size, args.image_size],
                "style_config": (repository_relative_path(args.style_config) if style else None),
            }
            write_jsonl(manifest_out, row_with_image)

    qa_out_path = dataset_dir / "questions/qa_with_images.jsonl"
    questions_out_path = dataset_dir / "questions/questions_with_images.jsonl"
    messages_out_path = dataset_dir / "messages_with_images.jsonl"
    qa_rows = 0
    with (
        qa_out_path.open("w") as qa_out,
        questions_out_path.open("w") as questions_out,
        messages_out_path.open("w") as messages_out,
    ):
        for _, entry in iter_jsonl(qa_path):
            qa = as_dict(entry)
            image_path = rendered.get(as_str(qa["sample_id"]))
            if image_path is None:
                continue
            qa = dict(qa)
            qa["image_path"] = image_path
            write_jsonl(qa_out, qa)
            write_jsonl(questions_out, question_view(qa))
            write_jsonl(messages_out, message_view(qa, prompt_prefix=args.prompt_prefix))
            qa_rows += 1

    metadata = {
        "schema": "catan_rendered_contract_dataset/v0",
        "curriculum_stage": CURRICULUM_STAGE,
        "requires_stage": REQUIRES_STAGE,
        "dataset_role": DATASET_ROLE,
        "category_groups": CATEGORY_GROUPS,
        "dataset_dir": repository_relative_path(dataset_dir),
        "image_size": [args.image_size, args.image_size],
        "style_config": (repository_relative_path(args.style_config) if style else None),
        "rendered_samples": len(rendered),
        "qa_rows": qa_rows,
        "files": {
            "manifest": image_manifest_path.name,
            "qa": "questions/qa_with_images.jsonl",
            "questions": "questions/questions_with_images.jsonl",
            "messages": messages_out_path.name,
            "images_dir": "images",
        },
    }
    write_json(dataset_dir / "render_metadata.json", metadata)

    print(f"rendered_samples={len(rendered)}")
    print(f"wrote_qa={qa_rows}")
    print(messages_out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
