"""Build post-atlas node visual-grounding data.

The purpose of this dataset is not to imitate real games or teach the atlas from
scratch. It assumes the fixed Catan atlas already exists, then trains visual
readout around stable atlas handles:

    stable token: <N41>
    transient visual state: EMPTY / <RED> <SETTLEMENT> / <BLUE> <CITY> / ...
    varied context: shuffled tile resources, numbers, ports, robber, and roads

This guards against the bad shortcut where a model learns that a node token
permanently "has" a piece. Each atlas node appears under many contradictory
transient states.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from cle.game_engine.models.player import Color
from evals.catan_board_bench.annotations import annotation_payload_for_contract
from evals.catan_board_bench.tokens import (
    atlas_metadata_json,
    building_token,
    color_token,
    node_token,
)
from sft.json_types import as_str, json_path
from sft.paths import GENERATED_SFT_ROOT

from ._contract import build_contract
from ._qas import build_qas
from ._sources import (
    CATEGORY_GROUPS,
    CURRICULUM_STAGE,
    DATASET_NAME,
    DATASET_ROLE,
    DATASET_SCHEMA,
    DEFAULT_COLORS,
    REQUIRES_STAGE,
    build_indices,
    occupancy_cases,
    validate_colors,
)
from ._views import answer_view, message_view, question_view, write_json, write_jsonl, write_readme


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GENERATED_SFT_ROOT / "node_factors",
        help="Directory for generated contracts, annotations, and QA rows.",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        default=list(DEFAULT_COLORS),
        help="Color names to permute over. Defaults to common 4p colors plus BLACK.",
    )
    parser.add_argument(
        "--all-colors",
        action="store_true",
        help="Use every engine Color enum value.",
    )
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument(
        "--variants-per-case",
        type=int,
        default=1,
        help="Number of independent board shuffles for each node/occupancy pair.",
    )
    parser.add_argument("--distractor-buildings", type=int, default=2)
    parser.add_argument("--distractor-roads", type=int, default=4)
    parser.add_argument(
        "--assume-rendered-images",
        action="store_true",
        help="Populate image_path as images/<sample>.png for a later render step.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    colors = [color.value for color in Color] if args.all_colors else validate_colors(list(args.colors))

    atlas = atlas_metadata_json()
    indices = build_indices(atlas)
    cases = occupancy_cases(colors)
    output_dir = args.output_dir
    questions_dir = output_dir / "questions"
    contracts_dir = output_dir / "contracts"
    output_dir.mkdir(parents=True, exist_ok=True)
    questions_dir.mkdir(parents=True, exist_ok=True)
    contracts_dir.mkdir(parents=True, exist_ok=True)
    if args.assume_rendered_images:
        (output_dir / "images").mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    annotations_path = output_dir / "annotations.jsonl"
    qa_path = questions_dir / "qa.jsonl"
    questions_path = questions_dir / "questions.jsonl"
    answer_key_path = questions_dir / "answer_key.jsonl"
    messages_path = output_dir / "messages.jsonl"

    sample_count = 0
    qa_count = 0
    category_counts: Counter[str] = Counter()
    with (
        manifest_path.open("w") as manifest_f,
        annotations_path.open("w") as annotations_f,
        qa_path.open("w") as qa_f,
        questions_path.open("w") as questions_f,
        answer_key_path.open("w") as answer_key_f,
        messages_path.open("w") as messages_f,
    ):
        for node_id in sorted(indices["nodes"]):
            for case in cases:
                if args.limit_samples is not None and sample_count >= args.limit_samples:
                    break
                for variant_index in range(args.variants_per_case):
                    if args.limit_samples is not None and sample_count >= args.limit_samples:
                        break
                    sample_id = f"node_factor_n{node_id:02d}_{case['name']}_v{variant_index:02d}"
                    contract = build_contract(
                        atlas=atlas,
                        indices=indices,
                        sample_id=sample_id,
                        sample_index=sample_count,
                        target_node_id=node_id,
                        occupancy=case,
                        colors=colors,
                        seed=args.seed + sample_count * 1009 + node_id + variant_index * 9173,
                        image_size=args.image_size,
                        distractor_buildings=args.distractor_buildings,
                        distractor_roads=args.distractor_roads,
                        assume_rendered_images=args.assume_rendered_images,
                    )
                    annotations = annotation_payload_for_contract(contract, image_size=args.image_size)
                    qas = build_qas(contract, node_id, annotations)

                    write_json(contracts_dir / f"{sample_id}.json", contract)
                    write_jsonl(annotations_f, annotations)
                    for qa in qas:
                        write_jsonl(qa_f, qa)
                        write_jsonl(questions_f, question_view(qa))
                        write_jsonl(answer_key_f, answer_view(qa))
                        write_jsonl(messages_f, message_view(qa))
                        category_counts[as_str(qa["category"])] += 1
                    qa_count += len(qas)

                    write_jsonl(
                        manifest_f,
                        {
                            "sample_id": sample_id,
                            "contract_path": json_path(contract, "sample", "contract_path"),
                            "image_path": json_path(contract, "sample", "image_path"),
                            "annotation_file": "annotations.jsonl",
                            "question_count": len(qas),
                            "variant_index": variant_index,
                            "target": {
                                "node_id": node_id,
                                "node_token": node_token(node_id),
                                "occupancy": "EMPTY"
                                if case["building"] is None
                                else f"{color_token(str(case['color']))} {building_token(str(case['building']))}",
                            },
                            "source": contract["source"],
                            "curriculum_stage": CURRICULUM_STAGE,
                            "requires_stage": REQUIRES_STAGE,
                            "dataset_role": DATASET_ROLE,
                        },
                    )
                    sample_count += 1
            if args.limit_samples is not None and sample_count >= args.limit_samples:
                break

    metadata = {
        "name": DATASET_NAME,
        "schema": DATASET_SCHEMA,
        "curriculum_stage": CURRICULUM_STAGE,
        "requires_stage": REQUIRES_STAGE,
        "dataset_role": DATASET_ROLE,
        "category_groups": CATEGORY_GROUPS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": sample_count,
        "qa_count": qa_count,
        "image_size": [args.image_size, args.image_size],
        "colors": colors,
        "occupancy_cases": [case["name"] for case in cases],
        "variants_per_case": args.variants_per_case,
        "distractor_buildings": args.distractor_buildings,
        "distractor_roads": args.distractor_roads,
        "category_counts": dict(category_counts),
        "files": {
            "manifest": manifest_path.name,
            "annotations": annotations_path.name,
            "messages": messages_path.name,
            "questions_dir": "questions",
            "qa": "questions/qa.jsonl",
            "questions": "questions/questions.jsonl",
            "answer_key": "questions/answer_key.jsonl",
            "contracts_dir": "contracts",
            "images_dir": "images" if args.assume_rendered_images else None,
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    write_readme(
        output_dir,
        sample_count=sample_count,
        qa_count=qa_count,
        assume_rendered_images=args.assume_rendered_images,
    )

    print(f"wrote_samples={sample_count}")
    print(f"wrote_qa={qa_count}")
    print(f"colors={','.join(colors)}")
    print(f"categories={','.join(sorted(category_counts))}")
    print(output_dir)
    return 0
