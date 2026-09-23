"""Render the strict text probe onto unannotated engine screenshots."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from evals.catan_board_bench.ascii_variations import STRICT_SCORER_VERSION, strict_scorer_digest
from evals.catan_board_bench.paths import (
    canonical_benchmark_reference,
    resolve_benchmark_reference,
)
from evals.catan_board_bench.render import DEFAULT_RENDER_STYLE, render_contract_image
from evals.catan_board_bench.text_format_optimization import DATASET_SCHEMA
from scripts.board_bench.builders.render_catan_strict_vision_probe.constants import (
    OUTPUT_SCHEMA,
    file_sha256,
    json_digest,
)
from scripts.board_bench.builders.render_catan_strict_vision_probe.projection import (
    canonical_id_map,
    canonicalize_question,
    validate_canonical_questions,
)
from scripts.board_bench.builders.render_catan_strict_vision_probe.source import (
    validate_render_args,
    validate_source_dataset,
)
from scripts.board_bench.shapes import (
    JsonDict,
    read_json_object,
    text,
    write_json,
    write_jsonl,
)

__all__ = ["render_strict_vision_probe"]

README_TEMPLATE = (
    "# Strict 60-question raw-vision probe\n\n"
    "This is the raw-image projection of `text_format_optimization_probe`. "
    "The engine public-state contracts are the oracle and are rendered as ordinary "
    "unannotated {image_size}px board screenshots. Text-only opaque aliases are inverted to "
    "canonical engine IDs in the questions and strict answers. The semantic targets, "
    "question IDs, board states, categories, and scorer are unchanged.\n"
)


def render_strict_vision_probe(
    source_dir: Path,
    output_dir: Path,
    *,
    image_size: int,
    view_padding_factor: float,
    target_board_canvas_fraction: float,
) -> JsonDict:
    validate_render_args(
        source_dir,
        output_dir,
        image_size=image_size,
        view_padding_factor=view_padding_factor,
        target_board_canvas_fraction=target_board_canvas_fraction,
    )
    source_metadata, source_manifest, source_questions = validate_source_dataset(source_dir)
    source_manifest_by_sample = {
        text(row["sample_id"], "sample_id"): row for row in source_manifest
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    contract_dir = output_dir / "contracts"
    image_dir.mkdir()
    contract_dir.mkdir()
    style = replace(DEFAULT_RENDER_STYLE, view_padding_factor=view_padding_factor)

    source_locks: dict[str, str] = {
        "metadata.json": file_sha256(source_dir / "metadata.json"),
        "manifest.jsonl": file_sha256(source_dir / "manifest.jsonl"),
        "qa.jsonl": file_sha256(source_dir / "qa.jsonl"),
    }
    canonical_maps: dict[str, dict[str, str]] = {}
    visual_manifest: list[JsonDict] = []

    for row in source_manifest:
        sample_id = text(row["sample_id"], "sample_id")
        source_contract_reference = canonical_benchmark_reference(
            text(row["source_contract"], "source_contract")
        )
        source_contract_path = resolve_benchmark_reference(source_contract_reference)
        alias_path = source_dir / "aliases" / f"{sample_id}.json"
        fact_path = source_dir / "facts" / f"{sample_id}.json"
        contract = read_json_object(source_contract_path)
        aliases = read_json_object(alias_path)
        canonical_maps[sample_id] = canonical_id_map(aliases)

        contract_path = contract_dir / f"{sample_id}.json"
        write_json(contract_path, contract)
        image_path = image_dir / f"{sample_id}.png"
        render_contract_image(contract, image_size=image_size, style=style).save(image_path)

        source_locks[str(source_contract_reference)] = file_sha256(source_contract_path)
        source_locks[f"aliases/{sample_id}.json"] = file_sha256(alias_path)
        source_locks[f"facts/{sample_id}.json"] = file_sha256(fact_path)
        visual_manifest.append(
            {
                "sample_id": sample_id,
                "original_sample_id": row.get("original_sample_id"),
                "source_fact_digest": row["fact_digest"],
                "source_contract": str(source_contract_reference),
                "source_contract_sha256": file_sha256(source_contract_path),
                "contract_path": str(contract_path.relative_to(output_dir)),
                "contract_sha256": file_sha256(contract_path),
                "engine_state_sha256": json_digest(contract),
                "image_path": str(image_path.relative_to(output_dir)),
                "image_sha256": file_sha256(image_path),
                "canonical_id_map_sha256": json_digest(
                    {key: value for key, value in canonical_maps[sample_id].items()}
                ),
            }
        )

    questions = [
        canonicalize_question(
            question,
            canonical_maps[text(question["sample_id"], "sample_id")],
            manifest_row=source_manifest_by_sample[text(question["sample_id"], "sample_id")],
        )
        for question in source_questions
    ]
    validate_canonical_questions(questions)
    write_jsonl(output_dir / "qa.jsonl", questions)
    write_jsonl(output_dir / "manifest.jsonl", visual_manifest)

    metadata: JsonDict = {
        "schema": OUTPUT_SCHEMA,
        "source_dataset_schema": DATASET_SCHEMA,
        "source_dataset": str(source_dir),
        "source_lock": {key: value for key, value in sorted(source_locks.items())},
        "source_lock_sha256": json_digest({key: value for key, value in source_locks.items()}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "board_count": len(visual_manifest),
        "question_count": len(questions),
        "categories": {
            key: count
            for key, count in sorted(
                Counter(text(row["category"], "question category") for row in questions).items()
            )
        },
        "strict_json_answers": True,
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
        "image_size": [image_size, image_size],
        "rendered_images": len(visual_manifest),
        "render_variant": {
            "renderer": "evals.catan_board_bench.render",
            "style": {key: value for key, value in asdict(style).items()},
            "view_padding_factor": view_padding_factor,
            "target_board_canvas_fraction": target_board_canvas_fraction,
            "image_annotation": None,
        },
        "identity_projection": {
            "source": "deterministically permuted board-local text IDs",
            "target": "canonical engine tile/node/edge/port IDs",
            "method": "invert source aliases before rendering/evaluation",
            "image_contains_entity_labels": False,
        },
        "source_metadata_sha256": json_digest(source_metadata),
        "source_question_payload_sha256": json_digest(list(source_questions)),
        "question_payload_sha256": json_digest(list(questions)),
        "visual_manifest_sha256": json_digest(list(visual_manifest)),
    }
    write_json(output_dir / "metadata.json", metadata)
    (output_dir / "README.md").write_text(README_TEMPLATE.format(image_size=image_size))
    return metadata
