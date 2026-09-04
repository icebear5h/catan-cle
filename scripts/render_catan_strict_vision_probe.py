#!/usr/bin/env python
"""Project the locked strict 60-question text probe onto raw engine screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from evals.catan_board_bench.paths import (
    canonical_benchmark_reference,
    resolve_benchmark_reference,
)
from evals.catan_board_bench.render import DEFAULT_RENDER_STYLE, render_contract_image
from evals.catan_board_bench.ascii_variations import (
    STRICT_SCORER_VERSION,
    full_fact_digest,
    full_public_graph_facts,
    strict_scorer_digest,
)
from evals.catan_board_bench.text_format_optimization import DATASET_SCHEMA


JsonDict = dict[str, Any]
DEFAULT_SOURCE_DIR = Path("evals/catan_board_bench/datasets/text_format_optimization_probe")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/generated/catan_board_bench/strict_vision_probe_60_1024_board90"
)
OUTPUT_SCHEMA = "catan_strict_vision_probe/v2"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_VIEW_PADDING_FACTOR = 0.933134
DEFAULT_BOARD_CANVAS_FRACTION = 0.9
ENTITY_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(?:T\d{2}|N\d{2}|E\d{2}|P\d{2})(?![A-Za-z0-9_])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--view-padding-factor",
        type=float,
        default=DEFAULT_VIEW_PADDING_FACTOR,
    )
    parser.add_argument(
        "--target-board-canvas-fraction",
        type=float,
        default=DEFAULT_BOARD_CANVAS_FRACTION,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = render_strict_vision_probe(
        args.source_dir,
        args.output_dir,
        image_size=args.image_size,
        view_padding_factor=args.view_padding_factor,
        target_board_canvas_fraction=args.target_board_canvas_fraction,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


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
    source_manifest_by_sample = {row["sample_id"]: row for row in source_manifest}

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
    visual_manifest = []

    for row in source_manifest:
        sample_id = row["sample_id"]
        source_contract_reference = canonical_benchmark_reference(row["source_contract"])
        source_contract_path = resolve_benchmark_reference(source_contract_reference)
        alias_path = source_dir / "aliases" / f"{sample_id}.json"
        fact_path = source_dir / "facts" / f"{sample_id}.json"
        contract = json.loads(source_contract_path.read_text())
        aliases = json.loads(alias_path.read_text())
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
                "canonical_id_map_sha256": json_digest(canonical_maps[sample_id]),
            }
        )

    questions = [
        canonicalize_question(
            question,
            canonical_maps[question["sample_id"]],
            manifest_row=source_manifest_by_sample[question["sample_id"]],
        )
        for question in source_questions
    ]
    validate_canonical_questions(questions)
    write_jsonl(output_dir / "qa.jsonl", questions)
    write_jsonl(output_dir / "manifest.jsonl", visual_manifest)

    metadata = {
        "schema": OUTPUT_SCHEMA,
        "source_dataset_schema": DATASET_SCHEMA,
        "source_dataset": str(source_dir),
        "source_lock": dict(sorted(source_locks.items())),
        "source_lock_sha256": json_digest(source_locks),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "board_count": len(visual_manifest),
        "question_count": len(questions),
        "categories": dict(sorted(Counter(row["category"] for row in questions).items())),
        "strict_json_answers": True,
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
        "image_size": [image_size, image_size],
        "rendered_images": len(visual_manifest),
        "render_variant": {
            "renderer": "evals.catan_board_bench.render",
            "style": asdict(style),
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
        "source_question_payload_sha256": json_digest(source_questions),
        "question_payload_sha256": json_digest(questions),
        "visual_manifest_sha256": json_digest(visual_manifest),
    }
    write_json(output_dir / "metadata.json", metadata)
    (output_dir / "README.md").write_text(
        "# Strict 60-question raw-vision probe\n\n"
        "This is the raw-image projection of `text_format_optimization_probe`. "
        "The engine public-state contracts are the oracle and are rendered as ordinary "
        f"unannotated {image_size}px board screenshots. Text-only opaque aliases are inverted to "
        "canonical engine IDs in the questions and strict answers. The semantic targets, "
        "question IDs, board states, categories, and scorer are unchanged.\n"
    )
    return metadata


def validate_render_args(
    source_dir: Path,
    output_dir: Path,
    *,
    image_size: int,
    view_padding_factor: float,
    target_board_canvas_fraction: float,
) -> None:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if view_padding_factor < 0:
        raise ValueError("view_padding_factor must be non-negative")
    if not 0 < target_board_canvas_fraction <= 1:
        raise ValueError("target_board_canvas_fraction must be in (0, 1]")
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("source_dir and output_dir must differ")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def validate_source_dataset(
    source_dir: Path,
) -> tuple[JsonDict, list[JsonDict], list[JsonDict]]:
    required = ("metadata.json", "manifest.jsonl", "qa.jsonl", "aliases", "facts")
    for name in required:
        if not (source_dir / name).exists():
            raise FileNotFoundError(source_dir / name)

    metadata = json.loads((source_dir / "metadata.json").read_text())
    expected_metadata = {
        "schema": DATASET_SCHEMA,
        "board_count": 12,
        "question_count": 60,
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
        "images": False,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"source metadata mismatch: {mismatches}")

    manifest = read_jsonl(source_dir / "manifest.jsonl")
    questions = read_jsonl(source_dir / "qa.jsonl")
    manifest_by_sample = {row["sample_id"]: row for row in manifest}
    if len(manifest) != 12 or len(manifest_by_sample) != 12:
        raise ValueError("source manifest must contain 12 unique boards")
    if len(questions) != 60 or len({row["id"] for row in questions}) != 60:
        raise ValueError("source QA must contain 60 unique questions")

    for row in manifest:
        sample_id = row["sample_id"]
        contract_path = resolve_benchmark_reference(row["source_contract"])
        if not contract_path.is_file():
            raise FileNotFoundError(contract_path)
        contract = json.loads(contract_path.read_text())
        regenerated_facts, regenerated_aliases = full_public_graph_facts(
            contract,
            sample_id=sample_id,
        )
        stored_facts = json.loads((source_dir / "facts" / f"{sample_id}.json").read_text())
        stored_aliases = json.loads((source_dir / "aliases" / f"{sample_id}.json").read_text())
        if regenerated_facts != stored_facts:
            raise ValueError(f"source facts do not match contract for {sample_id}")
        if regenerated_aliases != stored_aliases:
            raise ValueError(f"source aliases do not match contract for {sample_id}")
        if full_fact_digest(stored_facts) != row["fact_digest"]:
            raise ValueError(f"source fact digest mismatch for {sample_id}")

    question_counts = Counter(row["category"] for row in questions)
    if len(question_counts) != 10 or set(question_counts.values()) != {6}:
        raise ValueError(f"expected six questions in ten categories, got {question_counts}")
    for question in questions:
        sample = manifest_by_sample.get(question["sample_id"])
        if sample is None:
            raise ValueError(f"question references unknown board: {question['id']}")
        if question["fact_digest"] != sample["fact_digest"]:
            raise ValueError(f"question fact digest mismatch: {question['id']}")
    return metadata, manifest, questions


def canonical_id_map(aliases: JsonDict) -> dict[str, str]:
    mapping = {}
    for canonical, alias in aliases["tiles"].items():
        mapping[alias] = f"T{int(canonical):02d}"
    for canonical, alias in aliases["nodes"].items():
        mapping[alias] = f"N{int(canonical):02d}"
    for canonical, alias in aliases["edges"].items():
        node_a, node_b = (int(value) for value in canonical.split(","))
        mapping[alias] = f"E{min(node_a, node_b):02d}_{max(node_a, node_b):02d}"
    for canonical, alias in aliases["ports"].items():
        mapping[alias] = f"P{int(canonical):02d}"
    expected_count = 19 + 54 + 72 + 9
    if len(mapping) != expected_count or len(set(mapping.values())) != expected_count:
        raise ValueError("canonical ID map is incomplete or non-bijective")
    return mapping


def canonicalize_question(
    question: JsonDict,
    mapping: dict[str, str],
    *,
    manifest_row: JsonDict,
) -> JsonDict:
    projected_answer = normalize_projected_answer(
        question["category"],
        translate_json_ids(question["answer"], mapping),
    )
    projected = {
        **question,
        "question": translate_text_ids(question["question"], mapping),
        "answer": projected_answer,
        "target": translate_json_ids(question["target"], mapping),
        "source_answer": question["answer"],
        "source_answer_text": question["answer_text"],
        "source_fact_digest": question["fact_digest"],
        "engine_state_sha256": json_digest(
            json.loads(resolve_benchmark_reference(manifest_row["source_contract"]).read_text())
        ),
        "identity_projection": "canonical_engine_ids",
    }
    projected["answer_text"] = json.dumps(
        projected["answer"],
        separators=(",", ":"),
        sort_keys=True,
    )
    if projected["category"] == "road_inventory":
        projected["output_schema"] = projected["output_schema"].replace("Exx", "Exx_yy")
    projected.pop("fact_digest", None)
    return projected


def normalize_projected_answer(category: str, answer: Any) -> Any:
    if not isinstance(answer, dict):
        return answer
    normalized = dict(answer)
    if category == "node_adjacent_tiles":
        normalized["tiles"] = sorted(normalized["tiles"])
    elif category == "road_inventory":
        normalized["edges"] = sorted(normalized["edges"])
    elif category == "port_occupancy":
        normalized["occupants"] = sorted(
            normalized["occupants"], key=lambda occupant: occupant["node"]
        )
    return normalized


def translate_text_ids(text: str, mapping: dict[str, str]) -> str:
    return ENTITY_ID_PATTERN.sub(lambda match: mapping.get(match.group(0), match.group(0)), text)


def translate_json_ids(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return translate_text_ids(value, mapping)
    if isinstance(value, list):
        return [translate_json_ids(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: translate_json_ids(item, mapping) for key, item in value.items()}
    return value


def validate_canonical_questions(questions: Sequence[JsonDict]) -> None:
    if len(questions) != 60 or len({row["id"] for row in questions}) != 60:
        raise ValueError("canonical QA must contain 60 unique questions")
    categories = Counter(row["category"] for row in questions)
    if len(categories) != 10 or set(categories.values()) != {6}:
        raise ValueError("canonical QA category balance changed")
    for row in questions:
        if row.get("identity_projection") != "canonical_engine_ids":
            raise ValueError(f"question projection marker is missing: {row['id']}")
        if row["answer_text"] != json.dumps(row["answer"], separators=(",", ":"), sort_keys=True):
            raise ValueError(f"canonical answer text mismatch: {row['id']}")


def read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[JsonDict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: JsonDict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
