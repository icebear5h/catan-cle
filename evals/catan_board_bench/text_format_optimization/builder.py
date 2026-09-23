"""Builder that materializes the probe dataset and its manifest."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from evals.catan_board_bench.ascii_variations import (
    FACT_SCHEMA,
    STRICT_SCORER_VERSION,
    full_fact_digest,
    strict_scorer_digest,
    write_json,
    write_jsonl,
)
from evals.catan_board_bench.ascii_variations.facts import full_facts_from_json
from evals.catan_board_bench.text_format_optimization.codec import (
    parse_text_format,
    render_text_format,
)
from evals.catan_board_bench.text_format_optimization.schema import (
    _README,
    DATASET_SCHEMA,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    FORMAT_EXTENSIONS,
    FORMAT_NAMES,
    JsonDict,
)
from evals.catan_board_bench.text_format_optimization.support import (
    _json_digest,
    _read_jsonl,
    _require_empty_output,
    _sha256,
    _source_lock,
    _validate_distinct_paths,
    _validate_source_metadata,
)
from evals.json_types import as_str


def build_text_format_optimization_probe(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
) -> JsonDict:
    """Build four paired representations from an ASCII-variation source suite."""

    _validate_distinct_paths(source_dir, output_dir)
    _require_empty_output(output_dir)
    source_metadata = json.loads((source_dir / "metadata.json").read_text())
    _validate_source_metadata(source_metadata)
    questions = _read_jsonl(source_dir / "qa.jsonl")
    source_manifest = _read_jsonl(source_dir / "manifest.jsonl")
    if len(questions) != 60 or len(source_manifest) != 12:
        raise ValueError("optimization source must contain 12 boards and 60 questions")

    output_dir.mkdir(parents=True)
    facts_dir = output_dir / "facts"
    aliases_dir = output_dir / "aliases"
    representations_dir = output_dir / "representations"
    facts_dir.mkdir()
    aliases_dir.mkdir()
    representations_dir.mkdir()

    output_manifest: list[JsonDict] = []
    representation_hashes: JsonDict = {}
    for source_row in source_manifest:
        sample_id = as_str(source_row["sample_id"], "manifest sample_id")
        fact_path = source_dir / "facts" / f"{sample_id}.json"
        alias_path = source_dir / "aliases" / f"{sample_id}.json"
        facts = full_facts_from_json(json.loads(fact_path.read_text()))
        digest = full_fact_digest(facts)
        if digest != source_row["fact_digest"]:
            raise ValueError(f"source fact digest mismatch for {sample_id}")
        shutil.copyfile(fact_path, facts_dir / fact_path.name)
        shutil.copyfile(alias_path, aliases_dir / alias_path.name)

        sample_dir = representations_dir / sample_id
        sample_dir.mkdir()
        metrics: JsonDict = {}
        sample_hashes: JsonDict = {}
        representation_hashes[sample_id] = sample_hashes
        for format_name in FORMAT_NAMES:
            extension = FORMAT_EXTENSIONS[format_name]
            output_path = sample_dir / f"{format_name}{extension}"
            text = render_text_format(format_name, facts, sample_id=sample_id)
            if format_name == "tile_rows":
                source_bytes = (
                    source_dir / "representations" / sample_id / "tile_rows.txt"
                ).read_bytes()
                if source_bytes.decode().rstrip("\n") != text:
                    raise ValueError(f"tile_rows baseline changed for {sample_id}")
                output_path.write_bytes(source_bytes)
            else:
                output_path.write_text(text + "\n")
            parsed = parse_text_format(format_name, text)
            if full_fact_digest(parsed) != digest:
                raise ValueError(f"round-trip mismatch for {sample_id}/{format_name}")
            payload = output_path.read_bytes()
            sha256 = _sha256(payload)
            sample_hashes[format_name] = sha256
            metrics[format_name] = {
                "characters": len(text),
                "lines": len(text.splitlines()),
                "sha256": sha256,
            }
        output_manifest.append({**source_row, "representation_metrics": metrics})

    shutil.copyfile(source_dir / "qa.jsonl", output_dir / "qa.jsonl")
    write_jsonl(output_dir / "manifest.jsonl", output_manifest)
    (output_dir / "README.md").write_text(_README)
    source_lock = _source_lock(source_dir, source_manifest)
    metadata: JsonDict = {
        "schema": DATASET_SCHEMA,
        "fact_schema": FACT_SCHEMA,
        "source_dataset": str(source_dir),
        "source_dataset_schema": source_metadata["schema"],
        "source_lock_sha256": _json_digest(source_lock),
        "source_lock": source_lock,
        "board_count": 12,
        "source_game_count": len({row["source_game_id"] for row in output_manifest}),
        "question_count": 60,
        "questions_per_format": 60,
        "request_count": 60 * len(FORMAT_NAMES),
        "formats": list(FORMAT_NAMES),
        "format_extensions": dict(FORMAT_EXTENSIONS),
        "categories": dict(
            Counter(as_str(row["category"], "question category") for row in questions)
        ),
        "representation_hashes": representation_hashes,
        "entity_ids": "deterministically permuted and board-local",
        "strict_json_answers": True,
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
        "images": False,
        "precomputed_player_counts_in_facts": False,
        "precomputed_roll_payouts_in_facts": False,
        "incident_list_format": False,
        "query_indexes": [
            "tile_neighbors",
            "tile_corners",
            "port_nodes",
            "players_without_counts",
            "roll_tiles_without_payouts",
            "roll_sources_without_aggregation",
        ],
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata

