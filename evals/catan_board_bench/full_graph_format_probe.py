"""Dataset builder for the strict full-graph Catan format search."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from evals.catan_board_bench.ascii_variations import (
    FACT_SCHEMA,
    STRICT_SCORER_VERSION,
    full_fact_digest,
    strict_scorer_digest,
    write_json,
    write_jsonl,
)
from evals.catan_board_bench.full_graph_formats import (
    FORMAT_EXTENSIONS,
    FORMAT_NAMES,
    MINIMAL_SCHEMA,
    parse_full_graph_format,
    render_full_graph_format,
)


from evals.catan_board_bench.paths import RELATIVE_DATASETS_DIR


JsonDict = Dict[str, Any]
DATASET_SCHEMA = "catan_full_graph_format_probe/v1"
DEFAULT_SOURCE_DIR = RELATIVE_DATASETS_DIR / "ascii_variation_probe"
DEFAULT_OUTPUT_DIR = RELATIVE_DATASETS_DIR / "full_graph_format_probe"
_README = """# Catan Full-Graph Format Probe

A strict, text-only comparison of six lossless renderings of the same public
Catan board facts and the same 60 diagnostic questions.

Formats:

- `optimized_html`
- `full_graph_json`
- `datalog`
- `sql_relational`
- `integrated_ascii`
- `tile_rows` (byte-identical incumbent baseline)

Every rendering round-trips to the source `catan_full_public_graph/v1` digest.
The compact non-baseline formats may omit redundant incidence lists only when
they can be deterministically reconstructed from tile, edge, and port topology.
No representation contains player summaries or precomputed question answers.

`integrated_ascii` places tile state, buildings, and roads directly in the
global board diagram. Its static appendix contains topology and ports but does
not repeat dynamic tile, node, or edge state.
"""


def build_full_graph_format_probe(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
) -> JsonDict:
    """Build six format projections without modifying the frozen source probe."""

    _validate_distinct_paths(source_dir, output_dir)
    _require_empty_output(output_dir)
    source_metadata = json.loads((source_dir / "metadata.json").read_text())
    _validate_source_metadata(source_metadata)

    questions = _read_jsonl(source_dir / "qa.jsonl")
    source_manifest = _read_jsonl(source_dir / "manifest.jsonl")
    if len(questions) != 60:
        raise ValueError(f"expected 60 source questions, found {len(questions)}")
    if len(source_manifest) != 12:
        raise ValueError(f"expected 12 source manifest rows, found {len(source_manifest)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    facts_dir = output_dir / "facts"
    aliases_dir = output_dir / "aliases"
    representations_dir = output_dir / "representations"
    facts_dir.mkdir()
    aliases_dir.mkdir()
    representations_dir.mkdir()

    manifest_by_sample = {row["sample_id"]: row for row in source_manifest}
    if len(manifest_by_sample) != len(source_manifest):
        raise ValueError("source manifest contains duplicate sample IDs")

    output_manifest = []
    representation_hashes: dict[str, dict[str, str]] = {}
    for sample_id in sorted(manifest_by_sample):
        source_fact_path = source_dir / "facts" / f"{sample_id}.json"
        facts = json.loads(source_fact_path.read_text())
        digest = full_fact_digest(facts)
        if digest != manifest_by_sample[sample_id]["fact_digest"]:
            raise ValueError(f"source fact digest mismatch for {sample_id}")

        shutil.copyfile(source_fact_path, facts_dir / source_fact_path.name)
        source_alias_path = source_dir / "aliases" / f"{sample_id}.json"
        shutil.copyfile(source_alias_path, aliases_dir / source_alias_path.name)
        sample_output_dir = representations_dir / sample_id
        sample_output_dir.mkdir()

        metrics = {}
        representation_hashes[sample_id] = {}
        for format_name in FORMAT_NAMES:
            extension = FORMAT_EXTENSIONS[format_name]
            output_path = sample_output_dir / f"{format_name}{extension}"
            if format_name == "tile_rows":
                source_path = source_dir / "representations" / sample_id / "tile_rows.txt"
                source_bytes = source_path.read_bytes()
                text = source_bytes.decode().rstrip("\n")
                generated = render_full_graph_format(
                    format_name,
                    facts,
                    sample_id=sample_id,
                )
                if generated != text:
                    raise ValueError(f"tile_rows renderer changed frozen bytes for {sample_id}")
                output_path.write_bytes(source_bytes)
            else:
                text = render_full_graph_format(
                    format_name,
                    facts,
                    sample_id=sample_id,
                )
                output_path.write_text(text + "\n")

            parsed = parse_full_graph_format(format_name, text)
            if full_fact_digest(parsed) != digest:
                raise ValueError(f"round-trip mismatch for {sample_id}/{format_name}")
            representation_sha256 = _sha256(output_path.read_bytes())
            representation_hashes[sample_id][format_name] = representation_sha256
            metrics[format_name] = {
                "characters": len(text),
                "lines": len(text.splitlines()),
                "sha256": representation_sha256,
            }

        source_row = manifest_by_sample[sample_id]
        output_manifest.append(
            {
                **source_row,
                "representation_metrics": metrics,
            }
        )

    shutil.copyfile(source_dir / "qa.jsonl", output_dir / "qa.jsonl")
    copied_questions = _read_jsonl(output_dir / "qa.jsonl")
    if copied_questions != questions:
        raise ValueError("question copy changed source payload")
    write_jsonl(output_dir / "manifest.jsonl", output_manifest)
    (output_dir / "README.md").write_text(_README)

    source_lock = _source_lock(source_dir, source_manifest)
    metadata = {
        "schema": DATASET_SCHEMA,
        "fact_schema": FACT_SCHEMA,
        "minimal_representation_schema": MINIMAL_SCHEMA,
        "source_dataset": str(source_dir),
        "source_dataset_schema": source_metadata["schema"],
        "source_lock_sha256": _json_digest(source_lock),
        "source_lock": source_lock,
        "board_count": len(output_manifest),
        "source_game_count": len({row["source_game_id"] for row in output_manifest}),
        "question_count": len(questions),
        "questions_per_format": len(questions),
        "request_count": len(questions) * len(FORMAT_NAMES),
        "formats": list(FORMAT_NAMES),
        "format_extensions": FORMAT_EXTENSIONS,
        "categories": dict(Counter(row["category"] for row in questions)),
        "representation_hashes": representation_hashes,
        "entity_ids": "deterministically permuted and board-local",
        "strict_json_answers": True,
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
        "images": False,
        "precomputed_player_counts_in_facts": False,
        "incident_list_format": False,
        "integrated_ascii_dynamic_state_only_in_diagram": True,
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def _validate_source_metadata(metadata: JsonDict) -> None:
    expected = {
        "schema": "catan_ascii_variation_probe/v1",
        "fact_schema": FACT_SCHEMA,
        "board_count": 12,
        "question_count": 60,
        "strict_scorer_version": STRICT_SCORER_VERSION,
        "strict_scorer_sha256": strict_scorer_digest(),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"source probe metadata mismatch: {mismatches}")


def _source_lock(source_dir: Path, manifest: list[JsonDict]) -> JsonDict:
    paths = [
        source_dir / "metadata.json",
        source_dir / "qa.jsonl",
        source_dir / "manifest.jsonl",
    ]
    for row in manifest:
        sample_id = row["sample_id"]
        paths.extend(
            (
                source_dir / "facts" / f"{sample_id}.json",
                source_dir / "aliases" / f"{sample_id}.json",
                source_dir / "representations" / sample_id / "tile_rows.txt",
            )
        )
    return {str(path.relative_to(source_dir)): _sha256(path.read_bytes()) for path in sorted(paths)}


def _validate_distinct_paths(source_dir: Path, output_dir: Path) -> None:
    source = source_dir.resolve()
    output = output_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output datasets must be disjoint")


def _require_empty_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_digest(value: Any) -> str:
    return _sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode())
