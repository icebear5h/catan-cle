"""Deterministic helpers shared by the probe builder and the index renderers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evals.catan_board_bench.ascii_variations import (
    FACT_SCHEMA,
    STRICT_SCORER_VERSION,
    strict_scorer_digest,
)
from evals.catan_board_bench.text_format_optimization.schema import JsonDict
from evals.json_types import JsonValue, as_str


def _reject_duplicate_pairs(pairs: list[tuple[str, JsonValue]]) -> JsonDict:
    result: JsonDict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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
        raise ValueError(f"source metadata mismatch: {mismatches}")


def _source_lock(source_dir: Path, manifest: list[JsonDict]) -> JsonDict:
    paths = [source_dir / "metadata.json", source_dir / "qa.jsonl", source_dir / "manifest.jsonl"]
    for row in manifest:
        sample_id = as_str(row["sample_id"], "manifest sample_id")
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


def _json_digest(value: object) -> str:
    return _sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode())
