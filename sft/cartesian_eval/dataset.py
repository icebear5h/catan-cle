"""Immutable Cartesian-only projection of the exact historical 200 source cases."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from sft.board.coordinate_comparison import validate_comparison_rows
from sft.json_types import as_dict

from .contracts import (
    DECLARATIONS,
    VERSION,
    cartesian_metadata,
    compact,
    mapping_sha256,
    object_list,
    object_map,
    parse_json,
    render_prompt,
    require,
    score_response,
    sha256,
    text,
    validate_metadata,
)
from .geometry import ATLAS_ATOM, SCHEMA, mapping_artifact
from .reference import ATLAS_METADATA_SHA256, SOURCE_SHA256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/generated/sft/cartesian_eval_v2"
DEFAULT_SOURCE = DEFAULT_OUTPUT.parent / "coordinate_comparison_v1/paired.jsonl"
ROW_FIELDS = frozenset({"schema", "id", "row_id", "messages", "metadata", *DECLARATIONS})


def build_rows(source: Path = DEFAULT_SOURCE) -> list[dict[str, object]]:
    """Read and validate the actual immutable 400-row artifact before selecting atlas arms."""
    raw = source.read_bytes()
    require(sha256(raw) == SOURCE_SHA256, "immutable paired source hash mismatch")
    paired = [parse_json(line) for line in raw.decode("utf-8").splitlines()]
    validate_comparison_rows([as_dict(row) for row in paired])
    rows = []
    for row in paired:
        atlas = object_map(row["metadata"])
        if atlas["representation"] != "atlas":
            continue
        metadata = cartesian_metadata(atlas)
        row_id = f"{VERSION}/{text(metadata['source_id'])}"
        rows.append({
            "schema": SCHEMA, "id": row_id, "row_id": row_id,
            **{key: row[key] for key in DECLARATIONS}, "metadata": metadata,
            "messages": [
                {"role": "user", "content": render_prompt(atlas)},
                {"role": "assistant", "content": metadata["answer"]},
            ],
        })
    validate_rows(rows)
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Require every pinned source case exactly once, in its original atlas order.

    No local parent artifact is required: pins plus the old metadata validator
    certify membership/receipts/oracle, and prompts are reconstructed independently.
    """
    require(len(rows) == len(ATLAS_METADATA_SHA256) == 200, "panel requires exactly 200 rows")
    ids, source_ids, splits, operations, families = [], [], [], [], []
    for position, row in enumerate(rows):
        require(row.keys() == ROW_FIELDS, "invalid Cartesian row fields")
        metadata = object_map(row["metadata"])
        atlas = validate_metadata(metadata)
        require(metadata.keys() == cartesian_metadata(atlas).keys(), "unexpected dataset metadata")
        require(metadata["eval_position"] == position, "reference order/duplicate case mismatch")
        source_id = text(atlas["source_id"])
        row_id = f"{VERSION}/{source_id}"
        require(row["schema"] == SCHEMA and row["id"] == row["row_id"] == row_id,
                "Cartesian row identity mismatch")
        for key in DECLARATIONS:
            require(row[key] == atlas[key], f"row {key} mismatch")
        prompt = render_prompt(atlas)
        expected = text(metadata["answer"])
        require(ATLAS_ATOM.search(prompt + expected) is None, "visible atomic atlas tokens")
        require(compact(row["messages"]) == compact([
            {"role": "user", "content": prompt}, {"role": "assistant", "content": expected},
        ]), "rendered prompt/gold mismatch")
        score = object_map(score_response(expected, expected, metadata))
        require(score["correct"] is True and score["format_valid"] is True, "gold roundtrip failed")
        ids.append(row_id)
        source_ids.append(source_id)
        splits.append(text(atlas["split"]))
        operations.append(text(atlas["operation"]))
        families.append(text(atlas["family"]))
    require(len(set(ids)) == 200, "duplicate row IDs")
    return {
        "schema": SCHEMA, "valid": True, "rows": 200, "ids": ids, "source_ids": source_ids,
        "mapping_entities": 154, "mapping_sha256": mapping_sha256(),
        "source_sha256": SOURCE_SHA256, "by_split": dict(Counter(splits)),
        "by_representation": {"cartesian": 200}, "by_family": dict(Counter(families)),
        "by_operation": dict(Counter(operations)), "gold_roundtrip_rows": 200,
        "row_order": "original atlas-arm order from coordinate_comparison_v1/paired.jsonl",
    }


def render_preview(rows: Sequence[Mapping[str, object]]) -> str:
    selected: dict[str, Mapping[str, object]] = {}
    for row in rows:
        operation = text(row["task_type"])
        if operation not in selected or object_map(selected[operation]["metadata"])["answer"] == "NONE":
            selected[operation] = row
    lines = [
        "# Cartesian-only evaluation panel", "",
        "200 immutable cases; intended for stock Qwen3.8-27B inference with no finetuning.",
        "This artifact records dataset construction, not a model run. No tokenizer tokens are added.",
        "Only messages[0].content is model input; gold and canonical receipts are audit data.",
        "198 test cases and two original validation port-incidence cases retain their splits.",
        "Every prompt/gold below is an actual row, with no question-specific geometry expansion.", "",
    ]
    for operation, row in selected.items():
        metadata = object_map(row["metadata"])
        messages = object_list(row["messages"])
        lines.extend([
            f"## {operation}", "", f"Source: `{metadata['source_id']}` ({metadata['split']}).", "",
            "```text", text(object_map(messages[0])["content"]), "```", "", "Gold:",
            "```text", text(object_map(messages[1])["content"]), "```", "",
        ])
    return "\n".join(lines)


def build_dataset(
    output: Path = DEFAULT_OUTPUT, *, source: Path = DEFAULT_SOURCE,
) -> dict[str, object]:
    """Validate and serialize everything before an exclusive, new-destination-only write."""
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"destination already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"destination parent does not exist: {output.parent}")
    rows = build_rows(source)
    validation = validate_rows(rows)
    payloads = {
        "eval.jsonl": "".join(compact(row) + "\n" for row in rows),
        "mapping.json": json.dumps(mapping_artifact(), indent=2, sort_keys=True) + "\n",
        "preview.md": render_preview(rows),
    }
    manifest: dict[str, object] = {
        "schema": SCHEMA, "kind": "manifest", "rows": 200, "mapping_entities": 154,
        "representation": "cartesian", "mapping_sha256": mapping_sha256(),
        "source": {"path": str(source.resolve()), "sha256": SOURCE_SHA256, "rows": 400},
        "reference_source_ids": validation["source_ids"],
        "reference_atlas_metadata_sha256": list(ATLAS_METADATA_SHA256),
        "reference_hash_scope": "canonical JSON atlas metadata in exact source order",
        "mapping_hash_scope": "canonical JSON mapping.json payload",
        "intent": {"model": "Qwen3.8-27B", "weights": "stock", "finetuning": False,
                   "inference_only": True, "model_run_performed": False},
        "added_tokenizer_tokens": [], "admitted_for_training": False,
        "model_input": "messages[0].content only; metadata and mapping.json are oracle-only",
        "selection": "All 200 atlas arms of the hash-pinned, validated historical paired artifact",
        "limitations": [
            "Two port-origin incidence cases retain validation; the other 198 retain test.",
            "Static-topology exposure and shared dynamic states are inherited from the source.",
        ],
        "validation": validation,
        "files": {name: {"path": name, "sha256": sha256(payload.encode("utf-8"))}
                  for name, payload in payloads.items()},
    }
    payloads["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    encoded = {name: payload.encode("utf-8") for name, payload in payloads.items()}
    output.mkdir()
    for name, raw in encoded.items():
        with (output / name).open("xb") as handle:
            handle.write(raw)
    return {"output_dir": str(output), "rows": 200, "mapping_entities": 154,
            "mapping_sha256": mapping_sha256(), "source_sha256": SOURCE_SHA256}
