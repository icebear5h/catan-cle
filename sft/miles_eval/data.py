"""Lossless text-panel projection and validation for the Miles JSONL loader."""

from __future__ import annotations

import re
from pathlib import Path

from sft.board.board_fluency_scoring import SCHEMA as REVIEW_SCHEMA
from sft.board.coordinate_comparison import SCHEMA as COORDINATE_SCHEMA
from sft.board.coordinate_comparison import validate_comparison_rows
from sft.cartesian_eval import SCHEMA as CARTESIAN_SCHEMA
from sft.cartesian_eval import validate_rows as validate_cartesian_rows
from sft.cartesian_eval.scaling import SCHEMA as SCALING_SCHEMA
from sft.cartesian_eval.scaling import validate_rows as validate_scaling_rows
from sft.cartesian_eval.shorthand import SCHEMA as SHORTHAND_SCHEMA
from sft.cartesian_eval.shorthand import validate_rows as validate_shorthand_rows
from sft.miles_eval.contracts import (
    Json,
    JsonObject,
    as_json,
    compact,
    dimensions,
    json_list,
    json_object,
    parse_json,
    require,
    score,
    sha256,
    text,
)

DECLARATIONS = (
    "schema", "task_type", "training_family", "split", "task_role", "review_only",
    "admitted_for_training", "class", "family", "operation", "target", "answer", "provenance",
)
MEDIA_FIELDS = frozenset({"image", "images", "video", "videos", "audio", "audios", "multimodal_inputs"})
MEDIA_MARKERS = ("<image>", "<video>", "<audio>", "<|image_pad|>", "<|video_pad|>")


def _read_rows(raw: bytes) -> list[JsonObject]:
    lines = raw.decode("utf-8").splitlines()
    require(bool(lines) and all(line.strip() for line in lines), "empty panel or blank JSONL row")
    return [parse_json(line) for line in lines]


def _metadata(row: JsonObject) -> JsonObject:
    metadata = json_object(row.get("metadata"))
    require(not MEDIA_FIELDS.intersection(row) and not MEDIA_FIELDS.intersection(metadata),
            "media rows are unsupported")
    for field in DECLARATIONS:
        if field in row:
            require(field not in metadata or compact(row[field]) == compact(metadata[field]),
                    f"conflicting declaration: {field}")
            metadata[field] = row[field]
    # Review rows deliberately retain donor training provenance; they are not held-out test rows.
    if metadata.get("schema") != REVIEW_SCHEMA:
        require(metadata.get("split") in ("test", "validation", "transfer_test", "transfer_validation"),
                "evaluation requires a non-training split")
    require(metadata.get("split") != "train" and metadata.get("task_role") != "train"
            and metadata.get("admitted_for_training") is not True, "training rows are not eval")
    return metadata


def _message(value: Json, role: str) -> str:
    message = json_object(value)
    require(message.keys() == {"role", "content"} and message["role"] == role,
            f"expected exact text {role} message")
    content = text(message["content"])
    require(bool(content.strip()) and not any(marker in content for marker in MEDIA_MARKERS),
            "empty or media message")
    return content


def _project(rows: list[JsonObject], source_hash: str) -> list[JsonObject]:
    metadata = [_metadata(row) for row in rows]
    if any(item.get("schema") == COORDINATE_SCHEMA for item in metadata):
        validate_comparison_rows(rows)
    if any(item.get("schema") == CARTESIAN_SCHEMA for item in metadata):
        validate_cartesian_rows(rows)
    if any(item.get("schema") == SHORTHAND_SCHEMA for item in metadata):
        validate_shorthand_rows(rows)
    if any(item.get("schema") == SCALING_SCHEMA for item in metadata):
        validate_scaling_rows(rows)
    prepared: list[JsonObject] = []
    seen: set[str] = set()
    for row, scoring_metadata in zip(rows, metadata, strict=True):
        row_id = text(row.get("id", row.get("row_id")))
        require(bool(row_id) and row_id not in seen and row.get("row_id", row_id) == row_id,
                "duplicate or mismatched row ID")
        seen.add(row_id)
        messages = json_list(row.get("messages"))
        require(len(messages) == 2, "expected exactly user/assistant messages")
        prompt, expected = _message(messages[0], "user"), _message(messages[1], "assistant")
        require(score(expected, expected, scoring_metadata)["correct"] is True,
                f"gold self-score failed: {row_id}")
        dimensions(scoring_metadata)
        prepared.append({
            "prompt": [{"role": "user", "content": prompt}], "label": expected,
            "metadata": {"source_sha256": source_hash, "catan": {
                "id": row_id, "metadata": scoring_metadata, "expected": expected,
                "prompt_sha256": sha256(prompt.encode("utf-8")), "source_row": row,
            }},
        })
    return prepared


def panel_counts(rows: list[JsonObject]) -> JsonObject:
    counts: dict[str, dict[str, int]] = {key: {} for key in ("operation", "family", "representation")}
    for row in rows:
        catan = json_object(json_object(row["metadata"])["catan"])
        for key, value in dimensions(json_object(catan["metadata"])).items():
            counts[key][value] = counts[key].get(value, 0) + 1
    return {f"by_{key}": as_json(value) for key, value in counts.items()}


def prepare_panel(source: Path, destination: Path) -> dict[str, Json]:
    """Validate all golds before exclusively writing; the source is never changed."""
    raw = source.read_bytes()
    source_hash = sha256(raw)
    rows = _project(_read_rows(raw), source_hash)
    payload = "".join(compact(row) + "\n" for row in rows).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(payload)
    ids: list[Json] = [json_object(json_object(row["metadata"])["catan"])["id"] for row in rows]
    return {"rows": len(rows), "ids": ids, "source": str(source.resolve()),
            "destination": str(destination.resolve()), "source_sha256": source_hash,
            "destination_sha256": sha256(payload), "gold_self_scored": len(rows), **panel_counts(rows)}


def read_panel(path: Path) -> list[JsonObject]:
    """Read only our export, verifying each projection and the full coordinate panel."""
    rows = _read_rows(path.read_bytes())
    originals: list[JsonObject] = []
    hashes: set[str] = set()
    for row in rows:
        require(row.keys() == {"prompt", "label", "metadata"}, "invalid prepared row fields")
        metadata = json_object(row["metadata"])
        require(metadata.keys() == {"catan", "source_sha256"}, "invalid export metadata")
        source_hash = text(metadata["source_sha256"])
        require(re.fullmatch("[0-9a-f]{64}", source_hash) is not None, "invalid source hash")
        hashes.add(source_hash)
        catan = json_object(metadata["catan"])
        originals.append(json_object(catan.get("source_row")))
    require(len(hashes) == 1, "mixed source hashes in panel")
    projected = _project(originals, next(iter(hashes)))
    require(compact(as_json(rows)) == compact(as_json(projected)), "prepared projection identity mismatch")
    return rows
