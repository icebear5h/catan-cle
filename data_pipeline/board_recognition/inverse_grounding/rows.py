"""Inverse row validation and the fixed two-forward/one-inverse mix."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from data_pipeline.board_recognition import inverse_grounding as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_dict, as_str


def validate_inverse_ms_swift_row(row: JsonDict) -> None:
    if set(row) != api.ROW_KEYS:
        raise api.InverseGroundingError(f"inverse row keys must be {sorted(api.ROW_KEYS)}")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise api.InverseGroundingError("inverse row must have one user and one assistant turn")
    if as_dict(messages[0]).get("role") != "user" or as_dict(messages[1]).get("role") != "assistant":
        raise api.InverseGroundingError("inverse message roles are invalid")
    prompt = as_dict(messages[0]).get("content")
    answer = as_dict(messages[1]).get("content")
    if not isinstance(prompt, str) or not prompt.startswith("<image>\nWhere is ") or not prompt.endswith("?"):
        raise api.InverseGroundingError("inverse prompt has invalid format")
    description = prompt.removeprefix("<image>\nWhere is ").removesuffix("?")
    if not description or "<" in description or ">" in description:
        raise api.InverseGroundingError("inverse prompt leaks an atlas token")
    if not isinstance(answer, str) or answer not in api.ATLAS_TOKENS or not api.ATLAS_TOKEN_RE.fullmatch(answer):
        raise api.InverseGroundingError("inverse answer must be exactly one canonical atlas token")
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise api.InverseGroundingError("inverse row must reference exactly one image")
    image_name = images[0]
    if not isinstance(image_name, str) or Path(image_name).name != image_name:
        raise api.InverseGroundingError("inverse image must be a filename relative to image_root")


def _validate_schema_rows(schema_path: Path, rows: Sequence[JsonDict], *, label: str) -> int:
    validator = api.Draft202012Validator(json.loads(schema_path.read_text()))
    for index, row in enumerate(rows):
        errors = sorted(validator.iter_errors(row), key=lambda error: list(error.path))
        if errors:
            error = errors[0]
            path = ".".join(str(part) for part in error.path) or "<root>"
            raise api.InverseGroundingError(
                f"{label}[{index}] failed {schema_path.name} at {path}: {error.message}"
            )
    return len(rows)


def _forward_rows_by_state(
    annotations: Sequence[JsonDict], audits: Sequence[JsonDict]
) -> dict[str, list[tuple[JsonDict, JsonDict]]]:
    if len(annotations) != len(audits):
        raise api.InverseGroundingError("forward annotations and audits are not aligned")
    grouped: dict[str, list[tuple[JsonDict, JsonDict]]] = defaultdict(list)
    for annotation, audit in zip(annotations, audits, strict=True):
        grouped[as_str(audit["state_id"])].append((annotation, audit))
    if any(len(rows) != api.FORWARD_ROWS_PER_STATE for rows in grouped.values()):
        raise api.InverseGroundingError("forward projection is not eight rows per state")
    return dict(grouped)


def _mixed_rows_for_state(
    forward: Sequence[tuple[JsonDict, JsonDict]],
    inverse_rows: Sequence[JsonDict],
    inverse_audits: Sequence[JsonDict],
) -> tuple[list[JsonDict], list[JsonDict]]:
    if len(forward) != api.FORWARD_ROWS_PER_STATE or len(inverse_rows) != api.INVERSE_ROWS_PER_STATE:
        raise api.InverseGroundingError("cannot construct the fixed 2:1 mixed block")
    rows = []
    index = []
    for group in range(api.INVERSE_ROWS_PER_STATE):
        for annotation, audit in forward[group * 2 : group * 2 + 2]:
            rows.append(annotation)
            index.append(
                {"row_kind": "forward", "state_id": audit["state_id"], "query_id": audit["query_id"]}
            )
        rows.append(inverse_rows[group])
        index.append(
            {
                "row_kind": "inverse",
                "state_id": inverse_audits[group]["state_id"],
                "query_id": inverse_audits[group]["query_id"],
            }
        )
    return rows, index
