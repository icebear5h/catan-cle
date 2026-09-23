"""QA row loading, token canonicalization, and anchor inference."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

from cle.players.data import JsonValue
from evals.catan_board_bench.tokens import (
    edge_token,
    node_token,
    port_token,
    tile_token,
)
from scripts.probes.probe_catan_board_mech_geometry.constants import (
    CANONICAL_CATEGORY_MAP,
    PARTS,
    TOKEN_PATTERNS,
)

JsonDict = dict[str, JsonValue]

__all__ = [
    "JsonDict",
    "ProbeRow",
    "iter_jsonl",
    "load_manifest",
    "load_rows",
]


class ProbeRow(TypedDict):
    """One QA row resolved to an atlas anchor and an on-disk image."""

    id: JsonValue
    sample_id: JsonValue
    category: JsonValue
    question: JsonValue
    answer: JsonValue
    target_type: str
    target_token: str
    anchor_token: str
    image_path: Path


def iter_jsonl(path: Path) -> Iterator[tuple[int, JsonDict]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, row


def _to_int(value: JsonValue) -> int:
    """Mirror ``int(value)`` while rejecting values ``int()`` would reject."""
    if isinstance(value, (bool, int, float, str)):
        return int(value)
    raise TypeError(f"expected an int-like token component, got {type(value)!r}")


def _canon_token(token: str) -> str:
    token = str(token).strip().upper()
    if not (token.startswith("<") and token.endswith(">")):
        return token
    match = re.fullmatch(r"<T(\d{1,2})>", token)
    if match:
        return f"<T{int(match.group(1)):02d}>"
    match = re.fullmatch(r"<N(\d{1,2})>", token)
    if match:
        return f"<N{int(match.group(1)):02d}>"
    match = re.fullmatch(r"<P(\d{1,2})>", token)
    if match:
        return f"<P{int(match.group(1)):02d}>"
    match = re.fullmatch(r"<E(\d{1,2})_(\d{1,2})>", token)
    if match:
        return edge_token((int(match.group(1)), int(match.group(2))))
    return token


def _extract_tokens_by_kind(text: str, kind: str) -> list[str]:
    text = str(text or "")
    pattern = TOKEN_PATTERNS[kind]
    matches = pattern.findall(text)

    tokens: list[str] = []
    if kind == "edge":
        for match in matches:
            if isinstance(match, tuple) and len(match) == 2:
                tokens.append(edge_token((int(match[0]), int(match[1]))))
            else:
                tokens.append(str(match))
    elif kind == "tile":
        for raw in matches:
            value = raw if not isinstance(raw, tuple) else raw[0]
            tokens.append(tile_token(int(value)))
    elif kind == "node":
        for raw in matches:
            value = raw if not isinstance(raw, tuple) else raw[0]
            tokens.append(node_token(int(value)))
    elif kind == "port":
        for raw in matches:
            value = raw if not isinstance(raw, tuple) else raw[0]
            tokens.append(port_token(int(value)))

    seen: set[str] = set()
    deduped: list[str] = []
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        deduped.append(tok)
    return deduped


def _kind_from_token(token: str) -> tuple[str, str] | None:
    token = _canon_token(token)
    if re.fullmatch(r"<T\d{2}>", token):
        return "tile", token
    if re.fullmatch(r"<N\d{2}>", token):
        return "node", token
    if re.fullmatch(r"<E\d{2}_\d{2}>", token):
        return "edge", token
    if re.fullmatch(r"<P\d{2}>", token):
        return "port", token
    return None


def _infer_target_token(category: str, target: JsonDict) -> tuple[str, str] | None:
    target_kind = CANONICAL_CATEGORY_MAP.get(category)
    if target_kind is None:
        return None

    preference = {
        "tile": ["tile_token", "tile", "tile_id"],
        "node": ["node_token", "node", "node_id"],
        "edge": ["edge_token", "edge_id", "road_edge", "road_edge_tokens", "nodes"],
        "port": ["port_token", "port", "port_id"],
    }[target_kind]

    for key in preference:
        if key not in target:
            continue
        value = target.get(key)
        if value is None:
            continue

        if isinstance(value, list):
            if not value:
                continue
            if target_kind == "edge" and len(value) == 2:
                return "edge", edge_token((_to_int(value[0]), _to_int(value[1])))
            value = value[0]

        if isinstance(value, int):
            if target_kind == "tile":
                return "tile", tile_token(value)
            if target_kind == "node":
                return "node", node_token(value)
            if target_kind == "port":
                return "port", port_token(value)

        if isinstance(value, str):
            parsed = _kind_from_token(value)
            if parsed and parsed[0] == target_kind:
                return parsed

    # Fallback by explicit id fields that may appear with different keys.
    tile_tok = target.get("tile_token")
    if target_kind == "tile" and isinstance(tile_tok, str):
        parsed = _kind_from_token(tile_tok)
        if parsed:
            return parsed
    node_tok = target.get("node_token")
    if target_kind == "node" and isinstance(node_tok, str):
        parsed = _kind_from_token(node_tok)
        if parsed:
            return parsed
    edge_id = target.get("edge_id")
    if target_kind == "edge" and isinstance(edge_id, list) and len(edge_id) == 2:
        return "edge", edge_token((_to_int(edge_id[0]), _to_int(edge_id[1])))
    port_tok = target.get("port_token")
    if target_kind == "port" and isinstance(port_tok, str):
        parsed = _kind_from_token(port_tok)
        if parsed:
            return parsed
    return None


def _infer_anchor_token(row: JsonDict, target_kind: str) -> str | None:
    for text in (row.get("question"), row.get("answer")):
        tokens = _extract_tokens_by_kind(str(text), target_kind)
        if tokens:
            return tokens[0]

    target = row.get("target")
    if "target" in row and isinstance(target, dict):
        parsed = _infer_target_token(str(row.get("category", "")), target)
        if parsed and parsed[0] == target_kind:
            return parsed[1]
    return None


def load_manifest(path: Path | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if path is None or not path.exists():
        return mapping
    for _, row in iter_jsonl(path):
        sample_id = row.get("sample_id")
        image_path = row.get("image_path")
        if sample_id and image_path:
            mapping[str(sample_id)] = str(image_path)
    return mapping


def load_rows(
    qa_jsonl: Path, manifest: dict[str, str], dataset_root: Path, args: argparse.Namespace
) -> list[ProbeRow]:
    allowed_categories = {cat.strip() for cat in args.categories.split(",") if cat.strip()}
    requested_parts = {part.strip().lower() for part in args.parts.split(",") if part.strip()}
    requested_parts = {part for part in requested_parts if part in PARTS}
    if not requested_parts:
        requested_parts = set(PARTS)

    rows: list[ProbeRow] = []
    for _, row in iter_jsonl(qa_jsonl):
        category = row.get("category")
        if category not in allowed_categories:
            continue
        target_kind = CANONICAL_CATEGORY_MAP.get(str(category))
        if target_kind is None:
            continue
        if target_kind not in requested_parts:
            continue

        target = row.get("target")
        if not isinstance(target, dict):
            continue

        anchor_token = _infer_anchor_token(row, target_kind)
        if anchor_token is None:
            continue

        inferred = _infer_target_token(str(row.get("category", "")), target)
        target_token = inferred[1] if inferred else anchor_token

        sample_id = row.get("sample_id")
        raw_image = row.get("image_path")
        if not raw_image and sample_id:
            raw_image = manifest.get(str(sample_id))
        if not raw_image:
            continue

        image_full = Path(str(raw_image))
        if not image_full.is_absolute():
            image_full = (dataset_root / image_full).resolve()

        rows.append(
            ProbeRow(
                id=row.get("id"),
                sample_id=sample_id,
                category=row.get("category"),
                question=row.get("question"),
                answer=row.get("answer"),
                target_type=target_kind,
                target_token=target_token,
                anchor_token=anchor_token,
                image_path=image_full,
            )
        )
        if len(rows) >= args.rows_limit:
            break
    return rows
