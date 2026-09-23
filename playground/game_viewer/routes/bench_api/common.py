"""Small readers and coercions the bench routes share."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import cast

from flask import request


def _int_arg(name: str, default: int) -> int:
    try:
        return int(request.args.get(name, default))
    except ValueError:
        return default


def _parse_requested_eval_models(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return tuple()
    requested = tuple(_normalize_model_key(item) for item in raw.split(",") if item.strip())
    # De-duplicate while preserving user order
    out: list[str] = []
    seen: set[str] = set()
    for item in requested:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _coerce_category_payload(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {}

    out: dict[str, object] = {}
    for category, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        out[category] = {
            "attempted": _coerce_int(payload.get("attempted")),
            "requests": _coerce_int(payload.get("requests")),
            "errors": _coerce_int(payload.get("errors")),
            "exact_accuracy": _coerce_float(payload.get("exact_accuracy")),
            "component_accuracy": _coerce_float(payload.get("component_accuracy")),
            "avg_latency_ms": _coerce_float(payload.get("avg_latency_ms")),
        }
    return out


def _coerce_int(value: object) -> int:
    # The try/except is the real contract; the cast only names what int() accepts.
    try:
        return int(cast(int, value) or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: object) -> float:
    # The try/except is the real contract; the cast only names what float() accepts.
    try:
        return float(cast(float, value) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_model_key(raw: str) -> str:
    value = raw.strip().lower().replace("_", "-")
    if value.startswith("qwen-qwen"):
        value = "qwen" + value[len("qwen-qwen") :]
    if value in {"qwen3", "qwen3-vl", "qwen-3", "qwen3vl"}:
        value = "qwen3-vl-8b"
    if value in {"qwen3.5", "qwen-3.5", "qwen3.5-vl", "qwen-3.5-vl"}:
        value = "qwen3.5-9b"
    if value in {"qwen3.6", "qwen-qwen3.6-flash"}:
        value = "qwen3.6-flash"
    return value


def _parse_generated_at(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None
    return None


def _read_json(path: Path) -> dict[str, object]:
    payload: dict[str, object] = json.loads(path.read_text())
    return payload


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _count_by(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "unspecified")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _unique_values(rows: Sequence[Mapping[str, object]], field: str) -> list[str]:
    return sorted({str(row.get(field) or "unspecified") for row in rows})


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seq(value: object) -> Sequence[object]:
    """Read one JSON list the artifact schema guarantees."""
    return cast(Sequence[object], value)


def _maps(value: object) -> Sequence[Mapping[str, object]]:
    """Read one JSON list of objects the artifact schema guarantees."""
    return cast(Sequence[Mapping[str, object]], value)


def _map(value: object) -> Mapping[str, object]:
    """Read one nested JSON object the artifact schema guarantees."""
    return cast(Mapping[str, object], value)
