"""Defaults, schema names, and digests for the strict vision projection."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from cle.players.data import JsonValue

DEFAULT_SOURCE_DIR = Path("evals/catan_board_bench/datasets/text_format_optimization_probe")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/generated/catan_board_bench/strict_vision_probe_60_1024_board90"
)
OUTPUT_SCHEMA = "catan_strict_vision_probe/v2"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_VIEW_PADDING_FACTOR = 0.933134
DEFAULT_BOARD_CANVAS_FRACTION = 0.9
ENTITY_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(?:T\d{2}|N\d{2}|E\d{2}|P\d{2})(?![A-Za-z0-9_])")

__all__ = [
    "DEFAULT_BOARD_CANVAS_FRACTION",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SOURCE_DIR",
    "DEFAULT_VIEW_PADDING_FACTOR",
    "ENTITY_ID_PATTERN",
    "OUTPUT_SCHEMA",
    "file_sha256",
    "json_digest",
]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_digest(value: JsonValue) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
