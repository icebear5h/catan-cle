"""Strict local JSON boundaries and CPU merge contracts (no trainer imports)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from sft.safetensor_types import TensorHeader

INDEX = "model.safetensors.index.json"
MANIFEST = "merge_manifest.json"
COMPLETE = "MERGE_COMPLETE.json"
ADAPTER = "adapter_model.safetensors"
VISUAL = "visual_model.safetensors"
PREFIX = "base_model.model."
ROW_SUFFIX = ".token_adapter.trainable_tokens_delta"
ROW_MODULES = ("model.language_model.embed_tokens", "lm_head")
VOCAB_SIZE = 248320
TOKEN_COUNT = 154
VISUAL_COUNT = 333
SCHEMA = "catan_miles_merged_checkpoint/v1"
FLOAT_DTYPES = {"BF16", "F16", "F32"}
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8,
               "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def object_map(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    raw = cast("dict[object, object]", value)
    require(all(isinstance(key, str) for key in raw), "expected string object keys")
    return {text(key): item for key, item in raw.items()}


def sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected JSON array")
    return cast("list[object]", value)


def text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("expected string")
    return value


def integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("expected integer (not boolean)")
    return value


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = item
    return result


def _invalid_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON value: {value}")


def read_json(path: Path) -> dict[str, object]:
    decoded: object = json.loads(path.read_text(), object_pairs_hook=_unique,
                                 parse_constant=_invalid_constant)
    return object_map(decoded)


def write_json(path: Path, value: object) -> None:
    with path.open("x") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_name(value: object) -> str:
    name = text(value)
    path = Path(name)
    require(bool(name) and not path.is_absolute() and ".." not in path.parts
            and path.as_posix() == name and "\\" not in name, f"unsafe file name: {name}")
    return name


class FileIdentity(TypedDict):
    bytes: int
    sha256: str


def file_identity(path: Path) -> FileIdentity:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def identities(root: Path, names: set[str]) -> dict[str, FileIdentity]:
    return {name: file_identity(root / name) for name in sorted(names)}


@dataclass(frozen=True)
class Checkpoint:
    weight_map: dict[str, str]
    headers: dict[str, TensorHeader]
    shards: dict[str, tuple[str, ...]]
    total_size: int


@dataclass(frozen=True)
class MergePlan:
    base: Checkpoint
    lora: dict[str, tuple[str, str]]
    rows: dict[str, str]
    visual: dict[str, str]
    token_ids: tuple[int, ...]
    assets: dict[str, Path]
    base_files: dict[str, FileIdentity]
    adapter_files: dict[str, FileIdentity]


class Rounding(TypedDict):
    elements: int
    rounded_elements: int
    max_abs_rounding_error: float


def valid_token_ids(value: object) -> tuple[int, ...]:
    ids = tuple(integer(item) for item in sequence(value))
    require(len(ids) == TOKEN_COUNT and len(set(ids)) == TOKEN_COUNT
            and min(ids) >= 0 and max(ids) < VOCAB_SIZE,
            "expected exactly 154 unique token IDs within the full 248320 vocabulary")
    return ids
