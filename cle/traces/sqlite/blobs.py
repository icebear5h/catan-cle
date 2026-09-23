"""JSON/pickle encoding and the zlib blob envelope used by every trace column."""

from __future__ import annotations

import io
import json
import pickle
import zlib
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

from cle.game_engine.json import GameEncoder
from cle.game_engine.public_board import JsonValue
from cle.players.contracts import PlayerChoice
from cle.sandbox.contracts import SandboxSnapshot

__all__ = [
    "PACKED_MAGIC",
    "is_packed",
    "pack_blob",
    "unpack_blob",
]


def _jsonable(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        encoded_enum: JsonValue = value.value
        return encoded_enum
    if isinstance(value, bytes):
        return {"encoding": "hex", "value": value.hex()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(_jsonable(key)): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
            if not isinstance(value, PlayerChoice) or field.name != "rationale"
        }
    try:
        encoded = GameEncoder().default(value)
    except TypeError:
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": repr(value),
        }
    return _jsonable(encoded)


def _json_text(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
    )


# Large columns are stored zlib-packed behind this prefix. Legacy rows hold raw
# pickle (starts 0x80) or JSON text (starts "{"), so the prefix is unambiguous
# and unpack_blob passes them through unchanged. Columns that SQL inspects with
# json_extract/json_each (model_calls.response_json, live_failures.payload_json)
# stay plain text.
PACKED_MAGIC = b"\x00clz1\x00"


def pack_blob(data: bytes) -> bytes:
    return PACKED_MAGIC + zlib.compress(data, 6)


def is_packed(value: object) -> bool:
    return isinstance(value, (bytes, memoryview)) and bytes(value[: len(PACKED_MAGIC)]) == PACKED_MAGIC


def unpack_blob(value: bytes | str | memoryview | None) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.encode("utf-8")
    raw = bytes(value)
    if raw.startswith(PACKED_MAGIC):
        return zlib.decompress(raw[len(PACKED_MAGIC):])
    return raw


def _json_blob(value: object) -> bytes:
    return pack_blob(_json_text(value).encode("utf-8"))


def _load_json(value: bytes | str | memoryview | None) -> JsonValue:
    raw = unpack_blob(value)
    if raw is None:
        return None
    decoded: JsonValue = json.loads(raw.decode("utf-8"))
    return decoded


def _load_json_object(
    value: bytes | str | memoryview | None,
) -> dict[str, JsonValue] | None:
    """Decode a column that always stores a JSON object, or SQL NULL."""
    decoded = _load_json(value)
    if decoded is None or isinstance(decoded, dict):
        return decoded
    raise TypeError("Stored live trace column does not hold a JSON object")


def _snapshot_blob(snapshot: SandboxSnapshot) -> bytes:
    return pack_blob(pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL))


class _SnapshotUnpickler(pickle.Unpickler):
    """Load trusted local snapshots written before the engine namespace move."""

    def find_class(self, module: str, name: str) -> object:
        if module == "game_engine" or module.startswith("game_engine."):
            module = f"cle.{module}"
        loaded: object = super().find_class(module, name)
        return loaded


def _decode_snapshot(payload: bytes) -> SandboxSnapshot:
    raw = unpack_blob(payload)
    assert raw is not None
    snapshot = _SnapshotUnpickler(io.BytesIO(raw)).load()
    if not isinstance(snapshot, SandboxSnapshot):
        raise TypeError("Stored live trace snapshot has an invalid type")
    return snapshot
