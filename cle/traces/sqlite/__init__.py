"""SQLite/WAL trace storage for live sandbox steps and failed attempts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from cle.traces.sqlite import failures, reads, resume, rows, schema, writes
from cle.traces.sqlite.blobs import (
    PACKED_MAGIC as PACKED_MAGIC,
)
from cle.traces.sqlite.blobs import (
    _decode_snapshot as _decode_snapshot,
)
from cle.traces.sqlite.blobs import (
    _json_blob as _json_blob,
)
from cle.traces.sqlite.blobs import (
    _json_text as _json_text,
)
from cle.traces.sqlite.blobs import (
    _jsonable as _jsonable,
)
from cle.traces.sqlite.blobs import (
    _load_json as _load_json,
)
from cle.traces.sqlite.blobs import (
    _load_json_object as _load_json_object,
)
from cle.traces.sqlite.blobs import (
    _snapshot_blob as _snapshot_blob,
)
from cle.traces.sqlite.blobs import (
    _SnapshotUnpickler as _SnapshotUnpickler,
)
from cle.traces.sqlite.blobs import (
    is_packed as is_packed,
)
from cle.traces.sqlite.blobs import (
    pack_blob as pack_blob,
)
from cle.traces.sqlite.blobs import (
    unpack_blob as unpack_blob,
)
from cle.traces.sqlite.calls import (
    _attempt_payload as _attempt_payload,
)
from cle.traces.sqlite.calls import (
    _communication_payload as _communication_payload,
)
from cle.traces.sqlite.calls import (
    _provider_payload as _provider_payload,
)
from cle.traces.sqlite.calls import (
    _request_payload as _request_payload,
)
from cle.traces.sqlite.calls import (
    _response_payload as _response_payload,
)
from cle.traces.sqlite.constants import (
    DEFAULT_TRACE_PATH as DEFAULT_TRACE_PATH,
)
from cle.traces.sqlite.constants import (
    SCHEMA_VERSION as SCHEMA_VERSION,
)
from cle.traces.sqlite.constants import (
    LiveTraceResumePoint as LiveTraceResumePoint,
)
from cle.traces.sqlite.constants import (
    _normalize_display_name as _normalize_display_name,
)
from cle.traces.sqlite.constants import (
    _utc_now as _utc_now,
)
from cle.traces.sqlite.payloads import (
    _context_payload as _context_payload,
)
from cle.traces.sqlite.payloads import (
    _event_payload as _event_payload,
)
from cle.traces.sqlite.payloads import (
    _observation_payload as _observation_payload,
)
from cle.traces.sqlite.payloads import (
    _result_payload as _result_payload,
)

# Every helper stays importable from this path. Extracted store methods reach
# `_utc_now` back through this module at call time, so tests that patch
# `cle.traces.sqlite._utc_now` still control every recorded timestamp.

__all__ = [
    "DEFAULT_TRACE_PATH",
    "PACKED_MAGIC",
    "SCHEMA_VERSION",
    "LiveTraceResumePoint",
    "SQLiteLiveTraceStore",
    "is_packed",
    "pack_blob",
    "unpack_blob",
]


class SQLiteLiveTraceStore:
    """Append/query local live traces with one transaction per step or failure."""

    def __init__(self, path: str | Path = DEFAULT_TRACE_PATH) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    _initialize = schema._initialize

    start_game = writes.start_game
    record_step = writes.record_step
    update_step_public_state = writes.update_step_public_state

    record_failure = failures.record_failure
    mark_game_status = failures.mark_game_status
    rename_game = failures.rename_game

    list_games = reads.list_games
    get_usage = reads.get_usage
    get_game = reads.get_game
    get_step = reads.get_step

    load_resume_point = resume.load_resume_point
    load_snapshot = resume.load_snapshot

    _model_call_row = staticmethod(rows._model_call_row)
    _game_row = staticmethod(rows._game_row)


SQLiteLiveTraceStore.__module__ = "cle.traces.sqlite"
