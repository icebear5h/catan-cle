"""Pack legacy live-trace rows with the store's zlib encoding and reclaim the file.

The trace store writes large columns zlib-packed behind ``PACKED_MAGIC`` and
reads legacy raw rows transparently, so an existing database keeps working
without this script. It exists to shrink history that was written before
packing: every legacy row in the packed columns is rewritten in place, in
batches, and ``--vacuum`` rebuilds the file so the freed pages leave the disk.

Dry run by default: prints the projected saving and changes nothing.

    python -m scripts.compact_live_traces                 # report only
    python -m scripts.compact_live_traces --apply         # pack legacy rows
    python -m scripts.compact_live_traces --apply --vacuum

Safe to re-run: packed rows are skipped. Packing runs in short IMMEDIATE
transactions, so a running viewer keeps reading, but stop it before ``--vacuum``
(VACUUM needs the whole file to itself and free disk equal to its size).
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from collections.abc import Callable
from pathlib import Path

from cle.traces.sqlite import DEFAULT_TRACE_PATH, is_packed, pack_blob

# (table, column, primary key columns). These are the columns the store packs on
# write; response_json and payload_json are deliberately absent because SQL
# reads inside them (json_extract / json_each).
PACKED_COLUMNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("live_games", "initial_snapshot", ("game_id",)),
    ("live_steps", "sandbox_snapshot", ("game_id", "step_index")),
    ("live_steps", "result_json", ("game_id", "step_index")),
    ("live_steps", "public_state_json", ("game_id", "step_index")),
    ("model_calls", "request_json", ("game_id", "step_index", "call_index")),
    ("live_failures", "sandbox_snapshot", ("failure_id",)),
    ("live_failures", "public_state_json", ("failure_id",)),
)


def _as_bytes(value: object) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, bytes | bytearray | memoryview):
        return bytes(value)
    raise TypeError(f"unsupported packed-column value: {type(value).__name__}")


def compact(
    path: Path, *, apply: bool, batch_size: int, log: Callable[[str], None] = print
) -> dict[str, int]:
    """Pack every legacy row in PACKED_COLUMNS; return before/after byte totals."""
    before_total = after_total = rows_packed = rows_skipped = 0
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        for table, column, key_columns in PACKED_COLUMNS:
            keys = ", ".join(key_columns)
            rows = connection.execute(
                f"SELECT {keys}, {column} AS value FROM {table} WHERE {column} IS NOT NULL"
            ).fetchall()
            before = after = packed = skipped = 0
            pending: list[tuple[bytes, tuple[object, ...]]] = []

            def flush() -> None:
                if not pending or not apply:
                    pending.clear()
                    return
                where = " AND ".join(f"{name} = ?" for name in key_columns)
                connection.execute("BEGIN IMMEDIATE")
                connection.executemany(
                    f"UPDATE {table} SET {column} = ? WHERE {where}",
                    [(blob, *key) for blob, key in pending],
                )
                connection.commit()
                pending.clear()

            for row in rows:
                value = row["value"]
                size = len(_as_bytes(value))
                before += size
                if is_packed(value):
                    after += size
                    skipped += 1
                    continue
                blob = pack_blob(_as_bytes(value))
                after += len(blob)
                packed += 1
                pending.append((blob, tuple(row[name] for name in key_columns)))
                if len(pending) >= batch_size:
                    flush()
            flush()
            log(
                f"{table}.{column:18s} {len(rows):5d} rows  "
                f"{before / 1048576:8.1f} MB -> {after / 1048576:7.1f} MB  "
                f"(pack {packed}, already packed {skipped})"
            )
            before_total += before
            after_total += after
            rows_packed += packed
            rows_skipped += skipped
    finally:
        connection.close()
    return {
        "before_bytes": before_total,
        "after_bytes": after_total,
        "rows_packed": rows_packed,
        "rows_skipped": rows_skipped,
    }


def vacuum(path: Path, log: Callable[[str], None] = print) -> None:
    connection = sqlite3.connect(path, timeout=60)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        started = time.perf_counter()
        connection.execute("VACUUM")
        # In WAL mode VACUUM rebuilds through the log; the main file only
        # shrinks once that log is checkpointed back into it.
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        log(f"VACUUM finished in {time.perf_counter() - started:.1f}s")
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=Path(DEFAULT_TRACE_PATH))
    parser.add_argument("--apply", action="store_true", help="rewrite legacy rows packed")
    parser.add_argument("--vacuum", action="store_true", help="rebuild the file after packing (needs --apply)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    path = args.db.expanduser().resolve()
    if not path.exists():
        print(f"no trace database at {path}", file=sys.stderr)
        return 2
    if args.vacuum and not args.apply:
        print("--vacuum requires --apply", file=sys.stderr)
        return 2

    size_before = os.path.getsize(path)
    print(f"{path}  ({size_before / 1048576:.0f} MB on disk)  mode={'apply' if args.apply else 'dry-run'}")
    totals = compact(path, apply=args.apply, batch_size=args.batch_size)
    saved = totals["before_bytes"] - totals["after_bytes"]
    verb = "saved" if args.apply else "would save"
    print(
        f"{verb} {saved / 1048576:.0f} MB across {totals['rows_packed']} rows "
        f"({totals['rows_skipped']} already packed)"
    )
    if args.apply and args.vacuum:
        vacuum(path)
        print(f"file now {os.path.getsize(path) / 1048576:.0f} MB on disk")
    elif args.apply:
        print("rows are packed; run again with --vacuum to shrink the file itself")
    return 0


if __name__ == "__main__":
    sys.exit(main())
