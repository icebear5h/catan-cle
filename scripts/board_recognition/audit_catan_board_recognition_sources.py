#!/usr/bin/env python
"""Audit and lock replay payloads for board-recognition datagen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.board_recognition.sources import (
    DEFAULT_LEAKAGE_LEDGER,
    DEFAULT_REPLAY_DIR,
    DEFAULT_SOURCE_LOCK,
    ReplaySourceAuditError,
    build_replay_source_lock,
    validate_replay_source_lock,
    write_replay_source_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--leakage-ledger", type=Path, default=DEFAULT_LEAKAGE_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--minimum-accepted", type=int, default=40)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate_only:
        if not args.output.is_file():
            raise FileNotFoundError(args.output)
        lock = json.loads(args.output.read_text())
        report = validate_replay_source_lock(lock, minimum_accepted=args.minimum_accepted)
    else:
        try:
            lock = build_replay_source_lock(
                replay_dir=args.replay_dir,
                leakage_ledger=args.leakage_ledger,
                minimum_accepted=args.minimum_accepted,
            )
        except ReplaySourceAuditError:
            raise
        write_replay_source_lock(args.output, lock)
        report = validate_replay_source_lock(lock, minimum_accepted=args.minimum_accepted)
        report["output"] = str(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
