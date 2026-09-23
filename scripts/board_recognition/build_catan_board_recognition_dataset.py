#!/usr/bin/env python
"""Build or validate the engine-backed replay_v1 recognition corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.board_recognition.replay_dataset import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    build_replay_v1_dataset,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.sources import DEFAULT_SOURCE_LOCK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-rerender", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate_only:
        report = validate_replay_v1_dataset(
            args.output_dir,
            rerender=not args.skip_rerender,
        )
    else:
        report = build_replay_v1_dataset(
            output_dir=args.output_dir,
            source_lock_path=args.source_lock,
            image_size=args.image_size,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
