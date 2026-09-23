#!/usr/bin/env python
"""Build or validate the semantic early-to-late density curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.board_recognition.density_curriculum import (
    build_density_curriculum,
    validate_density_curriculum,
)

DEFAULT_SEMANTIC_DIR = Path(
    "artifacts/generated/board_recognition/replay_v1/ms_swift_semantic_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-export-dir", type=Path, default=DEFAULT_SEMANTIC_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate_only:
        report = validate_density_curriculum(
            args.semantic_export_dir,
            curriculum_dir=args.output_dir,
        )
    else:
        report = build_density_curriculum(
            args.semantic_export_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
