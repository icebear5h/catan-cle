#!/usr/bin/env python
"""Build or validate the ordered production vision-SFT curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.board_recognition.production_curriculum import (
    build_production_curriculum,
    parse_stage_list,
    slice_production_curriculum,
    validate_production_curriculum,
)

DEFAULT_DATASET_DIR = Path("artifacts/generated/board_recognition/replay_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--slice-stages",
        help="Write only these stages (comma separated, curriculum order) from the built file to --slice-output.",
    )
    parser.add_argument("--slice-output", type=Path, help="Output directory for --slice-stages, under the dataset dir.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.slice_stages:
        if args.slice_output is None:
            raise SystemExit("--slice-stages requires --slice-output")
        report = slice_production_curriculum(
            args.dataset_dir,
            stages=parse_stage_list(args.slice_stages),
            output_dir=args.slice_output,
            source_output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    elif args.validate_only:
        report = validate_production_curriculum(args.dataset_dir, output_dir=args.output_dir)
    else:
        report = build_production_curriculum(
            args.dataset_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
