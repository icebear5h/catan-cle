"""Command line entry point for the board-recognition curriculum builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.board_recognition.build_catan_board_recognition_curriculum.build import build_dataset
from scripts.board_recognition.build_catan_board_recognition_curriculum.paths import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPEC_PATH,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.validate import (
    validate_dataset,
)

# The pre-split module path stays the advertised program name and description.
PROG = "build_catan_board_recognition_curriculum.py"
DESCRIPTION = "Build and validate dense, symbolic Catan board-recognition curriculum data."

__all__ = ["DESCRIPTION", "PROG", "main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--pairs-per-entity-type", type=int, default=1)
    parser.add_argument("--seed", type=int, default=381_427)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate_only:
        report = validate_dataset(args.output_dir, spec_path=args.spec)
    else:
        report = build_dataset(
            output_dir=args.output_dir,
            spec_path=args.spec,
            image_size=args.image_size,
            pairs_per_entity_type=args.pairs_per_entity_type,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
