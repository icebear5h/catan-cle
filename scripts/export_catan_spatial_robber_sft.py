#!/usr/bin/env python
"""Export deterministic spatial-grounding and robber-localization SFT rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.board_recognition.spatial_robber import (
    DEFAULT_OUTPUT_NAME,
    export_spatial_robber_supplement,
    validate_spatial_robber_supplement,
)


DEFAULT_DATASET_DIR = Path("artifacts/generated/board_recognition/replay_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.dataset_dir / DEFAULT_OUTPUT_NAME
    if args.validate_only:
        report = validate_spatial_robber_supplement(args.dataset_dir, output_dir=output_dir)
    else:
        report = export_spatial_robber_supplement(
            args.dataset_dir,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
