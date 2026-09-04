#!/usr/bin/env python
"""Export inverse atlas grounding plus a 2:1 mixed ms-swift corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.board_recognition.inverse_grounding import (
    DEFAULT_OUTPUT_NAME,
    export_replay_v1_ms_swift_bidirectional,
    validate_replay_v1_ms_swift_bidirectional,
)


DEFAULT_DATASET_DIR = Path("artifacts/generated/board_recognition/replay_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--forward-projection-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.dataset_dir / DEFAULT_OUTPUT_NAME
    if args.validate_only:
        report = validate_replay_v1_ms_swift_bidirectional(
            args.dataset_dir,
            export_dir=output_dir,
        )
    else:
        report = export_replay_v1_ms_swift_bidirectional(
            args.dataset_dir,
            output_dir=output_dir,
            forward_projection_dir=args.forward_projection_dir,
            overwrite=args.overwrite,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
