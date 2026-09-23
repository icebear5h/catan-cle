#!/usr/bin/env python
"""Export replay_v1 states as compact atomic Qwen vision-SFT JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.board_recognition.sft import export_qwen_sft, validate_qwen_sft_export

DEFAULT_DATASET_DIR = Path("artifacts/generated/board_recognition/replay_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--queries-per-state", type=int, default=8)
    parser.add_argument("--seed", type=int, default=381_427)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.dataset_dir / "qwen_sft"
    if args.validate_only:
        report = validate_qwen_sft_export(args.dataset_dir, export_dir=output_dir)
    else:
        report = export_qwen_sft(
            args.dataset_dir,
            output_dir=output_dir,
            queries_per_state=args.queries_per_state,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
