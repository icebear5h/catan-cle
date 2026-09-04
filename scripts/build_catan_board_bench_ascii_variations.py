#!/usr/bin/env python
"""Build the strict full-graph Catan ASCII-variation smoke dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.catan_board_bench.ascii_variations import (
    build_ascii_variation_dataset,
)


DEFAULT_CONTRACT_DIR = Path(
    "evals/catan_board_bench/datasets/catan_board_bench_100/contracts"
)
DEFAULT_OUTPUT_DIR = Path("evals/catan_board_bench/datasets/ascii_variation_probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--board-count", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_ascii_variation_dataset(
        args.output_dir,
        contract_dir=args.contract_dir,
        board_count=args.board_count,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
