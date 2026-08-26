#!/usr/bin/env python
"""Build the strict six-format full-graph Catan probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.catan_board_bench.full_graph_format_probe import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    build_full_graph_format_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_full_graph_format_probe(
        args.output_dir,
        source_dir=args.source_dir,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
