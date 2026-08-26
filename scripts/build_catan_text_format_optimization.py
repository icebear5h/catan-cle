#!/usr/bin/env python
"""Build the query-indexed Catan text-format optimization probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.catan_board_bench.text_format_optimization import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    build_text_format_optimization_probe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    metadata = build_text_format_optimization_probe(
        args.output_dir,
        source_dir=args.source_dir,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
