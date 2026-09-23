"""Command line entry point for the node and edge readout export."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from data_pipeline.board_recognition.node_edge_impl._config import (
    READOUTS_PER_FAMILY,
    ROWS_PER_FAMILY,
)
from data_pipeline.board_recognition.node_edge_impl._export import export_node_edge_readout


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rows-per-family", type=int, default=ROWS_PER_FAMILY, help="Occupied and empty locations sampled per family per train image.")
    parser.add_argument("--readouts-per-family", type=int, default=READOUTS_PER_FAMILY)
    parser.add_argument("--train-full-coverage", action="store_true", help="Every node and edge of every train image: a pool for the rung mixer.")
    parser.add_argument("--eval-coverage", choices=("balanced", "full"), default="balanced", help="Eval splits: every occupied location plus as many hardest-first empties (balanced), or every location (full).")
    args = parser.parse_args(argv)
    result = export_node_edge_readout(
        args.dataset_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        rows_per_family=args.rows_per_family,
        readouts_per_family=args.readouts_per_family,
        train_full_coverage=args.train_full_coverage,
        eval_coverage=args.eval_coverage,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2, sort_keys=True))
    return 0
