"""Command line entry point for the isolated visual-primitive builder."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.build import build_dataset
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.constants import (
    DEFAULT_CONTRACT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_PREFIX,
)
from scripts.board_bench.shapes import obj

# The pre-split module path stays the advertised program name and description.
PROG = "build_catan_board_bench_piece_visuals.py"
DESCRIPTION = """Build isolated and cropped Catan visual-primitive QA examples.

This is criterion-1 data: can the model see Catan primitives before we ask it to
bind them to atlas ids. It composes examples from the same frontend assets used
by the Python board renderer, plus local board crops from the approved dummy
fixture.
"""

__all__ = ["DESCRIPTION", "PROG", "main"]


def main() -> int:
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--prompt-prefix", default=DEFAULT_PROMPT_PREFIX)
    args = parser.parse_args()

    metadata = build_dataset(
        output_dir=args.output_dir.resolve(),
        contract_path=args.contract.resolve(),
        image_size=args.image_size,
        variants=args.variants,
        prompt_prefix=args.prompt_prefix,
    )
    print(f"wrote_samples={metadata['sample_count']}")
    print(f"wrote_qa={metadata['qa_count']}")
    counts = obj(metadata["category_counts"], "category_counts")
    print("categories=" + ",".join(f"{k}:{v}" for k, v in counts.items()))
    print(args.output_dir)
    return 0
