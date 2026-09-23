"""Command line entry point for the strict vision projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.board_bench.builders.render_catan_strict_vision_probe.constants import (
    DEFAULT_BOARD_CANVAS_FRACTION,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    DEFAULT_VIEW_PADDING_FACTOR,
)
from scripts.board_bench.builders.render_catan_strict_vision_probe.render import (
    render_strict_vision_probe,
)

# The pre-split module path stays the advertised program name and description.
PROG = "render_catan_strict_vision_probe.py"
DESCRIPTION = "Project the locked strict 60-question text probe onto raw engine screenshots."

__all__ = ["DESCRIPTION", "PROG", "main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--view-padding-factor",
        type=float,
        default=DEFAULT_VIEW_PADDING_FACTOR,
    )
    parser.add_argument(
        "--target-board-canvas-fraction",
        type=float,
        default=DEFAULT_BOARD_CANVAS_FRACTION,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = render_strict_vision_probe(
        args.source_dir,
        args.output_dir,
        image_size=args.image_size,
        view_padding_factor=args.view_padding_factor,
        target_board_canvas_fraction=args.target_board_canvas_fraction,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0
