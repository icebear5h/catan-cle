"""Command line entry point for the single-piece curriculum export."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from data_pipeline.board_recognition.replay_dataset import DEFAULT_STYLE_PATH
from data_pipeline.board_recognition.single_piece_impl._config import (
    DEFAULT_NEGATIVES,
    EVAL_IMAGES_PER_BOARD_PER_ENTITY,
    TRAIN_IMAGES_PER_BOARD_PER_ENTITY,
)
from data_pipeline.board_recognition.single_piece_impl._export import (
    export_single_piece_curriculum,
)
from data_pipeline.board_recognition.single_piece_impl._topology import parse_negatives


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--style-path", type=Path, default=DEFAULT_STYLE_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--tile-rows",
        action="store_true",
        help="Add tile resource, number, and inverse rows for one tile per image.",
    )
    parser.add_argument(
        "--negatives",
        default=",".join(f"{kind}={count}" for kind, count in DEFAULT_NEGATIVES.items()),
        help="Empty negatives per image by kind, e.g. adjacent=1,far=1.",
    )
    parser.add_argument(
        "--train-images-per-board-per-entity",
        type=int,
        default=TRAIN_IMAGES_PER_BOARD_PER_ENTITY,
    )
    parser.add_argument(
        "--eval-images-per-board-per-entity",
        type=int,
        default=EVAL_IMAGES_PER_BOARD_PER_ENTITY,
    )
    args = parser.parse_args(argv)
    result = export_single_piece_curriculum(
        args.dataset_dir,
        output_dir=args.output_dir,
        style_path=args.style_path,
        overwrite=args.overwrite,
        train_images_per_board_per_entity=args.train_images_per_board_per_entity,
        eval_images_per_board_per_entity=args.eval_images_per_board_per_entity,
        workers=args.workers,
        tile_rows=args.tile_rows,
        negatives=parse_negatives(args.negatives),
    )
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2, sort_keys=True))
    return 0
