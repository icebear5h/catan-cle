"""Command line entry point for the terrain readout export."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from data_pipeline.board_recognition.terrain_impl._config import (
    READOUTS_PER_IMAGE,
    SYNTHETIC_IMAGE_SIZE,
)
from data_pipeline.board_recognition.terrain_impl._export import export_terrain_readout


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--readouts-per-image", type=int, default=READOUTS_PER_IMAGE)
    parser.add_argument("--synthetic-train", type=int, default=0, help="Randomised engine layouts to add to train.")
    parser.add_argument("--synthetic-validation", type=int, default=0)
    parser.add_argument("--synthetic-test", type=int, default=0)
    parser.add_argument("--synthetic-seed", type=int, default=20260904)
    parser.add_argument("--image-size", type=int, default=SYNTHETIC_IMAGE_SIZE)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    result = export_terrain_readout(
        args.dataset_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        readouts_per_image=args.readouts_per_image,
        synthetic={"train": args.synthetic_train, "validation": args.synthetic_validation, "test": args.synthetic_test},
        synthetic_seed_value=args.synthetic_seed,
        image_size=args.image_size,
        workers=args.workers,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2, sort_keys=True))
    return 0
