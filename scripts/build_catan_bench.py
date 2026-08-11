#!/usr/bin/env python
"""Build CatanBench from local Colonist replay fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

from catanbench.builder import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUESTION_DIR,
    DEFAULT_REPLAY_DIR,
    build_catanbench_sync,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--question-dir", type=Path, default=DEFAULT_QUESTION_DIR)
    parser.add_argument("--min-step", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--render-images", action="store_true")
    parser.add_argument("--no-quiet-replay", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_catanbench_sync(
        replay_dir=args.replay_dir,
        output_dir=args.output_dir,
        question_dir=args.question_dir,
        samples=args.samples,
        min_step=args.min_step,
        render_images=args.render_images,
        image_size=args.image_size,
        quiet_replay=not args.no_quiet_replay,
    )
    print(f"Built {result.sample_count} samples and {result.qa_count} QA rows")
    print(f"Rendered images: {result.rendered_images}")
    print(f"Output: {result.output_dir}")
    print(f"Question dir: {args.question_dir}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Questions: {result.questions_path}")
    print(f"Answer key: {result.answer_key_path}")


if __name__ == "__main__":
    main()
