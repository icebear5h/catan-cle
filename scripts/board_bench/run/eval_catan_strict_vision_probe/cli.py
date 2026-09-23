"""Command line entry point for the strict raw-vision probe."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.board_bench.run.eval_catan_strict_vision_probe.constants import (
    DEFAULT_DATASET_DIR,
    acquire_run_lock,
    release_run_lock,
    split_csv,
)
from scripts.board_bench.run.eval_catan_strict_vision_probe.dataset import (
    select_questions,
    validate_dataset,
)
from scripts.board_bench.run.eval_catan_strict_vision_probe.execute import run_evaluation
from scripts.board_bench.run.eval_catan_strict_vision_probe.jobs import build_jobs

# The pre-split module path stays the advertised program name and description.
PROG = "eval_catan_strict_vision_probe.py"
DESCRIPTION = "Evaluate hosted VLMs on the strict 60-question raw-image cohort."

__all__ = ["DESCRIPTION", "PROG", "main", "parse_args", "validate_cli_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--provider",
        choices=("novita", "openrouter"),
        default="novita",
    )
    parser.add_argument("--categories", default="")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--request-interval", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.max_questions is not None and args.max_questions <= 0:
        raise SystemExit("--max-questions must be positive")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.max_tokens <= 0 or args.timeout <= 0:
        raise SystemExit("--max-tokens and --timeout must be positive")
    if args.request_interval < 0:
        raise SystemExit("--request-interval must be non-negative")


def main() -> int:
    args = parse_args()
    validate_cli_args(args)
    metadata, manifest, all_questions = validate_dataset(args.dataset_dir)
    questions = select_questions(
        all_questions,
        categories=split_csv(args.categories),
        max_questions=args.max_questions,
    )
    if not questions:
        raise SystemExit("No questions selected")
    jobs = build_jobs(args.dataset_dir, manifest=manifest, questions=questions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path, lock_fd = acquire_run_lock(args.output_dir)
    try:
        run_evaluation(args, metadata=metadata, questions=questions, jobs=jobs)
    finally:
        release_run_lock(lock_path, lock_fd)
    return 0
