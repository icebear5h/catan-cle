"""Command line entry point for the text-format representation eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.catan_board_bench.scoring import VISUAL_CATEGORIES
from evals.catan_board_bench.text_representations import REPRESENTATION_NAMES
from scripts.board_bench.run.eval_catan_board_bench_text_formats.constants import (
    build_prompt,
    split_csv,
)
from scripts.board_bench.run.eval_catan_board_bench_text_formats.run import (
    build_plan,
    check_representations,
    render_boards,
    run_requests,
    select_text_questions,
)
from scripts.board_bench.shapes import text, write_json

# The pre-split module path stays the advertised program name and description.
PROG = "eval_catan_board_bench_text_formats.py"
DESCRIPTION = "Evaluate information-equivalent Catan board text representations."

__all__ = ["DESCRIPTION", "PROG", "main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument(
        "--representations",
        default=",".join(REPRESENTATION_NAMES),
    )
    parser.add_argument("--categories", default=",".join(VISUAL_CATEGORIES))
    parser.add_argument("--limit-samples", type=int, default=10)
    parser.add_argument("--questions-per-sample", type=int, default=11)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--provider-order",
        default="AkashML",
        help="Comma-separated OpenRouter providers; empty leaves routing unpinned.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    representations = split_csv(args.representations)
    check_representations(representations)
    categories = split_csv(args.categories)
    selected = select_text_questions(args, categories)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    representation_dir = args.output_dir / "representations"
    representation_dir.mkdir(exist_ok=True)

    rendered = render_boards(selected, representations, representation_dir)
    plan = build_plan(args, representations, categories, selected, rendered.digests)
    write_json(args.output_dir / "plan.json", plan)
    print(json.dumps(plan, indent=2))

    if args.dry_run:
        first = selected[0]
        sample_id = text(first["sample_id"], "sample_id")
        for representation in representations:
            print(f"\n--- {representation} ---")
            print(
                build_prompt(
                    representation,
                    rendered.texts[(sample_id, representation)],
                    first,
                )[:4000]
            )
        return

    run_requests(args, representations, selected, rendered, plan)
