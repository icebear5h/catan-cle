#!/usr/bin/env python3
"""Plan, run, or verify GPT-grounded narrator transcript reasoning."""

import argparse
import json
from pathlib import Path

from evals.transcript_reasoning import (
    DEFAULT_MODEL,
    build_generation_plan,
    run_generation,
    scan_reasoning_jobs,
    verify_generation_artifact,
)
from scripts.reasoning.jsonio import write_json

DEFAULT_OUTPUT_DIR = Path(
    "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI/"
    "narrator_reasoning/gpt_5_6_sol_20260820"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build board-grounded reasoning paragraphs from a paired transcript."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("plan", "run"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--game-id", default="242781000")
        command_parser.add_argument("--model", default=DEFAULT_MODEL)
        command_parser.add_argument(
            "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
        )
        if command == "run":
            command_parser.add_argument("--workers", type=int, default=4)
            command_parser.add_argument("--max-tokens", type=int, default=1200)
            command_parser.add_argument("--timeout", type=float, default=180.0)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "verify":
        print(json.dumps(verify_generation_artifact(args.output_dir), indent=2))
        return

    scan = scan_reasoning_jobs(args.game_id)
    plan = build_generation_plan(scan, args.model)
    if args.command == "plan":
        write_json(args.output_dir / "plan.json", plan)
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "game_id": plan["game_id"],
                    "model_id": plan["model_id"],
                    "job_count": plan["job_count"],
                    "transcript_sha256": plan["transcript_sha256"],
                },
                indent=2,
            )
        )
        return

    summary = run_generation(
        args.output_dir,
        scan,
        plan,
        workers=args.workers,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
