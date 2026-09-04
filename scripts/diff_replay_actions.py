#!/usr/bin/env python3
"""Run a causal full-game model-versus-human action-selection diff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.replay_action_diff import run_action_diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-player",
        default="captured",
        help="'captured' or a Colonist color ID (default: captured)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify the complete replay without making provider calls.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("off", "minimal", "low", "medium", "high", "xhigh", "max"),
        default="xhigh",
        help="Explicit native-reasoning condition (default: xhigh).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum completion tokens per request (default: 8192).",
    )
    parser.add_argument(
        "--max-requests-per-model",
        type=int,
        help="Bound new calls per model for a resumable smoke run.",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        help="Stop scheduling later decision batches once recorded spend reaches this cap.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry prior rows that contain provider errors.",
    )
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.max_requests_per_model is not None and args.max_requests_per_model < 1:
        parser.error("--max-requests-per-model must be positive")
    if args.max_cost_usd is not None and args.max_cost_usd <= 0:
        parser.error("--max-cost-usd must be positive")
    return args


def main() -> None:
    args = parse_args()
    summary = run_action_diff(
        game_id=args.game_id,
        models=args.models,
        output_dir=args.output_dir,
        target_player=args.target_player,
        dry_run=args.dry_run,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        max_requests_per_model=args.max_requests_per_model,
        max_cost_usd=args.max_cost_usd,
        retry_errors=args.retry_errors,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
