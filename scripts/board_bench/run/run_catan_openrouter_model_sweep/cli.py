"""Command line entry point for the locked OpenRouter model sweep."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from scripts.board_bench.run.run_catan_openrouter_model_sweep.catalog import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    OPENROUTER_MODELS_URL,
    fetch_catalog,
    read_json,
    validate_config,
    write_json,
)
from scripts.board_bench.run.run_catan_openrouter_model_sweep.commands import (
    board_command,
    reasoning_trace_command,
    run_command,
    strategy_reasoning_trace_command,
)
from scripts.board_bench.run.run_catan_openrouter_model_sweep.costs import track_cost
from scripts.board_bench.run.run_catan_openrouter_model_sweep.status import update_status
from scripts.board_bench.shapes import JsonDict, membership, number, obj, objs, text

# The pre-split module path stays the advertised program name and description.
PROG = "run_catan_openrouter_model_sweep.py"
DESCRIPTION = "Run the locked OpenRouter Catan VLM board/placement sweep."

__all__ = ["DESCRIPTION", "PROG", "main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--phase",
        choices=(
            "preflight",
            "board-full",
            "reasoning-preflight",
            "reasoning-seeds",
            "reasoning-strategy",
        ),
        required=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _model_lists(config: JsonDict) -> tuple[list[str], list[str], list[str]]:
    candidates = objs(config["candidates"], "candidates")
    guided = obj(config["strategy_guided_reasoning"], "strategy_guided_reasoning")
    excluded = membership(guided["excluded_models"], "excluded_models")
    board_models = [
        text(row["model_id"], "model_id") for row in candidates if row.get("board_reading")
    ]
    placement_models = [
        text(row["model_id"], "model_id")
        for row in candidates
        if row.get("initial_placement_reasoning")
    ]
    strategy_models = [
        model_id for model_id in placement_models if model_id not in excluded
    ]
    return board_models, placement_models, strategy_models


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    catalog = fetch_catalog()
    selected_catalog = validate_config(config, catalog)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_snapshot = args.output_dir / "openrouter_catalog_snapshot.json"
    if not catalog_snapshot.exists():
        write_json(
            catalog_snapshot,
            {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "source": OPENROUTER_MODELS_URL,
                "models": list(selected_catalog),
            },
        )

    board_models, placement_models, strategy_models = _model_lists(config)

    if args.phase in {"preflight", "board-full"}:
        full = args.phase == "board-full"
        allocations = obj(config["budget_allocation_usd"], "budget_allocation_usd")
        budget = number(allocations["board_reading"], "board_reading budget")
        for model_id in board_models:
            if track_cost(args.output_dir, "board_preflight") + track_cost(
                args.output_dir, "board_full"
            ) >= budget:
                raise RuntimeError("Board-reading track reached its budget allocation")
            run_command(
                board_command(config, args.output_dir, model_id, full=full),
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                update_status(config, args.output_dir, selected_catalog)

    if args.phase in {"preflight", "reasoning-preflight", "reasoning-seeds"}:
        run_command(
            reasoning_trace_command(
                config,
                args.output_dir,
                placement_models,
                preflight=args.phase != "reasoning-seeds",
            ),
            dry_run=args.dry_run,
        )

    if args.phase == "reasoning-strategy":
        run_command(
            strategy_reasoning_trace_command(
                config,
                args.output_dir,
                strategy_models,
            ),
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        update_status(config, args.output_dir, selected_catalog)
    return 0
