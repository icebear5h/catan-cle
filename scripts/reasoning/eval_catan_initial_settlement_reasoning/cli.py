"""Argument parsing and the sequential, budget-capped probe run."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from cle.game_engine.models.player import Color
from cle.harness.reasoning import native_reasoning_request
from cle.harness.suite import load_context_suite
from scripts.reasoning.eval_catan_initial_settlement_reasoning.artifacts import (
    JsonDict,
    existing_traces,
    json_float,
    read_json,
    trace_path,
    utc_now,
    write_json,
    write_json_once,
)
from scripts.reasoning.eval_catan_initial_settlement_reasoning.capture import capture_trace
from scripts.reasoning.eval_catan_initial_settlement_reasoning.costs import (
    catalog_cost_bounds,
    usage_cost_usd,
)
from scripts.reasoning.eval_catan_initial_settlement_reasoning.reports import (
    write_indexes,
    write_trace,
)
from scripts.reasoning.eval_catan_initial_settlement_reasoning.seeds import (
    COLORS,
    build_seed_input,
    load_prompt_variant,
)

PLAN_SCHEMA = "catan-initial-settlement-reasoning-plan/v1"

# The pre-split module path stays the advertised program name and description.
PROG = "eval_catan_initial_settlement_reasoning.py"
DESCRIPTION = (
    "Capture native reasoning traces on fresh seeded first-settlement choices."
)

__all__ = ["DESCRIPTION", "PLAN_SCHEMA", "PROG", "main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--max-seeds",
        type=int,
        help="Run only the first N planned seeds while keeping one immutable plan.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catalog-snapshot", type=Path, required=True)
    parser.add_argument(
        "--prompt-variant",
        type=Path,
        help="Eval-only phase-guidance variant; the default suite is not mutated.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
        default="high",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--prior-cost-usd", type=float, default=0.0)
    parser.add_argument("--prompt-token-allowance", type=int, default=20_000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be within 0..2")
    if args.max_cost_usd <= 0:
        parser.error("--max-cost-usd must be positive")
    if args.prior_cost_usd < 0:
        parser.error("--prior-cost-usd cannot be negative")
    if args.prior_cost_usd >= args.max_cost_usd:
        parser.error("--prior-cost-usd must be below --max-cost-usd")
    if args.prompt_token_allowance <= 0:
        parser.error("--prompt-token-allowance must be positive")
    if len(args.models) != len(set(args.models)):
        parser.error("--models cannot contain duplicates")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds cannot contain duplicates")
    if args.max_seeds is not None and args.max_seeds <= 0:
        parser.error("--max-seeds must be positive")
    return args


def _build_plan(
    args: argparse.Namespace,
    suite_identity: str,
    bounds: dict[str, JsonDict],
    prompt_variant: JsonDict | None,
) -> JsonDict:
    plan: JsonDict = {
        "schema": PLAN_SCHEMA,
        "generated_at": utc_now(),
        "models": list(args.models),
        "seeds": list(args.seeds),
        "colors": [color.value for color in COLORS],
        "actor": Color.BLUE.value,
        "decision": "fresh first settlement",
        "context_suite": suite_identity,
        "board_surface": "indexed_tile_rows/v3",
        "reasoning_request": dict(native_reasoning_request(args.reasoning_effort)),
        "temperature": args.temperature,
        "max_tokens_omitted": True,
        "timeout_seconds": 900.0,
        "max_retries": 0,
        "max_cost_usd": args.max_cost_usd,
        "prior_cost_usd": args.prior_cost_usd,
        "prompt_token_allowance": args.prompt_token_allowance,
        "catalog_snapshot": str(args.catalog_snapshot),
        "request_cost_upper_bounds": {key: value for key, value in bounds.items()},
        "scheduling": "sequential",
        "human_action_labels": False,
        "replay_input": False,
    }
    if prompt_variant is not None:
        plan["prompt_variant"] = prompt_variant
    return plan


def main() -> int:
    args = parse_args()
    base_suite = load_context_suite()
    suite = base_suite
    prompt_variant = None
    if args.prompt_variant is not None:
        suite, prompt_variant = load_prompt_variant(args.prompt_variant, base_suite)
    catalog = read_json(args.catalog_snapshot)
    bounds = catalog_cost_bounds(
        catalog,
        args.models,
        prompt_token_allowance=args.prompt_token_allowance,
    )
    plan = _build_plan(
        args, f"{suite.id}@{suite.version}", bounds, prompt_variant
    )
    plan_path = args.output_dir / "plan.json"
    if plan_path.exists():
        existing = read_json(plan_path)
        comparable = {
            key: value for key, value in plan.items() if key != "generated_at"
        }
        existing_comparable = {
            key: value for key, value in existing.items() if key != "generated_at"
        }
        if existing_comparable != comparable:
            raise RuntimeError("Cannot resume after direct reasoning plan changed")
    else:
        write_json(plan_path, plan)

    inputs = [
        build_seed_input(
            seed,
            suite=suite,
            prompt_variant=prompt_variant,
        )
        for seed in args.seeds
    ]
    for seed_input in inputs:
        write_json_once(
            args.output_dir / "inputs" / f"seed_{seed_input.manifest['seed']}.json",
            seed_input.manifest,
        )
    write_indexes(args.output_dir, plan)
    if args.dry_run:
        return 0

    load_dotenv()
    spent = args.prior_cost_usd + sum(
        usage_cost_usd(trace) for trace in existing_traces(args.output_dir)
    )
    selected_inputs = inputs[: args.max_seeds] if args.max_seeds else inputs
    for seed_input in selected_inputs:
        for model_id in args.models:
            seed = seed_input.manifest["seed"]
            path = trace_path(args.output_dir, int(str(seed)), model_id)
            if path.exists():
                continue
            request_bound = json_float(
                bounds[model_id]["request_upper_bound_usd"], "request upper bound"
            )
            if spent + request_bound > args.max_cost_usd:
                raise RuntimeError(
                    f"Worst-case request for {model_id} would exceed the cost cap: "
                    f"${spent:.6f} spent/reserved + ${request_bound:.6f} bound > "
                    f"${args.max_cost_usd:.6f}"
                )
            print(
                f"seed={seed} model={model_id} "
                f"worst_case=${request_bound:.6f}",
                flush=True,
            )
            trace = asyncio.run(
                capture_trace(
                    seed_input,
                    model_id,
                    reasoning_effort=args.reasoning_effort,
                    temperature=args.temperature,
                )
            )
            write_trace(args.output_dir, trace)
            spent += usage_cost_usd(trace)
            if spent > args.max_cost_usd:
                raise RuntimeError("Recorded cost exceeded the direct-trace budget")
            write_indexes(args.output_dir, plan)

    write_indexes(args.output_dir, plan)
    return 0
