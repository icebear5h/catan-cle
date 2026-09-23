"""Subprocess command construction for each sweep phase."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from scripts.board_bench.run.run_catan_openrouter_model_sweep.catalog import model_key
from scripts.board_bench.run.run_catan_openrouter_model_sweep.costs import track_cost
from scripts.board_bench.shapes import JsonDict, obj, text, values

__all__ = [
    "board_command",
    "reasoning_trace_command",
    "run_command",
    "strategy_reasoning_trace_command",
]


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def board_command(
    config: JsonDict,
    output_dir: Path,
    model_id: str,
    *,
    full: bool,
) -> list[str]:
    board = obj(config["board_reading"], "board_reading")
    destination = output_dir / ("board_full" if full else "board_preflight") / model_key(model_id)
    command = [
        sys.executable,
        "-m",
        "scripts.board_bench.run.eval_catan_strict_vision_probe",
        "--provider",
        "openrouter",
        "--dataset-dir",
        text(board["dataset_dir"], "dataset_dir"),
        "--model",
        model_id,
        "--max-tokens",
        str(board["max_tokens"]),
        "--temperature",
        str(board["temperature"]),
        "--concurrency",
        "1",
        "--output-dir",
        str(destination),
    ]
    if full:
        if (destination / "responses.jsonl").exists():
            command.append("--resume")
    else:
        command.extend(("--categories", "node_state", "--max-questions", "1"))
        if (destination / "responses.jsonl").exists():
            command.append("--resume")
    return command


def reasoning_trace_command(
    config: JsonDict,
    output_dir: Path,
    models: Iterable[str],
    *,
    preflight: bool,
) -> list[str]:
    reasoning = obj(config["initial_settlement_reasoning"], "initial_settlement_reasoning")
    allocations = obj(config["budget_allocation_usd"], "budget_allocation_usd")
    prior_cost = track_cost(output_dir, "initial_placement")
    command = [
        sys.executable,
        "-m",
        "scripts.reasoning.eval_catan_initial_settlement_reasoning",
        "--models",
        *models,
        "--seeds",
        *(str(seed) for seed in values(reasoning["seeds"], "seeds")),
        "--output-dir",
        str(output_dir / "initial_settlement_reasoning_traces"),
        "--catalog-snapshot",
        str(output_dir / "openrouter_catalog_snapshot.json"),
        "--reasoning-effort",
        str(obj(reasoning["reasoning"], "reasoning")["effort"]),
        "--temperature",
        str(reasoning["temperature"]),
        "--max-cost-usd",
        str(allocations["initial_placement"]),
        "--prior-cost-usd",
        str(prior_cost),
        "--prompt-token-allowance",
        str(reasoning["prompt_token_allowance"]),
    ]
    if preflight:
        command.extend(
            ("--max-seeds", str(len(values(reasoning["preflight_seeds"], "preflight_seeds"))))
        )
    return command


def strategy_reasoning_trace_command(
    config: JsonDict,
    output_dir: Path,
    models: Iterable[str],
) -> list[str]:
    guided = obj(config["strategy_guided_reasoning"], "strategy_guided_reasoning")
    allocations = obj(config["budget_allocation_usd"], "budget_allocation_usd")
    prior_cost = track_cost(output_dir, "initial_placement") + track_cost(
        output_dir, "initial_settlement_reasoning_traces"
    )
    return [
        sys.executable,
        "-m",
        "scripts.reasoning.eval_catan_initial_settlement_reasoning",
        "--models",
        *models,
        "--seeds",
        *(str(seed) for seed in values(guided["seeds"], "seeds")),
        "--output-dir",
        str(output_dir / text(guided["output_dirname"], "output_dirname")),
        "--catalog-snapshot",
        str(output_dir / "openrouter_catalog_snapshot.json"),
        "--prompt-variant",
        str(guided["prompt_variant"]),
        "--reasoning-effort",
        str(obj(guided["reasoning"], "reasoning")["effort"]),
        "--temperature",
        str(guided["temperature"]),
        "--max-cost-usd",
        str(allocations["initial_placement"]),
        "--prior-cost-usd",
        str(prior_cost),
        "--prompt-token-allowance",
        str(guided["prompt_token_allowance"]),
    ]
