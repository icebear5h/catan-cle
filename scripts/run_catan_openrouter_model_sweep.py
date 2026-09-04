#!/usr/bin/env python3
"""Run the locked OpenRouter Catan VLM board/placement sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx


DEFAULT_CONFIG = Path(
    "configs/model_sweeps/openrouter_vlm_27b_72b_v1.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/runs/model_selection/openrouter_vlm_27b_72b_20260831"
)
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def model_key(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")


def fetch_catalog() -> dict[str, Any]:
    response = httpx.get(OPENROUTER_MODELS_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("OpenRouter model catalog has an unexpected shape")
    return payload


def validate_config(
    config: dict[str, Any],
    catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    if config.get("schema") != "catan-openrouter-model-sweep/v1":
        raise ValueError("Unsupported model-sweep config schema")
    budget = config.get("authorized_budget_usd")
    allocations = config.get("budget_allocation_usd") or {}
    if not isinstance(budget, (int, float)) or budget <= 0:
        raise ValueError("authorized_budget_usd must be positive")
    if sum(allocations.values()) > budget:
        raise ValueError("Track budget allocations exceed the authorized budget")

    by_id = {row.get("id"): row for row in catalog["data"]}
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Model sweep has no candidates")
    seen = set()
    selected_catalog = []
    for candidate in candidates:
        model_id = candidate.get("model_id")
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            raise ValueError(f"Invalid or duplicate model ID: {model_id!r}")
        seen.add(model_id)
        model = by_id.get(model_id)
        if model is None:
            raise ValueError(f"OpenRouter catalog does not contain {model_id}")
        architecture = model.get("architecture") or {}
        if "image" not in (architecture.get("input_modalities") or []):
            raise ValueError(f"Candidate does not advertise image input: {model_id}")
        selected_catalog.append(
            {
                "model_id": model_id,
                "name": model.get("name"),
                "context_length": model.get("context_length"),
                "architecture": architecture,
                "supported_parameters": model.get("supported_parameters") or [],
                "pricing": model.get("pricing") or {},
                "top_provider": model.get("top_provider") or {},
            }
        )
    return selected_catalog


def response_cost_usd(row: dict[str, Any]) -> float:
    usage = row.get("usage")
    if not isinstance(usage, dict):
        result = row.get("result")
        usage = result.get("usage") if isinstance(result, dict) else None
    if not isinstance(usage, dict):
        response = row.get("response")
        usage = response.get("usage") if isinstance(response, dict) else None
    value = usage.get("cost") if isinstance(usage, dict) else None
    return float(value) if isinstance(value, (int, float)) and value >= 0 else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def track_cost(output_dir: Path, track: str) -> float:
    root = output_dir / track
    if not root.exists():
        return 0.0
    response_rows = (
        row
        for path in root.rglob("responses.jsonl")
        for row in read_jsonl(path)
    )
    trace_rows = (
        read_json(path)
        for path in (root / "traces").glob("seed_*/*.json")
    ) if (root / "traces").exists() else ()
    return sum(response_cost_usd(row) for row in response_rows) + sum(
        response_cost_usd(row) for row in trace_rows
    )


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def board_command(
    config: dict[str, Any],
    output_dir: Path,
    model_id: str,
    *,
    full: bool,
) -> list[str]:
    board = config["board_reading"]
    destination = output_dir / ("board_full" if full else "board_preflight") / model_key(model_id)
    command = [
        sys.executable,
        "scripts/eval_catan_strict_vision_probe.py",
        "--provider",
        "openrouter",
        "--dataset-dir",
        board["dataset_dir"],
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
    config: dict[str, Any],
    output_dir: Path,
    models: Iterable[str],
    *,
    preflight: bool,
) -> list[str]:
    reasoning = config["initial_settlement_reasoning"]
    prior_cost = track_cost(output_dir, "initial_placement")
    command = [
        sys.executable,
        "scripts/eval_catan_initial_settlement_reasoning.py",
        "--models",
        *models,
        "--seeds",
        *(str(seed) for seed in reasoning["seeds"]),
        "--output-dir",
        str(output_dir / "initial_settlement_reasoning_traces"),
        "--catalog-snapshot",
        str(output_dir / "openrouter_catalog_snapshot.json"),
        "--reasoning-effort",
        str(reasoning["reasoning"]["effort"]),
        "--temperature",
        str(reasoning["temperature"]),
        "--max-cost-usd",
        str(config["budget_allocation_usd"]["initial_placement"]),
        "--prior-cost-usd",
        str(prior_cost),
        "--prompt-token-allowance",
        str(reasoning["prompt_token_allowance"]),
    ]
    if preflight:
        command.extend(("--max-seeds", str(len(reasoning["preflight_seeds"]))))
    return command


def strategy_reasoning_trace_command(
    config: dict[str, Any],
    output_dir: Path,
    models: Iterable[str],
) -> list[str]:
    guided = config["strategy_guided_reasoning"]
    prior_cost = track_cost(output_dir, "initial_placement") + track_cost(
        output_dir, "initial_settlement_reasoning_traces"
    )
    return [
        sys.executable,
        "scripts/eval_catan_initial_settlement_reasoning.py",
        "--models",
        *models,
        "--seeds",
        *(str(seed) for seed in guided["seeds"]),
        "--output-dir",
        str(output_dir / guided["output_dirname"]),
        "--catalog-snapshot",
        str(output_dir / "openrouter_catalog_snapshot.json"),
        "--prompt-variant",
        str(guided["prompt_variant"]),
        "--reasoning-effort",
        str(guided["reasoning"]["effort"]),
        "--temperature",
        str(guided["temperature"]),
        "--max-cost-usd",
        str(config["budget_allocation_usd"]["initial_placement"]),
        "--prior-cost-usd",
        str(prior_cost),
        "--prompt-token-allowance",
        str(guided["prompt_token_allowance"]),
    ]


def update_status(
    config: dict[str, Any],
    output_dir: Path,
    selected_catalog: list[dict[str, Any]],
) -> None:
    board_cost = track_cost(output_dir, "board_preflight") + track_cost(
        output_dir, "board_full"
    )
    capped_replay_control_cost = track_cost(output_dir, "initial_placement")
    direct_reasoning_cost = track_cost(
        output_dir, "initial_settlement_reasoning_traces"
    )
    strategy_reasoning_cost = track_cost(
        output_dir,
        config["strategy_guided_reasoning"]["output_dirname"],
    )
    placement_cost = (
        capped_replay_control_cost
        + direct_reasoning_cost
        + strategy_reasoning_cost
    )
    status = {
        "schema": "catan-openrouter-model-sweep-status/v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config_id": config["id"],
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "authorized_budget_usd": config["authorized_budget_usd"],
        "cost_usd": {
            "board_reading": board_cost,
            "superseded_capped_replay_control": capped_replay_control_cost,
            "fresh_initial_settlement_reasoning": direct_reasoning_cost,
            "strategy_guided_reasoning": strategy_reasoning_cost,
            "initial_placement_total": placement_cost,
            "total": board_cost + placement_cost,
        },
        "candidate_count": len(config["candidates"]),
        "placement_reasoning_candidate_count": sum(
            bool(row.get("initial_placement_reasoning"))
            for row in config["candidates"]
        ),
        "strategy_reasoning_candidate_count": sum(
            bool(row.get("initial_placement_reasoning"))
            and row["model_id"]
            not in config["strategy_guided_reasoning"]["excluded_models"]
            for row in config["candidates"]
        ),
        "catalog": selected_catalog,
    }
    if status["cost_usd"]["total"] > config["authorized_budget_usd"]:
        raise RuntimeError("Recorded model-sweep cost exceeds the authorized budget")
    write_json(output_dir / "status.json", status)


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
                "models": selected_catalog,
            },
        )

    board_models = [
        row["model_id"]
        for row in config["candidates"]
        if row.get("board_reading")
    ]
    placement_models = [
        row["model_id"]
        for row in config["candidates"]
        if row.get("initial_placement_reasoning")
    ]
    strategy_models = [
        model_id
        for model_id in placement_models
        if model_id
        not in config["strategy_guided_reasoning"]["excluded_models"]
    ]

    if args.phase in {"preflight", "board-full"}:
        full = args.phase == "board-full"
        budget = config["budget_allocation_usd"]["board_reading"]
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


if __name__ == "__main__":
    raise SystemExit(main())
