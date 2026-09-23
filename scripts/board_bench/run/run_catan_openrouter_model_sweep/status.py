"""Sweep status artifact with the budget guard."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.board_bench.run.run_catan_openrouter_model_sweep.catalog import write_json
from scripts.board_bench.run.run_catan_openrouter_model_sweep.costs import track_cost
from scripts.board_bench.shapes import JsonDict, membership, number, obj, objs, text

__all__ = ["update_status"]


def update_status(
    config: JsonDict,
    output_dir: Path,
    selected_catalog: list[JsonDict],
) -> None:
    guided = obj(config["strategy_guided_reasoning"], "strategy_guided_reasoning")
    excluded = membership(guided["excluded_models"], "excluded_models")
    candidates = objs(config["candidates"], "candidates")
    board_cost = track_cost(output_dir, "board_preflight") + track_cost(
        output_dir, "board_full"
    )
    capped_replay_control_cost = track_cost(output_dir, "initial_placement")
    direct_reasoning_cost = track_cost(
        output_dir, "initial_settlement_reasoning_traces"
    )
    strategy_reasoning_cost = track_cost(
        output_dir,
        text(guided["output_dirname"], "output_dirname"),
    )
    placement_cost = (
        capped_replay_control_cost
        + direct_reasoning_cost
        + strategy_reasoning_cost
    )
    total_cost = board_cost + placement_cost
    budget = number(config["authorized_budget_usd"], "authorized_budget_usd")
    status: JsonDict = {
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
            "total": total_cost,
        },
        "candidate_count": len(candidates),
        "placement_reasoning_candidate_count": sum(
            bool(row.get("initial_placement_reasoning")) for row in candidates
        ),
        "strategy_reasoning_candidate_count": sum(
            bool(row.get("initial_placement_reasoning"))
            and row["model_id"] not in excluded
            for row in candidates
        ),
        "catalog": list(selected_catalog),
    }
    if total_cost > budget:
        raise RuntimeError("Recorded model-sweep cost exceeds the authorized budget")
    write_json(output_dir / "status.json", status)
