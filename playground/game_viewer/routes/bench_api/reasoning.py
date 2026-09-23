"""Building the reasoning-trace payloads for one allowlisted run."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from flask import Response, jsonify, request

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.game_engine.public_board import snapshot_public_board
from evals.catan_board_bench.annotations import (
    RenderState,
    contract_to_render_state,
)

from .common import (
    _coerce_float,
    _coerce_int,
    _map,
    _read_json,
    _seq,
)
from .paths import (
    PROJECT_ROOT,
)


def _reasoning_trace_render_state(
    trace: dict[str, object],
) -> tuple[RenderState, str]:
    trace_input = trace.get("input")
    if not isinstance(trace_input, dict):
        raise ValueError("Reasoning trace has no input contract")
    seed = trace_input.get("seed")
    raw_colors = trace_input.get("colors")
    actor_name = trace_input.get("actor")
    board = trace_input.get("board_presentation")
    if not isinstance(seed, int):
        raise ValueError("Reasoning trace seed is invalid")
    if not isinstance(raw_colors, list) or len(raw_colors) != 4:
        raise ValueError("Reasoning trace color order is invalid")
    try:
        colors = tuple(Color[str(value)] for value in raw_colors)
        actor = Color[str(actor_name)]
    except KeyError as exc:
        raise ValueError("Reasoning trace contains an unknown color") from exc
    if actor != colors[0]:
        raise ValueError("Reasoning trace actor is not first in player order")
    if not isinstance(board, dict) or not isinstance(board.get("board_sha256"), str):
        raise ValueError("Reasoning trace has no board SHA-256")

    engine = GameEngine(colors, seed=seed, shuffle_players=False)
    snapshot = snapshot_public_board(engine.observe(actor))
    expected_sha256 = board["board_sha256"]
    if snapshot.facts_sha256 != expected_sha256:
        raise ValueError("Reconstructed board does not match reasoning trace SHA-256")
    return contract_to_render_state(snapshot.contract()), snapshot.facts_sha256


def _confined_reasoning_trace_path(raw_path: object, run_dir: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Reasoning trace index has no JSON path")
    relative = Path(raw_path)
    if relative.is_absolute() or relative.suffix != ".json":
        raise ValueError("Reasoning trace index contains an invalid JSON path")
    path = (run_dir / relative).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError("Reasoning trace path is outside the run directory") from exc
    return path


def reasoning_traces_payload(
    run_id: str,
    run_config: Mapping[str, object],
    catalog: list[dict[str, object]],
) -> Response | tuple[Response, int]:
    """The run's trace catalogue, once the caller has resolved the run."""
    run_dir = cast(Path, run_config["path"])
    plan_path = run_dir / "plan.json"
    summary_path = run_dir / "summary.json"
    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (plan_path, summary_path)
        if not path.is_file()
    ]
    if missing:
        return jsonify(
            {
                "error": "Initial-settlement reasoning artifacts are missing",
                "missing": missing,
            }
        ), 404

    plan = _read_json(plan_path)
    summary = _read_json(summary_path)
    models = [str(model_id) for model_id in _seq(plan.get("models", []))]
    seeds = [_coerce_int(seed) for seed in _seq(plan.get("seeds", []))]
    trace_rows = [
        row for row in _seq(summary.get("traces", [])) if isinstance(row, dict)
    ]
    by_seed: dict[int, list[dict[str, object]]] = {seed: [] for seed in seeds}
    for row in trace_rows:
        seed = _coerce_int(row.get("seed"))
        if seed in by_seed and row.get("model_id") in models:
            by_seed[seed].append(row)

    seed_status = []
    for seed in seeds:
        captured_models = {
            str(row.get("model_id")) for row in by_seed.get(seed, [])
        }
        captured = len(captured_models)
        if captured == 0:
            status = "not_started"
        elif captured == len(models):
            status = "complete"
        else:
            status = "partial"
        seed_status.append(
            {
                "seed": seed,
                "status": status,
                "captured": captured,
                "expected": len(models),
                "missing_models": [
                    model_id
                    for model_id in models
                    if model_id not in captured_models
                ],
            }
        )

    return jsonify(
        {
            "schema": "catan-initial-settlement-reasoning-ui/v1",
            "run_id": run_id,
            "run_title": run_config["title"],
            "run_description": run_config["description"],
            "available_runs": catalog,
            "excluded_models": run_config["excluded_models"],
            "complete": bool(summary.get("complete")),
            "captured_traces": len(trace_rows),
            "planned_traces": len(models) * len(seeds),
            "recorded_cost_usd": _coerce_float(
                summary.get("recorded_cost_usd")
            ),
            "models": models,
            "seeds": seed_status,
            "traces": trace_rows,
            "conditions": {
                "decision": plan.get("decision"),
                "actor": plan.get("actor"),
                "colors": plan.get("colors", []),
                "context_suite": plan.get("context_suite"),
                "board_surface": plan.get("board_surface"),
                "reasoning_request": plan.get("reasoning_request"),
                "temperature": plan.get("temperature"),
                "max_tokens_omitted": bool(plan.get("max_tokens_omitted")),
                "replay_input": bool(plan.get("replay_input")),
                "human_action_labels": bool(plan.get("human_action_labels")),
                "scheduling": plan.get("scheduling"),
                "prompt_variant": plan.get("prompt_variant"),
            },
        }
    )


def reasoning_trace_payload(
    run_id: str,
    run_config: Mapping[str, object],
) -> Response | tuple[Response, int]:
    """One recorded trace, once the caller has resolved the run."""
    run_dir = cast(Path, run_config["path"])
    model_id = request.args.get("model_id", "")
    raw_seed = request.args.get("seed", "")
    if not model_id or not raw_seed:
        return jsonify({"error": "seed and model_id are required"}), 400
    try:
        seed = int(raw_seed)
    except ValueError:
        return jsonify({"error": "seed must be an integer"}), 400

    plan_path = run_dir / "plan.json"
    summary_path = run_dir / "summary.json"
    if not plan_path.is_file() or not summary_path.is_file():
        return jsonify({"error": "Initial-settlement reasoning artifacts are missing"}), 404
    plan = _read_json(plan_path)
    planned_models = [str(value) for value in _seq(plan.get("models", []))]
    planned_seeds = [_coerce_int(value) for value in _seq(plan.get("seeds", []))]
    if model_id not in planned_models:
        return jsonify({"error": f"Unknown reasoning-trace model: {model_id}"}), 404
    if seed not in planned_seeds:
        return jsonify({"error": f"Unknown reasoning-trace seed: {seed}"}), 404

    summary = _read_json(summary_path)
    row = next(
        (
            item
            for item in _seq(summary.get("traces", []))
            if isinstance(item, dict)
            and _coerce_int(item.get("seed")) == seed
            and item.get("model_id") == model_id
        ),
        None,
    )
    if row is None:
        return jsonify(
            {
                "error": "Reasoning trace has not been captured",
                "seed": seed,
                "model_id": model_id,
            }
        ), 404

    try:
        trace_path = _confined_reasoning_trace_path(
            row.get("json_path"), run_dir
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not trace_path.is_file():
        return jsonify({"error": "Indexed reasoning trace file is missing"}), 404
    trace = _read_json(trace_path)
    if (
        _coerce_int(_map(trace.get("input", {})).get("seed")) != seed
        or _map(trace.get("request", {})).get("requested_model") != model_id
    ):
        return jsonify({"error": "Reasoning trace identity does not match its index"}), 409
    try:
        render_state, board_sha256 = _reasoning_trace_render_state(trace)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409
    trace["run_id"] = run_id
    trace["render_state"] = render_state
    trace["render_state_provenance"] = {
        "seed": seed,
        "board_sha256": board_sha256,
        "verified_against_trace": True,
        "renderer": "existing HexBoard contract",
    }
    return jsonify(trace)
