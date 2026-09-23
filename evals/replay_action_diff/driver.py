"""Top-level entry point that wires scan, query, compare, and report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

from cle.harness.reasoning import native_reasoning_request
from evals import replay_action_diff
from evals.decision_buckets import load_decision_bucket_suite
from evals.json_types import as_dicts, as_list
from evals.replay_action_diff.comparisons import build_comparisons
from evals.replay_action_diff.contracts import COMPARISON_PARSER_VERSION, SCHEMA_VERSION
from evals.replay_action_diff.identity import semantic_manifest_hash, utc_now
from evals.replay_action_diff.query import query_replay
from evals.replay_action_diff.report import render_report
from evals.replay_action_diff.responses import (
    _read_jsonl,
    _write_json,
    _write_jsonl,
    validate_response_compatibility,
)
from evals.replay_action_diff.scan import scan_replay
from evals.replay_action_diff.shapes import RunSummary
from evals.replay_action_diff.summaries import summarize_run


def run_action_diff(
    game_id: str,
    models: Sequence[str],
    output_dir: Path,
    target_player: str = "captured",
    dry_run: bool = False,
    reasoning_effort: str = "xhigh",
    max_tokens: int = 8_192,
    max_requests_per_model: Optional[int] = None,
    max_cost_usd: Optional[float] = None,
    retry_errors: bool = False,
) -> RunSummary:
    """Preflight a full replay, optionally query models, and write artifacts."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if max_cost_usd is not None and max_cost_usd <= 0:
        raise ValueError("max_cost_usd must be positive")
    reasoning = native_reasoning_request(reasoning_effort)
    suite = replay_action_diff.load_context_suite()
    bucket_suite = load_decision_bucket_suite()
    output_dir.mkdir(parents=True, exist_ok=True)
    scan = scan_replay(game_id, target_player)
    semantic_errors = scan["semantic_errors"]
    if semantic_errors:
        _write_json(output_dir / "preflight_errors.json", semantic_errors)
        raise RuntimeError(
            f"Preflight found {len(as_list(semantic_errors, 'semantic_errors'))} "
            "replay semantic errors"
        )

    manifest = as_dicts(scan["records"], "scan records")
    _write_jsonl(output_dir / "decision_manifest.jsonl", manifest)
    manifest_hash = hashlib.sha256(
        (output_dir / "decision_manifest.jsonl").read_bytes()
    ).hexdigest()
    manifest_semantic_hash = semantic_manifest_hash(manifest)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "comparison_parser_version": COMPARISON_PARSER_VERSION,
        "generated_at": utc_now(),
        "game_id": game_id,
        "models": list(models),
        "target_player": target_player,
        "target_player_id": scan["target_player_id"],
        "target_engine_color": scan["target_engine_color"],
        "parsed_action_count": scan["parsed_action_count"],
        "canonicalizations": scan.get("canonicalizations", []),
        "exact_decision_count": sum(
            row.get("classification") == "exact" for row in manifest
        ),
        "manifest_sha256": manifest_hash,
        "manifest_semantic_sha256": manifest_semantic_hash,
        "settings": {
            "context_version": f"{suite.id}@{suite.version}",
            "decision_bucket_suite": f"{bucket_suite.id}@{bucket_suite.version}",
            "stateless_game_plan": True,
            "allow_lookahead": False,
            "execute_model_actions": False,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "max_cost_usd": max_cost_usd,
            "native_reasoning_request": {
                model_id: dict(reasoning)
                for model_id in models
            },
            "headline_includes_forced_exact_choices": True,
            "coarse_actions_excluded": True,
            "same_event_robber_order": "MOVE_ROBBER then STEAL",
        },
    }
    plan_path = output_dir / "plan.json"
    responses_path = output_dir / "responses.jsonl"
    if responses_path.exists() and plan_path.exists():
        existing_plan = json.loads(plan_path.read_text())
        existing_responses = _read_jsonl(responses_path)
        validate_response_compatibility(manifest, existing_responses)
        invariant_fields = (
            "schema_version",
            "game_id",
            "models",
            "target_player",
            "target_player_id",
            "target_engine_color",
            "settings",
        )
        mismatches = [
            field
            for field in invariant_fields
            if existing_plan.get(field) != plan.get(field)
        ]
        existing_semantic_hash = existing_plan.get("manifest_semantic_sha256")
        if (
            existing_semantic_hash is not None
            and existing_semantic_hash != manifest_semantic_hash
        ):
            mismatches.append("manifest_semantic_sha256")
        if mismatches:
            raise RuntimeError(
                "Cannot resume response artifact after run inputs changed: "
                + ", ".join(mismatches)
            )
    _write_json(plan_path, plan)

    new_requests = {}
    if not dry_run:
        new_requests = query_replay(
            game_id=game_id,
            target_player=target_player,
            manifest=manifest,
            models=models,
            responses_path=responses_path,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            max_requests_per_model=max_requests_per_model,
            max_cost_usd=max_cost_usd,
            retry_errors=retry_errors,
        )

    response_rows = _read_jsonl(responses_path)
    comparisons = build_comparisons(manifest, response_rows, models)
    summary = summarize_run(scan, response_rows, models)
    summary["new_requests_this_invocation"] = new_requests
    summary["complete"] = all(
        summary["models"][model_id]["responses_present"]
        == summary["exact_decisions"]
        and summary["models"][model_id]["api_errors"] == 0
        for model_id in models
    )
    _write_jsonl(output_dir / "comparisons.jsonl", comparisons)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        render_report(summary, comparisons, models)
    )
    return summary
