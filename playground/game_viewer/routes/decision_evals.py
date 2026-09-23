"""Read-only API for bucketed agent-decision spot checks."""

from __future__ import annotations

from functools import lru_cache

from flask import Blueprint, Response, jsonify, request

from evals.decision_spot_checks import (
    CURATED_DECISION_RUNS,
    DecisionEvalArtifactError,
    compact_decision,
    decision_bucket_catalog,
    decision_detail,
    filter_decisions,
    list_decision_eval_runs,
    load_decision_eval_run,
)
from evals.json_types import JsonDict

decision_evals_bp = Blueprint("decision_evals", __name__)


@decision_evals_bp.route("/api/decision-evals/catalog", methods=["GET"])
def get_decision_eval_catalog() -> Response | tuple[Response, int]:
    """Return the authored bucket, rubric, label, and sampling specification."""
    return jsonify(decision_bucket_catalog())


@decision_evals_bp.route("/api/decision-evals/runs", methods=["GET"])
def get_decision_eval_runs() -> Response | tuple[Response, int]:
    """List curated decision-eval artifacts available to the frontend."""
    return jsonify({"runs": list_decision_eval_runs()})


@decision_evals_bp.route("/api/decision-evals/decisions", methods=["GET"])
def get_decision_eval_decisions() -> Response | tuple[Response, int]:
    """Return a filtered compact decision list plus unfiltered bucket counts."""
    try:
        run = _requested_run()
        decisions = filter_decisions(
            run,
            bucket_id=request.args.get("bucket"),
            stage=request.args.get("stage"),
            agreement=request.args.get("agreement"),
            classification=request.args.get("classification"),
        )
        offset = _bounded_int_arg("offset", 0, minimum=0, maximum=1_000_000)
        limit = _bounded_int_arg("limit", 200, minimum=1, maximum=500)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except DecisionEvalArtifactError as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(
        {
            "run": _compact_run(run),
            "bucket_counts": run["bucket_counts"],
            "stage_counts": run["stage_counts"],
            "total": len(decisions),
            "offset": offset,
            "limit": limit,
            "decisions": [
                compact_decision(decision)
                for decision in decisions[offset : offset + limit]
            ],
        }
    )


@decision_evals_bp.route("/api/decision-evals/decision", methods=["GET"])
def get_decision_eval_decision() -> Response | tuple[Response, int]:
    """Return one review-ready decision with model output and prompt context."""
    decision_id = (request.args.get("decision_id") or "").strip()
    if not decision_id:
        return jsonify({"error": "decision_id is required"}), 400
    try:
        run = _requested_run()
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except DecisionEvalArtifactError as exc:
        return jsonify({"error": str(exc)}), 500

    detail = decision_detail(run, decision_id)
    if detail is None:
        return jsonify({"error": f"Unknown decision_id: {decision_id}"}), 404
    return jsonify({"run": _compact_run(run), "decision": detail})


def _requested_run() -> JsonDict:
    run_id = (request.args.get("run_id") or "").strip()
    if not run_id:
        run_id = next(iter(CURATED_DECISION_RUNS), "")
    if run_id not in CURATED_DECISION_RUNS:
        raise KeyError(f"Unknown decision eval run: {run_id}")
    return _load_run(run_id)


@lru_cache(maxsize=8)
def _load_run(run_id: str) -> JsonDict:
    run: JsonDict = load_decision_eval_run(CURATED_DECISION_RUNS[run_id])
    return run


def _compact_run(run: JsonDict) -> JsonDict:
    return {
        key: run[key]
        for key in (
            "schema",
            "id",
            "title",
            "description",
            "game_id",
            "model_id",
            "target_player_id",
            "target_engine_color",
            "bucket_suite",
            "decision_count",
            "bucketed_decision_count",
            "response_count",
            "disagreement_count",
        )
    }


def _bounded_int_arg(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = request.args.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value
