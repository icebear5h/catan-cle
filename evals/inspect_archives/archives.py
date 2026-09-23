"""Bundle builders for the strict-vision and policy archives."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset

from evals.decision_spot_checks import (
    CURATED_DECISION_RUNS,
    DecisionEvalRunConfig,
    load_decision_eval_run,
)
from evals.decision_spot_checks.shapes import decision_model
from evals.inspect_archives.config import (
    ARCHIVE_IMPORT_SCHEMA,
    DEFAULT_POLICY_RUN_ID,
    MODEL_SELECTION_REPORT,
    InspectArchiveBundle,
    JsonDict,
)
from evals.inspect_archives.model_api import archived_model, replay_archived_output
from evals.inspect_archives.policy import (
    _latest_policy_responses,
    _policy_available_actions,
    _policy_score_payload,
    is_valid_policy_selection,
)
from evals.inspect_archives.samples import _policy_sample, _strict_sample
from evals.inspect_archives.scorers import (
    nontrivial_recorded_human_action_match,
    policy_parse_clean,
    policy_selection_valid,
    recorded_human_action_match,
    strict_exact,
    strict_json_valid,
    strict_protocol_exact,
)
from evals.inspect_archives.support import (
    _read_json,
    _read_jsonl,
    _relative_path,
    _resolved_path,
    _unique_rows,
)
from evals.inspect_archives.validation import (
    _preflight_strict_dataset,
    _require_dataset_file,
    _validate_strict_row,
    _validate_strict_run,
)
from evals.json_types import as_dict, as_dicts, as_int, as_list, as_str
from scripts.board_bench.run.eval_catan_strict_vision_probe import build_jobs, validate_dataset


def build_strict_vision_archive(
    run_dir: str | Path,
    *,
    limit: int | None = None,
) -> InspectArchiveBundle:
    """Validate one strict-vision run and construct its provider-free task."""

    source_dir = _resolved_path(run_dir)
    plan = _read_json(source_dir / "plan.json")
    summary = _read_json(source_dir / "summary.json")
    response_rows = _read_jsonl(source_dir / "responses.jsonl")
    _validate_strict_run(source_dir, plan, summary, response_rows)

    dataset_dir = _resolved_path(as_str(plan["dataset_dir"], "plan dataset_dir"))
    _preflight_strict_dataset(dataset_dir)
    _, manifest, questions = validate_dataset(dataset_dir)
    jobs = build_jobs(dataset_dir, manifest=manifest, questions=questions)
    jobs_by_id = {str(job["qa"]["id"]): job for job in jobs}
    response_by_id = _unique_rows(response_rows, "question_id", source_dir)
    selected_ids = [str(value) for value in as_list(plan["question_ids"], "question_ids")]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected_ids = selected_ids[:limit]

    samples = []
    for question_id in selected_ids:
        row = response_by_id[question_id]
        job = jobs_by_id[question_id]
        _require_dataset_file(Path(job["image_path"]), dataset_dir)
        _require_dataset_file(Path(job["contract_path"]), dataset_dir)
        _validate_strict_row(source_dir, plan, row, job)
        samples.append(_strict_sample(source_dir, plan, row, job))

    model_id = str(plan["model"])
    overall = as_dict(summary["overall"], "summary overall")
    expected: JsonDict = {
        "exact": overall["exact"],
        "json_valid": overall["json_valid"],
        "protocol_exact": overall["protocol_exact"],
        "requests": overall["requests"],
    }
    task = Task(
        name="catan_strict_raw_vision_60_archive",
        display_name=f"Catan strict raw vision 60 · {model_id}",
        version="strict_typed_json/v2",
        dataset=MemoryDataset(samples=samples, name="catan_strict_raw_vision_60"),
        solver=replay_archived_output(),
        scorer=[strict_exact(), strict_json_valid(), strict_protocol_exact()],
        metadata={
            "archive_import_schema": ARCHIVE_IMPORT_SCHEMA,
            "archive_source": _relative_path(source_dir),
            "archive_model_id": model_id,
            "archive_provider": as_dict(plan["request_settings"], "request_settings")[
                "provider"
            ],
            "benchmark_contract": plan["suite"],
            "input_mode": "raw_image",
            "manifest_sha256": plan["manifest_sha256"],
            "scorer": plan["scorer"],
            "request_settings": plan["request_settings"],
            "source_complete": bool(summary.get("complete")),
            "source_metrics": expected,
            "model_selection_report": MODEL_SELECTION_REPORT,
            "imported_records": len(samples),
            "source_records": len(response_rows),
        },
    )
    return InspectArchiveBundle(
        archive_id=source_dir.name,
        task=task,
        model=archived_model(model_id),
        model_id=model_id,
        source_dir=source_dir,
        input_mode="raw_image",
        source_records=len(response_rows),
        imported_records=len(samples),
        expected_metrics=expected,
    )


def build_policy_archive(
    config: DecisionEvalRunConfig | None = None,
    *,
    limit: int | None = None,
) -> InspectArchiveBundle:
    """Construct an Inspect task from the original replay-policy provider run.

    Later setup-selection and rationale repair overlays are deliberately outside
    this archival contract so they can never be merged silently into the
    published benchmark result.
    """

    selected_config = config or CURATED_DECISION_RUNS[DEFAULT_POLICY_RUN_ID]
    effective_config = replace(
        selected_config,
        response_override_paths=(),
        rationale_repair_paths=(),
    )
    run = load_decision_eval_run(effective_config)
    run_decisions = as_dicts(run["decisions"], "run decisions")
    response_decisions = [
        row for row in run_decisions if decision_model(row)["response_present"]
    ]
    raw_responses = _latest_policy_responses(
        effective_config.artifact_dir / "responses.jsonl",
        str(run["model_id"]),
    )
    response_ids = {str(row["decision_id"]) for row in response_decisions}
    if set(raw_responses) != response_ids:
        raise ValueError("original provider responses do not match joined policy decisions")
    decisions = response_decisions
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        decisions = decisions[:limit]
    samples = [
        _policy_sample(
            effective_config,
            decision,
            raw_responses[str(decision["decision_id"])],
        )
        for decision in decisions
    ]
    model_id = str(run["model_id"])
    exact = [row for row in run_decisions if row["classification"] == "exact"]
    responded_exact = [row for row in exact if decision_model(row)["response_present"]]
    nontrivial = [row for row in responded_exact if not row["forced"]]
    expected: JsonDict = {
        "decision_count": run["decision_count"],
        "response_count": run["response_count"],
        "exact_responses": len(responded_exact),
        "exact_human_matches": sum(
            bool(decision_model(row)["agreement"]) for row in responded_exact
        ),
        "nontrivial_responses": len(nontrivial),
        "nontrivial_human_matches": sum(
            bool(decision_model(row)["agreement"]) for row in nontrivial
        ),
        "parse_warnings": sum(
            bool(decision_model(row)["parse_error"]) for row in response_decisions
        ),
        "valid_selections": sum(
            is_valid_policy_selection(
                _policy_score_payload(
                    row,
                    raw_responses[str(row["decision_id"])],
                ),
                _policy_available_actions(
                    row,
                    raw_responses[str(row["decision_id"])],
                ),
            )
            for row in response_decisions
        ),
    }
    task = Task(
        name="catan_replay_policy_archive",
        display_name=f"Catan replay policy · {model_id} · {run['game_id']}",
        version=str(as_dict(run["bucket_suite"], "bucket_suite")["version"]),
        dataset=MemoryDataset(samples=samples, name=f"catan_replay_{run['game_id']}"),
        solver=replay_archived_output(),
        scorer=[
            policy_selection_valid(),
            policy_parse_clean(),
            recorded_human_action_match(),
            nontrivial_recorded_human_action_match(),
        ],
        metadata={
            "archive_import_schema": ARCHIVE_IMPORT_SCHEMA,
            "archive_source": _relative_path(selected_config.artifact_dir),
            "archive_variant": "original_provider_run",
            "archive_model_id": model_id,
            "benchmark_contract": "replay-action-diff-v1",
            "input_mode": "perspective_safe_symbolic_text",
            "game_id": run["game_id"],
            "target_engine_color": run["target_engine_color"],
            "bucket_suite": run["bucket_suite"],
            "source_metrics": expected,
            "model_selection_report": MODEL_SELECTION_REPORT,
            "imported_records": len(samples),
            "source_records": run["decision_count"],
            "interpretation": (
                "Recorded-human action agreement is descriptive and is not a policy-quality oracle."
            ),
        },
    )
    return InspectArchiveBundle(
        archive_id=as_str(run["id"], "run id"),
        task=task,
        model=archived_model(model_id),
        model_id=model_id,
        source_dir=selected_config.artifact_dir,
        input_mode="perspective_safe_symbolic_text",
        source_records=as_int(run["decision_count"], "decision_count"),
        imported_records=len(samples),
        expected_metrics=expected,
    )

