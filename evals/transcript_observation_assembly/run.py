"""Generation driver and artifact verification."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

from evals.json_types import JsonDict, JsonValue, as_dict, as_dicts, as_list, as_str
from evals.transcript_observation_assembly.artifacts import (
    _append_jsonl,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from evals.transcript_observation_assembly.generation import generate_assembly_job
from evals.transcript_observation_assembly.job import ObservationAssemblyError, ObservationScan
from evals.transcript_observation_assembly.protocol import (
    PARAGRAPH_KINDS,
    RESULT_SCHEMA,
    RUN_SCHEMA,
)
from evals.transcript_reasoning.generation import first_tool_is_board
from evals.transcript_reasoning.run import _within_window


def _plan_identity(plan: Mapping[str, object]) -> Dict[str, object]:
    return {key: value for key, value in plan.items() if key != "created_at"}


def run_generation(
    output_dir: Path,
    scan: ObservationScan,
    plan: JsonDict,
    *,
    workers: int = 4,
    max_tokens: int = 2_000,
    timeout: float = 180.0,
) -> JsonDict:
    """Resume generation and atomically select the latest successful results."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output_dir = Path(output_dir)
    plan_path = output_dir / "plan.json"
    results_path = output_dir / "results.jsonl"
    attempts_path = output_dir / "attempts.jsonl"
    if plan_path.exists():
        existing_plan = _read_json(plan_path)
        if _plan_identity(existing_plan) != _plan_identity(plan):
            raise ObservationAssemblyError(
                "Existing output plan does not match this replay/model/input state"
            )
    else:
        _write_json(plan_path, plan)

    existing_results: Dict[JsonValue, JsonDict] = {
        row["job_id"]: row
        for row in _read_jsonl(results_path)
        if row.get("input_hash")
    }
    jobs = [
        job
        for job in scan["jobs"]
        if not (
            existing_results.get(job.job_id, {}).get("input_hash") == job.input_hash
            and existing_results.get(job.job_id, {}).get("status")
            in {"ready", "empty"}
        )
    ]
    completed = dict(existing_results)
    model_id = str(plan["model_id"])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                generate_assembly_job,
                job,
                model_id,
                max_tokens=max_tokens,
                timeout=timeout,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            result, attempt = future.result()
            completed[result["job_id"]] = result
            _append_jsonl(attempts_path, [attempt])
            ordered = [
                completed[job.job_id]
                for job in scan["jobs"]
                if job.job_id in completed
            ]
            _write_jsonl(results_path, ordered)
            print(
                f"[{len(ordered)}/{len(scan['jobs'])}] "
                f"{result['job_id']} {result['status']}"
            )
    return verify_generation_artifact(output_dir)


def verify_generation_artifact(output_dir: Path) -> JsonDict:
    """Validate causal identity, exact evidence partition, and board inspection."""
    output_dir = Path(output_dir)
    plan = _read_json(output_dir / "plan.json")
    results = _read_jsonl(output_dir / "results.jsonl")
    if plan.get("schema") != RUN_SCHEMA:
        raise ObservationAssemblyError("Unexpected generation plan schema")
    plan_jobs = plan.get("jobs")
    if not isinstance(plan_jobs, list):
        raise ObservationAssemblyError("Generation plan jobs must be a list")
    expected = {
        as_str(job["job_id"], "plan job_id"): job
        for job in as_dicts(plan_jobs, "plan jobs")
    }
    if len(expected) != len(plan_jobs):
        raise ObservationAssemblyError("Generation plan contains duplicate jobs")
    actual = {result.get("job_id"): result for result in results}
    if len(actual) != len(results):
        raise ObservationAssemblyError("Generation results contain duplicate jobs")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected), key=str)
    if missing or extra:
        raise ObservationAssemblyError(
            f"Generation result coverage mismatch: missing={missing}, extra={extra}"
        )

    paragraph_count = 0
    empty_count = 0
    omitted_count = 0
    total_cost = 0.0
    for job_id, job in expected.items():
        result = actual[job_id]
        for field in (
            "schema",
            "input_hash",
            "model_id",
            "generator_version",
            "replay_index",
            "previous_replay_index",
            "anchor_kind",
            "decision_ids",
            "board_state_hash",
            "window_start_s",
            "window_end_s",
        ):
            expected_value = {
                "schema": RESULT_SCHEMA,
                "model_id": plan.get("model_id"),
                "generator_version": plan.get("generator_version"),
            }.get(field, job.get(field))
            if result.get(field) != expected_value:
                raise ObservationAssemblyError(f"{field} mismatch for {job_id}")
        if result.get("status") not in {"ready", "empty"}:
            raise ObservationAssemblyError(f"Unsuccessful result for {job_id}")
        if not first_tool_is_board(result.get("called_tools") or []):
            raise ObservationAssemblyError(f"Board inspection missing for {job_id}")
        paragraphs = result.get("paragraphs")
        omitted = result.get("omitted_evidence_ids")
        if not isinstance(paragraphs, list) or not isinstance(omitted, list):
            raise ObservationAssemblyError(f"Invalid evidence output for {job_id}")
        paragraph_records = as_dicts(paragraphs, f"{job_id} paragraphs")
        cited = [
            item
            for paragraph in paragraph_records
            for item in as_list(paragraph["evidence_ids"], f"{job_id} evidence_ids")
        ]
        if len(cited) != len(set(cited)) or len(omitted) != len(set(omitted)):
            raise ObservationAssemblyError(f"Duplicate evidence output for {job_id}")
        if set(cited).intersection(omitted) or set(cited).union(omitted) != set(
            as_list(job["evidence_ids"], f"{job_id} plan evidence_ids")
        ):
            raise ObservationAssemblyError(f"Evidence partition mismatch for {job_id}")
        if result["status"] == "ready" and not paragraphs:
            raise ObservationAssemblyError(f"Ready result has no paragraphs for {job_id}")
        if result["status"] == "empty":
            if paragraphs:
                raise ObservationAssemblyError(f"Empty result has paragraphs for {job_id}")
            empty_count += 1
        for paragraph in paragraph_records:
            if paragraph.get("kind") not in PARAGRAPH_KINDS:
                raise ObservationAssemblyError(f"Invalid paragraph kind for {job_id}")
            if not _within_window(job, paragraph):
                raise ObservationAssemblyError(
                    f"Paragraph crosses packet bounds for {job_id}"
                )
            if (
                paragraph.get("subject_replay_index") != job["replay_index"]
                or paragraph.get("available_replay_index") != job["replay_index"]
            ):
                raise ObservationAssemblyError(
                    f"Paragraph violates strict causal attachment for {job_id}"
                )
        paragraph_count += len(paragraphs)
        omitted_count += len(omitted)
        cost = as_dict(result.get("usage") or {}, f"{job_id} usage").get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            total_cost += float(cost)

    return {
        "schema": "narrator-observation-assembly-verification-v1",
        "game_id": plan["game_id"],
        "model_id": plan["model_id"],
        "decision_count": plan["decision_count"],
        "anchor_count": plan["anchor_count"],
        "job_count": len(expected),
        "ready_count": len(expected) - empty_count,
        "empty_count": empty_count,
        "paragraph_count": paragraph_count,
        "evidence_count": plan["evidence_count"],
        "omitted_evidence_count": omitted_count,
        "total_cost": round(total_cost, 6),
    }
