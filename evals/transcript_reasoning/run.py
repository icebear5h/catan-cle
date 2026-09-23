"""Generation driver and artifact verification."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

from evals.json_types import JsonDict, JsonValue, as_dict, as_dicts, as_number, as_str
from evals.transcript_reasoning.artifacts import (
    _append_jsonl,
    _plan_identity,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from evals.transcript_reasoning.generation import first_tool_is_board, generate_reasoning_job
from evals.transcript_reasoning.job import NarratorReasoningError, ReasoningScan
from evals.transcript_reasoning.protocol import RESULT_SCHEMA, RUN_SCHEMA


def run_generation(
    output_dir: Path,
    scan: ReasoningScan,
    plan: JsonDict,
    *,
    workers: int = 4,
    max_tokens: int = 1_200,
    timeout: float = 180.0,
) -> JsonDict:
    """Resume a generation run and atomically materialize latest selected results."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output_dir = Path(output_dir)
    plan_path = output_dir / "plan.json"
    results_path = output_dir / "results.jsonl"
    attempts_path = output_dir / "attempts.jsonl"
    if plan_path.exists():
        existing_plan = _read_json(plan_path)
        if _plan_identity(existing_plan) != _plan_identity(plan):
            raise NarratorReasoningError(
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
            and existing_results.get(job.job_id, {}).get("status") in {"ready", "empty"}
        )
    ]
    completed = dict(existing_results)
    model_id = str(plan["model_id"])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                generate_reasoning_job,
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
            ordered_results = [
                completed[job.job_id]
                for job in scan["jobs"]
                if job.job_id in completed
            ]
            _write_jsonl(results_path, ordered_results)
            print(
                f"[{len(ordered_results)}/{len(scan['jobs'])}] "
                f"{result['job_id']} {result['status']}"
            )

    return verify_generation_artifact(output_dir)


def _within_window(job: JsonDict, paragraph: JsonDict) -> bool:
    bounds = (
        job["source_start_s"],
        paragraph["start_s"],
        paragraph["end_s"],
        job["source_end_s"],
        job["window_end_s"],
    )
    values = [as_number(value, "paragraph window bound") for value in bounds]
    return all(left <= right for left, right in zip(values, values[1:]))


def verify_generation_artifact(output_dir: Path) -> JsonDict:
    """Validate completeness, input identity, evidence bounds, and board-tool use."""
    output_dir = Path(output_dir)
    plan = _read_json(output_dir / "plan.json")
    results = _read_jsonl(output_dir / "results.jsonl")
    if plan.get("schema") != RUN_SCHEMA:
        raise NarratorReasoningError("Unexpected generation plan schema")
    plan_jobs = plan.get("jobs")
    if not isinstance(plan_jobs, list):
        raise NarratorReasoningError("Generation plan jobs must be a list")
    expected = {
        as_str(job["job_id"], "plan job_id"): job
        for job in as_dicts(plan_jobs, "plan jobs")
    }
    if len(expected) != len(plan_jobs):
        raise NarratorReasoningError("Generation plan contains duplicate job IDs")
    actual = {result.get("job_id"): result for result in results}
    if len(actual) != len(results):
        raise NarratorReasoningError("Generation results contain duplicate job IDs")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected), key=str)
    if missing or extra:
        raise NarratorReasoningError(
            f"Generation result coverage mismatch: missing={missing}, extra={extra}"
        )

    paragraph_count = 0
    empty_count = 0
    total_cost = 0.0
    for job_id, job in expected.items():
        result = actual[job_id]
        if result.get("schema") != RESULT_SCHEMA:
            raise NarratorReasoningError(f"Unexpected result schema for {job_id}")
        if result.get("input_hash") != job.get("input_hash"):
            raise NarratorReasoningError(f"Input hash mismatch for {job_id}")
        if result.get("model_id") != plan.get("model_id"):
            raise NarratorReasoningError(f"Model mismatch for {job_id}")
        if result.get("status") not in {"ready", "empty"}:
            raise NarratorReasoningError(f"Unsuccessful result for {job_id}")
        if not first_tool_is_board(result.get("called_tools") or []):
            raise NarratorReasoningError(f"Board inspection missing for {job_id}")
        paragraphs = as_dicts(result.get("paragraphs") or [], f"{job_id} paragraphs")
        if result["status"] == "ready" and not paragraphs:
            raise NarratorReasoningError(f"Ready result has no paragraphs for {job_id}")
        if result["status"] == "empty":
            if paragraphs:
                raise NarratorReasoningError(f"Empty result has paragraphs for {job_id}")
            empty_count += 1
        for paragraph in paragraphs:
            if not _within_window(job, paragraph):
                raise NarratorReasoningError(
                    f"Paragraph crosses transcript window for {job_id}"
                )
        paragraph_count += len(paragraphs)
        cost = as_dict(result.get("usage") or {}, f"{job_id} usage").get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            total_cost += float(cost)

    return {
        "schema": "narrator-reasoning-verification-v1",
        "game_id": plan["game_id"],
        "model_id": plan["model_id"],
        "job_count": len(expected),
        "ready_count": len(expected) - empty_count,
        "empty_count": empty_count,
        "paragraph_count": paragraph_count,
        "total_cost": round(total_cost, 6),
    }
