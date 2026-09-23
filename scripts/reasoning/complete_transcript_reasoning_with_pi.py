#!/usr/bin/env python3
"""Export failed narrator jobs for Pi agents and ingest their validated drafts."""

import argparse
import json
import math
import shutil
from pathlib import Path

from evals.json_types import JsonDict, JsonValue
from evals.transcript_reasoning import (
    ATTEMPT_SCHEMA,
    PROMPT_VERSION,
    RESULT_SCHEMA,
    ReasoningJob,
    parse_reasoning_response,
    scan_reasoning_jobs,
    verify_generation_artifact,
)
from scripts.reasoning.jsonio import (
    append_jsonl,
    object_list,
    read_json,
    read_jsonl,
    string_list,
    utc_now,
    write_json,
    write_jsonl,
)

DEFAULT_ARTIFACT_DIR = Path(
    "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI/"
    "narrator_reasoning/gpt_5_6_sol_20260820"
)

JobsById = dict[str, ReasoningJob]
LoadedArtifact = tuple[JsonDict, list[ReasoningJob], JobsById, list[JsonDict]]


def _load_scan_and_results(artifact_dir: Path) -> LoadedArtifact:
    plan = read_json(artifact_dir / "plan.json")
    scan = scan_reasoning_jobs(str(plan["game_id"]))
    ordered_jobs: list[ReasoningJob] = list(scan["jobs"])
    jobs = {job.job_id: job for job in ordered_jobs}
    plan_jobs = {
        str(job["job_id"]): job for job in object_list(plan, "jobs", "plan.json")
    }
    if set(jobs) != set(plan_jobs):
        raise ValueError("Current replay scan does not match the persisted plan")
    for job_id, job in jobs.items():
        if job.input_hash != plan_jobs[job_id]["input_hash"]:
            raise ValueError(f"Input hash changed for {job_id}")
    results = read_jsonl(artifact_dir / "results.jsonl")
    return plan, ordered_jobs, jobs, results


def _input_packet(job: ReasoningJob) -> JsonDict:
    return {
        "schema": "pi-narrator-reasoning-input-v1",
        "generator_version": PROMPT_VERSION,
        "job_id": job.job_id,
        "game_id": job.game_id,
        "replay_index": job.replay_index,
        "availability_window": {
            "start_s": job.window_start_s,
            "end_s": job.window_end_s,
        },
        "utterances": list(job.utterances),
        "public_board_tool_result": job.board_snapshot,
        "location_tool_results": dict(job.locations),
        "instructions": [
            "Reconstruct only substantive first-person narrator reasoning.",
            "Remove filler, repetition, and ASR false starts without adding strategy.",
            "Use the public board and location results only to clarify supported references.",
            "Preserve uncertainty and mistakes; do not infer the next action or hidden hands.",
            "Return zero paragraphs for filler-only commentary.",
        ],
        "output_shape": {
            "job_id": job.job_id,
            "paragraphs": [
                {
                    "text": "coherent prose",
                    "evidence_ids": ["u0"],
                    "uncertainties": [],
                }
            ],
        },
    }


def export_failed_jobs(artifact_dir: Path, shard_count: int) -> JsonDict:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    plan, _, jobs, results = _load_scan_and_results(artifact_dir)
    failed_ids = [str(row["job_id"]) for row in results if row.get("status") == "error"]
    if not failed_ids:
        raise ValueError("The artifact has no failed jobs to export")

    inputs_dir = artifact_dir / "pi_inputs"
    drafts_dir = artifact_dir / "pi_drafts"
    if inputs_dir.exists():
        shutil.rmtree(inputs_dir)
    if drafts_dir.exists():
        shutil.rmtree(drafts_dir)
    inputs_dir.mkdir(parents=True)
    drafts_dir.mkdir(parents=True)

    chunk_size = math.ceil(len(failed_ids) / shard_count)
    shards: list[JsonDict] = []
    for shard_index in range(shard_count):
        job_ids = failed_ids[shard_index * chunk_size : (shard_index + 1) * chunk_size]
        if not job_ids:
            continue
        shard_name = f"shard_{shard_index:02d}"
        shard_dir = inputs_dir / shard_name
        shard_dir.mkdir()
        for job_id in job_ids:
            job = jobs[job_id]
            write_json(shard_dir / f"{job.replay_index:04d}.json", _input_packet(job))
        shard: JsonDict = {
            "shard": shard_name,
            "job_count": len(job_ids),
            "job_ids": list(job_ids),
            "input_dir": str(shard_dir),
            "draft_path": str(drafts_dir / f"{shard_name}.jsonl"),
        }
        shards.append(shard)

    manifest: JsonDict = {
        "schema": "pi-narrator-reasoning-shards-v1",
        "game_id": plan["game_id"],
        "model_family": "gpt-5.6-sol",
        "failed_job_count": len(failed_ids),
        "shards": list(shards),
    }
    write_json(artifact_dir / "pi_shards.json", manifest)
    return manifest


def _manual_result(
    job: ReasoningJob, plan: JsonDict, paragraphs: JsonValue
) -> tuple[JsonDict, JsonDict]:
    recorded_at = utc_now()
    raw_response = json.dumps({"paragraphs": paragraphs})
    validated = parse_reasoning_response(job, raw_response)
    status = "ready" if validated else "empty"
    result: JsonDict = {
        "schema": RESULT_SCHEMA,
        "job_id": job.job_id,
        "game_id": job.game_id,
        "replay_index": job.replay_index,
        "input_hash": job.input_hash,
        "model_id": plan["model_id"],
        "generator_version": PROMPT_VERSION,
        "generation_backend": "pi-agent-gpt-5.6-sol",
        "status": status,
        "recorded_at": recorded_at,
        "board_state_hash": job.board_snapshot["board_state_hash"],
        "window_start_s": job.window_start_s,
        "window_end_s": job.window_end_s,
        "called_tools": ["inspect_board"],
        "paragraphs": list(validated),
        "usage": {},
        "latency_ms": None,
        "error": None,
    }
    attempt: JsonDict = {
        "schema": ATTEMPT_SCHEMA,
        "job_id": job.job_id,
        "input_hash": job.input_hash,
        "model_id": plan["model_id"],
        "generation_backend": "pi-agent-gpt-5.6-sol",
        "recorded_at": recorded_at,
        "status": status,
        "messages": [],
        "tool_messages": [{"role": "tool", "name": "inspect_board"}],
        "called_tools": ["inspect_board"],
        "raw_response": raw_response,
        "usage": {},
        "latency_ms": None,
        "error": None,
    }
    return result, attempt


def ingest_pi_drafts(artifact_dir: Path) -> JsonDict:
    plan, ordered_jobs, jobs, results = _load_scan_and_results(artifact_dir)
    manifest = read_json(artifact_dir / "pi_shards.json")
    shard_records = object_list(manifest, "shards", "pi_shards.json")
    expected_ids = {
        job_id
        for shard in shard_records
        for job_id in string_list(shard, "job_ids", "pi_shards.json")
    }
    drafts: dict[str, JsonDict] = {}
    for shard in shard_records:
        draft_path = Path(str(shard["draft_path"]))
        for row in read_jsonl(draft_path):
            raw_job_id = row.get("job_id")
            if raw_job_id in drafts:
                raise ValueError(f"Duplicate Pi draft for {raw_job_id}")
            if raw_job_id not in expected_ids:
                raise ValueError(f"Unexpected Pi draft job {raw_job_id}")
            if not isinstance(row.get("paragraphs"), list):
                raise ValueError(f"Pi draft paragraphs must be a list for {raw_job_id}")
            drafts[str(raw_job_id)] = row
    missing = sorted(expected_ids - set(drafts))
    if missing:
        raise ValueError(f"Missing {len(missing)} Pi drafts: {missing}")

    latest = {str(row["job_id"]): row for row in results}
    attempts: list[JsonDict] = []
    for job_id in sorted(expected_ids, key=lambda value: jobs[value].replay_index):
        result, attempt = _manual_result(
            jobs[job_id], plan, drafts[job_id]["paragraphs"]
        )
        latest[job_id] = result
        attempts.append(attempt)
    ordered_results = [latest[job.job_id] for job in ordered_jobs]
    write_jsonl(artifact_dir / "results.jsonl", ordered_results)
    append_jsonl(artifact_dir / "attempts.jsonl", attempts)
    return verify_generation_artifact(artifact_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("export", "ingest"))
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--shards", type=int, default=8)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "export":
        result = export_failed_jobs(args.artifact_dir, args.shards)
    else:
        result = ingest_pi_drafts(args.artifact_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
