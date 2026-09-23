#!/usr/bin/env python3
"""Export observation packets for Pi GPT-5.6 agents and ingest their drafts."""

import argparse
import json
import math
import shutil
from pathlib import Path

from evals.json_types import JsonDict
from evals.transcript_observation_assembly import (
    ATTEMPT_SCHEMA,
    DEFAULT_DECISION_ARTIFACT_DIR,
    PROMPT_VERSION,
    RESULT_SCHEMA,
    ObservationAssemblyJob,
    parse_assembly_response,
    scan_observation_jobs,
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
    "narrator_reasoning/gpt_5_6_sol_observation_v3_20260820"
)

JobsById = dict[str, ObservationAssemblyJob]
LoadedArtifact = tuple[JsonDict, list[ObservationAssemblyJob], JobsById, list[JsonDict]]


def _load_scan_and_results(
    artifact_dir: Path, decision_artifact_dir: Path
) -> LoadedArtifact:
    plan = read_json(artifact_dir / "plan.json")
    scan = scan_observation_jobs(str(plan["game_id"]), decision_artifact_dir)
    ordered_jobs: list[ObservationAssemblyJob] = list(scan["jobs"])
    jobs = {job.job_id: job for job in ordered_jobs}
    plan_jobs = {
        str(job["job_id"]): job for job in object_list(plan, "jobs", "plan.json")
    }
    if set(jobs) != set(plan_jobs):
        raise ValueError("Current observation scan does not match the persisted plan")
    for job_id, job in jobs.items():
        if job.input_hash != plan_jobs[job_id]["input_hash"]:
            raise ValueError(f"Input hash changed for {job_id}")
    results = read_jsonl(artifact_dir / "results.jsonl")
    return plan, ordered_jobs, jobs, results


def export_pending_jobs(
    artifact_dir: Path,
    decision_artifact_dir: Path,
    shard_count: int,
) -> JsonDict:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    plan, _, jobs, results = _load_scan_and_results(artifact_dir, decision_artifact_dir)
    successful = {
        str(row["job_id"])
        for row in results
        if row.get("status") in {"ready", "empty"}
        and row.get("input_hash") == jobs[str(row["job_id"])].input_hash
    }
    pending_ids = [
        str(job["job_id"])
        for job in object_list(plan, "jobs", "plan.json")
        if str(job["job_id"]) not in successful
    ]
    if not pending_ids:
        raise ValueError("The artifact has no pending jobs to export")

    inputs_dir = artifact_dir / "pi_inputs"
    drafts_dir = artifact_dir / "pi_drafts"
    if inputs_dir.exists():
        shutil.rmtree(inputs_dir)
    if drafts_dir.exists():
        shutil.rmtree(drafts_dir)
    inputs_dir.mkdir(parents=True)
    drafts_dir.mkdir(parents=True)

    chunk_size = math.ceil(len(pending_ids) / shard_count)
    shards: list[JsonDict] = []
    for shard_index in range(shard_count):
        job_ids = pending_ids[shard_index * chunk_size : (shard_index + 1) * chunk_size]
        if not job_ids:
            continue
        shard_name = f"shard_{shard_index:02d}"
        shard_dir = inputs_dir / shard_name
        shard_dir.mkdir()
        for job_id in job_ids:
            job = jobs[job_id]
            write_json(shard_dir / f"{job.replay_index:04d}.json", _input_packet(job))
        shards.append(
            {
                "shard": shard_name,
                "job_count": len(job_ids),
                "job_ids": list(job_ids),
                "input_dir": str(shard_dir),
                "draft_path": str(drafts_dir / f"{shard_name}.jsonl"),
            }
        )

    manifest: JsonDict = {
        "schema": "pi-narrator-observation-shards-v1",
        "game_id": plan["game_id"],
        "model_family": "gpt-5.6-sol",
        "pending_job_count": len(pending_ids),
        "shards": list(shards),
    }
    write_json(artifact_dir / "pi_shards.json", manifest)
    return manifest


def _input_packet(job: ObservationAssemblyJob) -> JsonDict:
    return {
        "schema": "pi-narrator-observation-input-v1",
        "generator_version": PROMPT_VERSION,
        "job_id": job.job_id,
        "game_id": job.game_id,
        "replay_index": job.replay_index,
        "anchor_kind": job.anchor_kind,
        "decision_ids": list(job.decision_ids),
        "availability_window": {
            "start_s": job.window_start_s,
            "end_s": job.window_end_s,
        },
        "utterances": list(job.utterances),
        "already_visible_public_events": list(job.visible_observations),
        "public_board_tool_result": job.board_snapshot,
        "location_tool_results": dict(job.locations),
        "allowed_kinds": [
            "decision_reasoning",
            "board_observation",
            "opponent_assessment",
            "reaction",
            "reflection",
        ],
        "instructions": [
            "Assemble related commentary across replay-row boundaries.",
            "Use only captions, visible public events, and the public board.",
            "Never infer the action at this cursor, future events, or hidden hands.",
            "Preserve uncertainty, alternatives, mistakes, and changes of mind.",
            "Use decision_reasoning only at a decision anchor and only for explicit deliberation.",
            "Cite every evidence ID exactly once or list it as omitted filler/repetition/chatter.",
            "Return at most eight concise first-person prose paragraphs.",
        ],
        "output_shape": {
            "job_id": job.job_id,
            "paragraphs": [
                {
                    "kind": "board_observation",
                    "text": "coherent prose",
                    "evidence_ids": ["e0001"],
                    "uncertainties": [],
                }
            ],
            "omitted_evidence_ids": ["e0002"],
        },
    }


def _manual_result(
    job: ObservationAssemblyJob, plan: JsonDict, draft: JsonDict
) -> tuple[JsonDict, JsonDict]:
    recorded_at = utc_now()
    raw_payload: JsonDict = {
        "paragraphs": draft["paragraphs"],
        "omitted_evidence_ids": draft["omitted_evidence_ids"],
    }
    raw_response = json.dumps(raw_payload)
    paragraphs, omitted_ids = parse_assembly_response(job, raw_response)
    status = "ready" if paragraphs else "empty"
    result: JsonDict = {
        "schema": RESULT_SCHEMA,
        "job_id": job.job_id,
        "game_id": job.game_id,
        "replay_index": job.replay_index,
        "previous_replay_index": job.previous_replay_index,
        "anchor_kind": job.anchor_kind,
        "decision_ids": list(job.decision_ids),
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
        "evidence_count": len(job.utterances),
        "omitted_evidence_ids": list(omitted_ids),
        "paragraphs": list(paragraphs),
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


def ingest_pi_drafts(artifact_dir: Path, decision_artifact_dir: Path) -> JsonDict:
    plan, ordered_jobs, jobs, results = _load_scan_and_results(
        artifact_dir, decision_artifact_dir
    )
    manifest = read_json(artifact_dir / "pi_shards.json")
    shard_records = object_list(manifest, "shards", "pi_shards.json")
    expected_ids = {
        job_id
        for shard in shard_records
        for job_id in string_list(shard, "job_ids", "pi_shards.json")
    }
    drafts: dict[str, JsonDict] = {}
    for shard in shard_records:
        for row in read_jsonl(Path(str(shard["draft_path"]))):
            raw_job_id = row.get("job_id")
            if raw_job_id in drafts:
                raise ValueError(f"Duplicate Pi draft for {raw_job_id}")
            if raw_job_id not in expected_ids:
                raise ValueError(f"Unexpected Pi draft job {raw_job_id}")
            if not isinstance(row.get("paragraphs"), list) or not isinstance(
                row.get("omitted_evidence_ids"), list
            ):
                raise ValueError(f"Malformed Pi draft for {raw_job_id}")
            drafts[str(raw_job_id)] = row
    missing = sorted(expected_ids - set(drafts))
    if missing:
        raise ValueError(f"Missing {len(missing)} Pi drafts: {missing}")

    latest = {str(row["job_id"]): row for row in results}
    attempts: list[JsonDict] = []
    for job_id in expected_ids:
        result, attempt = _manual_result(jobs[job_id], plan, drafts[job_id])
        latest[job_id] = result
        attempts.append(attempt)
    ordered_results = [latest[job.job_id] for job in ordered_jobs]
    attempts.sort(key=lambda row: jobs[str(row["job_id"])].replay_index)
    write_jsonl(artifact_dir / "results.jsonl", ordered_results)
    append_jsonl(artifact_dir / "attempts.jsonl", attempts)
    return verify_generation_artifact(artifact_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("export", "ingest"))
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--decision-artifact-dir",
        type=Path,
        default=DEFAULT_DECISION_ARTIFACT_DIR,
    )
    parser.add_argument("--shards", type=int, default=16)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "export":
        result = export_pending_jobs(
            args.artifact_dir, args.decision_artifact_dir, args.shards
        )
    else:
        result = ingest_pi_drafts(args.artifact_dir, args.decision_artifact_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
