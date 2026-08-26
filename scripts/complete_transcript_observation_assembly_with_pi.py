#!/usr/bin/env python3
"""Export observation packets for Pi GPT-5.6 agents and ingest their drafts."""

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from evals.transcript_observation_assembly import (
    ATTEMPT_SCHEMA,
    DEFAULT_DECISION_ARTIFACT_DIR,
    PROMPT_VERSION,
    RESULT_SCHEMA,
    parse_assembly_response,
    scan_observation_jobs,
    verify_generation_artifact,
)

DEFAULT_ARTIFACT_DIR = Path(
    "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI/"
    "narrator_reasoning/gpt_5_6_sol_observation_v3_20260820"
)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_scan_and_results(
    artifact_dir: Path, decision_artifact_dir: Path
) -> tuple[dict, dict, dict, list[dict]]:
    plan = _read_json(artifact_dir / "plan.json")
    scan = scan_observation_jobs(str(plan["game_id"]), decision_artifact_dir)
    jobs = {job.job_id: job for job in scan["jobs"]}
    plan_jobs = {job["job_id"]: job for job in plan["jobs"]}
    if set(jobs) != set(plan_jobs):
        raise ValueError("Current observation scan does not match the persisted plan")
    for job_id, job in jobs.items():
        if job.input_hash != plan_jobs[job_id]["input_hash"]:
            raise ValueError(f"Input hash changed for {job_id}")
    results = _read_jsonl(artifact_dir / "results.jsonl")
    return plan, scan, jobs, results


def export_pending_jobs(
    artifact_dir: Path,
    decision_artifact_dir: Path,
    shard_count: int,
) -> dict:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    plan, _, jobs, results = _load_scan_and_results(
        artifact_dir, decision_artifact_dir
    )
    successful = {
        row["job_id"]
        for row in results
        if row.get("status") in {"ready", "empty"}
        and row.get("input_hash") == jobs[row["job_id"]].input_hash
    }
    pending_ids = [job["job_id"] for job in plan["jobs"] if job["job_id"] not in successful]
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
    shards = []
    for shard_index in range(shard_count):
        job_ids = pending_ids[
            shard_index * chunk_size : (shard_index + 1) * chunk_size
        ]
        if not job_ids:
            continue
        shard_name = f"shard_{shard_index:02d}"
        shard_dir = inputs_dir / shard_name
        shard_dir.mkdir()
        for job_id in job_ids:
            job = jobs[job_id]
            packet = {
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
                "location_tool_results": job.locations,
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
            _write_json(shard_dir / f"{job.replay_index:04d}.json", packet)
        shards.append(
            {
                "shard": shard_name,
                "job_count": len(job_ids),
                "job_ids": job_ids,
                "input_dir": str(shard_dir),
                "draft_path": str(drafts_dir / f"{shard_name}.jsonl"),
            }
        )

    manifest = {
        "schema": "pi-narrator-observation-shards-v1",
        "game_id": plan["game_id"],
        "model_family": "gpt-5.6-sol",
        "pending_job_count": len(pending_ids),
        "shards": shards,
    }
    _write_json(artifact_dir / "pi_shards.json", manifest)
    return manifest


def _manual_result(job, plan: dict, draft: dict) -> tuple[dict, dict]:
    recorded_at = _utc_now()
    raw_payload = {
        "paragraphs": draft["paragraphs"],
        "omitted_evidence_ids": draft["omitted_evidence_ids"],
    }
    raw_response = json.dumps(raw_payload)
    paragraphs, omitted_ids = parse_assembly_response(job, raw_response)
    status = "ready" if paragraphs else "empty"
    result = {
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
        "omitted_evidence_ids": omitted_ids,
        "paragraphs": paragraphs,
        "usage": {},
        "latency_ms": None,
        "error": None,
    }
    attempt = {
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


def ingest_pi_drafts(artifact_dir: Path, decision_artifact_dir: Path) -> dict:
    plan, scan, jobs, results = _load_scan_and_results(
        artifact_dir, decision_artifact_dir
    )
    manifest = _read_json(artifact_dir / "pi_shards.json")
    expected_ids = {
        job_id for shard in manifest["shards"] for job_id in shard["job_ids"]
    }
    drafts = {}
    for shard in manifest["shards"]:
        for row in _read_jsonl(Path(shard["draft_path"])):
            job_id = row.get("job_id")
            if job_id in drafts:
                raise ValueError(f"Duplicate Pi draft for {job_id}")
            if job_id not in expected_ids:
                raise ValueError(f"Unexpected Pi draft job {job_id}")
            if not isinstance(row.get("paragraphs"), list) or not isinstance(
                row.get("omitted_evidence_ids"), list
            ):
                raise ValueError(f"Malformed Pi draft for {job_id}")
            drafts[job_id] = row
    missing = sorted(expected_ids - set(drafts))
    if missing:
        raise ValueError(f"Missing {len(missing)} Pi drafts: {missing}")

    latest = {row["job_id"]: row for row in results}
    attempts = []
    for job_id in expected_ids:
        result, attempt = _manual_result(jobs[job_id], plan, drafts[job_id])
        latest[job_id] = result
        attempts.append(attempt)
    ordered_results = [latest[job.job_id] for job in scan["jobs"]]
    attempts.sort(key=lambda row: jobs[row["job_id"]].replay_index)
    _write_jsonl(artifact_dir / "results.jsonl", ordered_results)
    _append_jsonl(artifact_dir / "attempts.jsonl", attempts)
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
