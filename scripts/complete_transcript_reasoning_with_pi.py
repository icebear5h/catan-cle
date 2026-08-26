#!/usr/bin/env python3
"""Export failed narrator jobs for Pi agents and ingest their validated drafts."""

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from evals.transcript_reasoning import (
    ATTEMPT_SCHEMA,
    PROMPT_VERSION,
    RESULT_SCHEMA,
    parse_reasoning_response,
    scan_reasoning_jobs,
    verify_generation_artifact,
)

DEFAULT_ARTIFACT_DIR = Path(
    "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI/"
    "narrator_reasoning/gpt_5_6_sol_20260820"
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


def _load_scan_and_results(artifact_dir: Path):
    plan = _read_json(artifact_dir / "plan.json")
    scan = scan_reasoning_jobs(str(plan["game_id"]))
    jobs = {job.job_id: job for job in scan["jobs"]}
    plan_jobs = {job["job_id"]: job for job in plan["jobs"]}
    if set(jobs) != set(plan_jobs):
        raise ValueError("Current replay scan does not match the persisted plan")
    for job_id, job in jobs.items():
        if job.input_hash != plan_jobs[job_id]["input_hash"]:
            raise ValueError(f"Input hash changed for {job_id}")
    results = _read_jsonl(artifact_dir / "results.jsonl")
    return plan, scan, jobs, results


def export_failed_jobs(artifact_dir: Path, shard_count: int) -> dict:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    plan, _, jobs, results = _load_scan_and_results(artifact_dir)
    failed_ids = [row["job_id"] for row in results if row.get("status") == "error"]
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
    shards = []
    for shard_index in range(shard_count):
        job_ids = failed_ids[shard_index * chunk_size : (shard_index + 1) * chunk_size]
        if not job_ids:
            continue
        shard_name = f"shard_{shard_index:02d}"
        shard_dir = inputs_dir / shard_name
        shard_dir.mkdir()
        for job_id in job_ids:
            job = jobs[job_id]
            packet = {
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
                "location_tool_results": job.locations,
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
            _write_json(shard_dir / f"{job.replay_index:04d}.json", packet)
        shard = {
            "shard": shard_name,
            "job_count": len(job_ids),
            "job_ids": job_ids,
            "input_dir": str(shard_dir),
            "draft_path": str(drafts_dir / f"{shard_name}.jsonl"),
        }
        shards.append(shard)

    manifest = {
        "schema": "pi-narrator-reasoning-shards-v1",
        "game_id": plan["game_id"],
        "model_family": "gpt-5.6-sol",
        "failed_job_count": len(failed_ids),
        "shards": shards,
    }
    _write_json(artifact_dir / "pi_shards.json", manifest)
    return manifest


def _manual_result(job, plan: dict, paragraphs: list[dict]) -> tuple[dict, dict]:
    recorded_at = _utc_now()
    raw_response = json.dumps({"paragraphs": paragraphs})
    validated = parse_reasoning_response(job, raw_response)
    status = "ready" if validated else "empty"
    result = {
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
        "paragraphs": validated,
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


def ingest_pi_drafts(artifact_dir: Path) -> dict:
    plan, scan, jobs, results = _load_scan_and_results(artifact_dir)
    manifest = _read_json(artifact_dir / "pi_shards.json")
    expected_ids = {
        job_id for shard in manifest["shards"] for job_id in shard["job_ids"]
    }
    drafts = {}
    for shard in manifest["shards"]:
        draft_path = Path(shard["draft_path"])
        for row in _read_jsonl(draft_path):
            job_id = row.get("job_id")
            if job_id in drafts:
                raise ValueError(f"Duplicate Pi draft for {job_id}")
            if job_id not in expected_ids:
                raise ValueError(f"Unexpected Pi draft job {job_id}")
            if not isinstance(row.get("paragraphs"), list):
                raise ValueError(f"Pi draft paragraphs must be a list for {job_id}")
            drafts[job_id] = row
    missing = sorted(expected_ids - set(drafts))
    if missing:
        raise ValueError(f"Missing {len(missing)} Pi drafts: {missing}")

    latest = {row["job_id"]: row for row in results}
    attempts = []
    for job_id in sorted(expected_ids, key=lambda value: jobs[value].replay_index):
        result, attempt = _manual_result(jobs[job_id], plan, drafts[job_id]["paragraphs"])
        latest[job_id] = result
        attempts.append(attempt)
    ordered_results = [latest[job.job_id] for job in scan["jobs"]]
    _write_jsonl(artifact_dir / "results.jsonl", ordered_results)
    _append_jsonl(artifact_dir / "attempts.jsonl", attempts)
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
