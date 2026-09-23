"""Command line entry point for the tile prompt ablation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from collections.abc import Sequence
from pathlib import Path

from scripts.board_bench.run.eval_catan_tile_prompt_ablation import (
    CONDITIONS,
    sha256_bytes,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.defaults import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_QA_PATH,
    TileJob,
    TileQuestion,
    acquire_run_lock,
    job_key,
    read_jsonl,
    release_run_lock,
    split_csv,
    validate_request_contract,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.plan import (
    build_plan,
    validate_or_write_plan,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.prompts import build_prompt
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.questions import (
    build_jobs,
    load_questions,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.records import (
    accepted_records,
    response_record,
)
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.summary import summarize
from scripts.board_bench.run.eval_catan_tile_prompt_ablation.transport import (
    call_openrouter_image,
)
from scripts.board_bench.shapes import JsonDict, obj, text

# The pre-split module path stays the advertised program name and description.
PROG = "eval_catan_tile_prompt_ablation.py"
DESCRIPTION = "Evaluate isolated Catan tile prompts across label and guidance styles."

__all__ = ["DESCRIPTION", "PROG", "main", "parse_args", "run"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
    parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider-order", default=DEFAULT_PROVIDER)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--seed", type=int, default=38_271)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _prepare(pending: Sequence[TileJob]) -> list[tuple[TileJob, bytes]]:
    prepared: list[tuple[TileJob, bytes]] = []
    for job in pending:
        image_bytes = Path(job["qa"]["image_path"]).read_bytes()
        if sha256_bytes(image_bytes) != job["qa"]["image_sha256"]:
            raise SystemExit(
                "Image changed after plan construction: " f"{job['qa']['image_path']}"
            )
        prepared.append((job, image_bytes))
    return prepared


def _dispatch(
    args: argparse.Namespace,
    *,
    pending: list[TileJob],
    plan: JsonDict,
    provider_order: Sequence[str],
    response_path: Path,
) -> None:
    prepared = _prepare(pending)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    with response_path.open("a") as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    call_openrouter_image,
                    api_key,
                    args.model,
                    image_bytes=image_bytes,
                    prompt=job["prompt"],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    provider_order=provider_order,
                ): job
                for job, image_bytes in prepared
            }
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"error": str(exc)}
                record = response_record(job, result, plan)
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                status = (
                    "ERROR"
                    if record["error"]
                    else "OK"
                    if obj(record["score"], "score")["pair_correct"]
                    else "MISS"
                )
                print(
                    f"[{completed}/{len(pending)}] {status} "
                    f"{job['condition']} {job['qa']['sample_id']} "
                    f"got={text(record['response'], 'response')!r}"
                )


def run(
    args: argparse.Namespace,
    *,
    questions: Sequence[TileQuestion],
    conditions: Sequence[str],
    provider_order: Sequence[str],
    jobs: Sequence[TileJob],
) -> None:
    plan = build_plan(
        args,
        questions=questions,
        conditions=conditions,
        provider_order=provider_order,
        jobs=jobs,
    )
    validate_or_write_plan(args.output_dir / "plan.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        for condition in conditions:
            print(f"\n--- {condition} ---\n{build_prompt(condition)}")
        return

    response_path = args.output_dir / "responses.jsonl"
    prior = read_jsonl(response_path) if response_path.exists() else []
    accepted = accepted_records(prior, jobs=jobs, plan=plan)
    pending = [job for job in jobs if job_key(job) not in accepted]
    print(
        f"Running {len(pending)} pending of {len(jobs)} requests "
        f"with concurrency={args.concurrency}"
    )
    if pending:
        _dispatch(
            args,
            pending=pending,
            plan=plan,
            provider_order=provider_order,
            response_path=response_path,
        )

    all_records = read_jsonl(response_path) if response_path.exists() else []
    accepted = accepted_records(all_records, jobs=jobs, plan=plan)
    summary = summarize(accepted, jobs=jobs, plan=plan)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("\nSummary")
    print(json.dumps(summary["conditions"], indent=2, sort_keys=True))
    if not summary["complete"]:
        label = "Preflight" if args.preflight else "Evaluation"
        raise SystemExit(f"{label} is incomplete; rerun to resume pending jobs")
    if args.preflight and (
        len(jobs) != 4 or {job["condition"] for job in jobs} != set(CONDITIONS)
    ):
        raise SystemExit("Preflight plan must contain one job per condition")
    if args.preflight and any(
        obj(row["score"], "score")["protocol_valid"] is not True
        for row in accepted.values()
    ):
        raise SystemExit("Preflight produced a protocol-invalid response")


def main() -> None:
    args = parse_args()
    conditions = split_csv(args.conditions)
    provider_order = split_csv(args.provider_order)
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.max_requests is not None and args.max_requests <= 0:
        raise SystemExit("--max-requests must be positive")
    validate_request_contract(args.model, provider_order)
    questions = load_questions(args.qa_path, args.image_root)
    jobs = build_jobs(
        questions,
        conditions=conditions,
        seed=args.seed,
        preflight=args.preflight,
        max_requests=args.max_requests,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path, lock_fd = acquire_run_lock(args.output_dir)
    try:
        run(
            args,
            questions=questions,
            conditions=conditions,
            provider_order=provider_order,
            jobs=jobs,
        )
    finally:
        release_run_lock(lock_path, lock_fd)
