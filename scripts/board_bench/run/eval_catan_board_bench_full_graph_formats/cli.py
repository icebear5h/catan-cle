"""Command line entry point for the full-graph format probe."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from collections.abc import Sequence
from pathlib import Path

from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats import dataset_config
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.constants import (
    FormatJob,
    acquire_run_lock,
    job_key,
    read_jsonl,
    release_run_lock,
    split_csv,
    write_json,
)
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.dataset import (
    validate_dataset_integrity,
    validate_dataset_metadata,
    validate_request_contract,
)
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.jobs import (
    build_jobs,
    build_plan,
    validate_or_write_plan,
)
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.records import (
    accepted_records,
    admissible_record,
    recompute_record_score,
    response_record,
)
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.summary import summarize
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.transport import (
    call_openrouter_text,
)
from scripts.board_bench.shapes import JsonDict, obj, read_json_object, text

# The pre-split module path stays the advertised program name and description.
PROG = "eval_catan_board_bench_full_graph_formats.py"
DESCRIPTION = "Evaluate six lossless full-graph Catan text formats through OpenRouter."

__all__ = ["DESCRIPTION", "PROG", "main", "parse_args", "run_evaluation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
    parser.add_argument("--dataset-dir", type=Path, default=dataset_config.DEFAULT_DATASET_DIR)
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument("--formats", default=",".join(dataset_config.FORMAT_NAMES))
    parser.add_argument("--categories", default="")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--provider-order", default="AkashML")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_cli(args: argparse.Namespace, formats: list[str]) -> None:
    unknown = set(formats) - set(dataset_config.FORMAT_NAMES)
    if unknown:
        raise SystemExit(f"Unknown formats: {sorted(unknown)}")
    if not formats:
        raise SystemExit("At least one format is required")
    if len(formats) != len(set(formats)):
        raise SystemExit("Duplicate formats are not allowed")
    if args.max_questions is not None and args.max_questions <= 0:
        raise SystemExit("--max-questions must be positive")
    if args.max_requests is not None and args.max_requests <= 0:
        raise SystemExit("--max-requests must be positive")


def _dispatch(
    args: argparse.Namespace,
    *,
    pending: list[FormatJob],
    plan: JsonDict,
    accepted: dict[tuple[str, str], JsonDict],
    provider_order: Sequence[str],
    response_path: Path,
) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    with response_path.open("a") as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    call_openrouter_text,
                    api_key,
                    args.model,
                    prompt=job["prompt"],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    provider_order=provider_order,
                ): job
                for job in pending
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
                if admissible_record(record, job=job, plan=plan):
                    accepted.setdefault(
                        job_key(job),
                        recompute_record_score(record, job["qa"]),
                    )
                status = (
                    "ERR"
                    if record["error"]
                    else ("OK" if obj(record["score"], "score")["correct"] else "MISS")
                )
                print(
                    f"[{completed}/{len(pending)}] {status} "
                    f"{job['format']} {job['qa']['id']} "
                    f"got={text(record['response'], 'response')[:100]!r}"
                )


def run_evaluation(
    args: argparse.Namespace,
    *,
    dataset_metadata: JsonDict,
    questions: Sequence[JsonDict],
    formats: Sequence[str],
    jobs: Sequence[FormatJob],
    provider_order: Sequence[str],
) -> None:
    plan = build_plan(
        args,
        dataset_metadata=dataset_metadata,
        questions=questions,
        formats=formats,
        jobs=jobs,
        provider_order=provider_order,
    )
    validate_or_write_plan(args.output_dir / "plan.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True))

    if args.dry_run:
        for job in jobs[: min(6, len(jobs))]:
            print(f"\n--- {job['format']} / {job['qa']['id']} ---\n{job['prompt'][:5000]}")
        return

    response_path = args.output_dir / "responses.jsonl"
    attempts = read_jsonl(response_path) if response_path.exists() else []
    accepted = accepted_records(attempts, jobs=jobs, plan=plan)
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
            accepted=accepted,
            provider_order=provider_order,
            response_path=response_path,
        )

    final_records = [accepted[job_key(job)] for job in jobs if job_key(job) in accepted]
    summary = summarize(final_records, plan)
    write_json(args.output_dir / "summary.json", summary)
    print("\nSummary")
    print(json.dumps(summary["formats"], indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    formats = split_csv(args.formats)
    _validate_cli(args, formats)

    dataset_metadata = read_json_object(args.dataset_dir / "metadata.json")
    validate_dataset_metadata(dataset_metadata)
    all_questions = validate_dataset_integrity(args.dataset_dir, dataset_metadata)
    questions = all_questions
    categories = split_csv(args.categories)
    if categories:
        questions = [row for row in questions if row["category"] in categories]
    if args.max_questions is not None:
        questions = questions[: args.max_questions]
    if not questions:
        raise SystemExit("No questions selected")

    provider_order = split_csv(args.provider_order)
    validate_request_contract(args.model, provider_order)
    jobs = build_jobs(
        args.dataset_dir,
        questions=questions,
        formats=formats,
        max_requests=args.max_requests,
    )
    if len({job_key(job) for job in jobs}) != len(jobs):
        raise SystemExit("Generated format/question job keys are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path, lock_fd = acquire_run_lock(args.output_dir)
    try:
        run_evaluation(
            args,
            dataset_metadata=dataset_metadata,
            questions=questions,
            formats=formats,
            jobs=jobs,
            provider_order=provider_order,
        )
    finally:
        release_run_lock(lock_path, lock_fd)
