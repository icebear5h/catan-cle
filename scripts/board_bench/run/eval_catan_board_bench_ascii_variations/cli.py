"""Command line entry point for the ASCII variation probe."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from evals.catan_board_bench.ascii_variations import ASCII_VARIANTS
from scripts.board_bench.run.eval_catan_board_bench_ascii_variations.constants import (
    DEFAULT_DATASET_DIR,
    job_key,
    split_csv,
)
from scripts.board_bench.run.eval_catan_board_bench_ascii_variations.jobs import (
    build_jobs,
    build_plan,
    validate_or_write_plan,
)
from scripts.board_bench.run.eval_catan_board_bench_ascii_variations.records import (
    latest_records,
    response_record,
    successful_record,
)
from scripts.board_bench.run.eval_catan_board_bench_ascii_variations.summary import summarize
from scripts.board_bench.run.eval_catan_board_bench_ascii_variations.transport import (
    call_openrouter_text,
)
from scripts.board_bench.shapes import JsonDict, obj, read_jsonl, text, write_json

# The pre-split module path stays the advertised program name and description.
PROG = "eval_catan_board_bench_ascii_variations.py"
DESCRIPTION = "Evaluate strict Catan full-graph ASCII variations through OpenRouter."

__all__ = ["DESCRIPTION", "PROG", "main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument("--variants", default=",".join(ASCII_VARIANTS))
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


def _select_questions(args: argparse.Namespace) -> list[JsonDict]:
    questions = read_jsonl(args.dataset_dir / "qa.jsonl")
    categories = split_csv(args.categories)
    if categories:
        questions = [row for row in questions if row["category"] in categories]
    if args.max_questions is not None:
        questions = questions[: args.max_questions]
    if not questions:
        raise SystemExit("No questions selected")
    return questions


def _run_pending(
    args: argparse.Namespace,
    pending: list[JsonDict],
    plan: JsonDict,
    latest: dict[tuple[str, str], JsonDict],
    provider_order: list[str],
) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    response_path = args.output_dir / "responses.jsonl"
    with response_path.open("a") as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    call_openrouter_text,
                    api_key,
                    args.model,
                    prompt=text(job["prompt"], "job prompt"),
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
                latest[job_key(job)] = record
                status = (
                    "ERR"
                    if record["error"]
                    else ("OK" if obj(record["score"], "score")["correct"] else "MISS")
                )
                print(
                    f"[{completed}/{len(pending)}] {status} "
                    f"{job['variant']} {obj(job['qa'], 'job qa')['id']} "
                    f"got={text(record['response'], 'response')[:100]!r}"
                )


def main() -> None:
    args = parse_args()
    variants = split_csv(args.variants)
    unknown = set(variants) - set(ASCII_VARIANTS)
    if unknown:
        raise SystemExit(f"Unknown variants: {sorted(unknown)}")
    if not variants:
        raise SystemExit("At least one variant is required")

    questions = _select_questions(args)
    jobs = build_jobs(
        args.dataset_dir,
        questions=questions,
        variants=variants,
        max_requests=args.max_requests,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provider_order = split_csv(args.provider_order)
    plan = build_plan(
        args,
        questions=questions,
        variants=variants,
        jobs=jobs,
        provider_order=provider_order,
    )
    validate_or_write_plan(args.output_dir / "plan.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True))

    if args.dry_run:
        for job in jobs[: min(6, len(jobs))]:
            print(
                f"\n--- {job['variant']} / {obj(job['qa'], 'job qa')['id']} ---\n"
                f"{text(job['prompt'], 'job prompt')[:5000]}"
            )
        return

    response_path = args.output_dir / "responses.jsonl"
    prior_records = read_jsonl(response_path) if response_path.exists() else []
    latest = latest_records(prior_records)
    pending = [
        job
        for job in jobs
        if not successful_record(latest.get(job_key(job)), job=job, plan=plan)
    ]
    print(
        f"Running {len(pending)} pending of {len(jobs)} requests "
        f"with concurrency={args.concurrency}"
    )

    if pending:
        _run_pending(args, pending, plan, latest, provider_order)

    final_records = [latest[job_key(job)] for job in jobs if job_key(job) in latest]
    summary = summarize(final_records, plan)
    write_json(args.output_dir / "summary.json", summary)
    print("\nSummary")
    print(json.dumps(summary["variants"], indent=2, sort_keys=True))
