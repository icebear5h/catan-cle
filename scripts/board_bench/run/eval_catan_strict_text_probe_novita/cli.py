"""Command line entry point for the strict text probe on Novita."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from collections.abc import Sequence
from pathlib import Path

from scripts.board_bench.builders.render_catan_strict_vision_probe import read_jsonl, write_json
from scripts.board_bench.run import eval_catan_board_bench_full_graph_formats as text_evaluator
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats import FormatJob
from scripts.board_bench.run.eval_catan_strict_text_probe_novita.constants import (
    DEFAULT_DATASET_DIR,
    FORMAT_NAME,
    configured_text_evaluator,
    select_questions,
    split_csv,
    validate_cli_args,
)
from scripts.board_bench.run.eval_catan_strict_text_probe_novita.plan import (
    build_plan,
    validate_or_write_plan,
)
from scripts.board_bench.run.eval_catan_strict_text_probe_novita.records import (
    accepted_records,
    admissible_record,
    recompute_record_score,
    response_record,
    summarize,
)
from scripts.board_bench.run.eval_catan_strict_text_probe_novita.transport import call_job
from scripts.board_bench.run.eval_catan_strict_vision_probe import (
    acquire_run_lock,
    release_run_lock,
)
from scripts.board_bench.shapes import JsonDict, obj, read_json_object, text

# The pre-split module path stays the advertised program name and description.
PROG = "eval_catan_strict_text_probe_novita.py"
DESCRIPTION = "Evaluate the frozen strict 60-question text projection through Novita."

__all__ = ["DESCRIPTION", "PROG", "main", "parse_args", "run_evaluation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model", default="qwen/qwen3.8-max")
    parser.add_argument("--categories", default="")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--request-interval", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _dispatch(
    args: argparse.Namespace,
    *,
    api_key: str,
    pending: list[FormatJob],
    plan: JsonDict,
    accepted: dict[str, JsonDict],
    response_path: Path,
) -> None:
    mode = "a" if args.resume else "w"
    with response_path.open(mode) as output:
        if pending:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {pool.submit(call_job, api_key, args, job): job for job in pending}
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
                            text(job["qa"]["id"], "question id"),
                            recompute_record_score(record, job["qa"]),
                        )
                    status = (
                        "ERR"
                        if record["error"]
                        else ("OK" if obj(record["score"], "score")["correct"] else "MISS")
                    )
                    print(
                        f"[{completed}/{len(pending)}] {status} {job['qa']['id']} "
                        f"got={text(record['response'], 'response')[:120]!r}"
                    )


def run_evaluation(
    args: argparse.Namespace,
    *,
    metadata: JsonDict,
    questions: Sequence[JsonDict],
    jobs: Sequence[FormatJob],
) -> None:
    plan = build_plan(args, metadata=metadata, questions=questions, jobs=jobs)
    validate_or_write_plan(args.output_dir / "plan.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        print(f"\n--- {jobs[0]['qa']['id']} ---\n{jobs[0]['prompt'][:8000]}")
        return

    api_key = os.getenv("NOVITA_API_KEY")
    if not api_key:
        raise SystemExit("NOVITA_API_KEY is not set")
    response_path = args.output_dir / "responses.jsonl"
    existing = read_jsonl(response_path) if args.resume and response_path.exists() else []
    accepted = accepted_records(existing, jobs=jobs, plan=plan)
    pending = [job for job in jobs if text(job["qa"]["id"], "question id") not in accepted]
    print(
        f"Running {len(pending)} pending of {len(jobs)} requests with concurrency={args.concurrency}"
    )

    _dispatch(
        args,
        api_key=api_key,
        pending=pending,
        plan=plan,
        accepted=accepted,
        response_path=response_path,
    )

    final_records = [
        accepted[text(job["qa"]["id"], "question id")]
        for job in jobs
        if text(job["qa"]["id"], "question id") in accepted
    ]
    summary = summarize(final_records, plan)
    write_json(args.output_dir / "summary.json", summary)
    print("\nSummary")
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))


def main() -> int:
    args = parse_args()
    validate_cli_args(
        args.max_questions,
        args.concurrency,
        args.max_tokens,
        args.timeout,
        args.request_interval,
    )
    with configured_text_evaluator():
        metadata = read_json_object(args.dataset_dir / "metadata.json")
        text_evaluator.validate_dataset_metadata(metadata)
        all_questions = text_evaluator.validate_dataset_integrity(args.dataset_dir, metadata)
        questions = select_questions(
            all_questions,
            categories=split_csv(args.categories),
            max_questions=args.max_questions,
        )
        jobs = text_evaluator.build_jobs(
            args.dataset_dir,
            questions=questions,
            formats=[FORMAT_NAME],
            max_requests=None,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path, lock_fd = acquire_run_lock(args.output_dir)
    try:
        run_evaluation(args, metadata=metadata, questions=questions, jobs=jobs)
    finally:
        release_run_lock(lock_path, lock_fd)
    return 0
