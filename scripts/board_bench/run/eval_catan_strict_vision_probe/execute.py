"""Request execution with a hard wall-clock deadline per job."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import time
from collections.abc import Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Protocol

from scripts.board_bench.builders.render_catan_strict_vision_probe import write_json
from scripts.board_bench.run.eval_catan_board_bench_openrouter import call_novita, call_openrouter
from scripts.board_bench.run.eval_catan_strict_vision_probe.constants import (
    SYSTEM_PROMPT,
    VisionJob,
)
from scripts.board_bench.run.eval_catan_strict_vision_probe.jobs import (
    build_plan,
    validate_or_write_plan,
)
from scripts.board_bench.run.eval_catan_strict_vision_probe.records import (
    accepted_records,
    admissible_record,
    recompute_record_score,
    response_record,
)
from scripts.board_bench.run.eval_catan_strict_vision_probe.summary import summarize
from scripts.board_bench.shapes import JsonDict, obj, read_jsonl, text

__all__ = [
    "ResultSink",
    "WorkerTarget",
    "call_job",
    "call_job_in_child",
    "call_job_with_hard_deadline",
    "persist_result",
    "run_evaluation",
]


class ResultSink(Protocol):
    """The response-log sink this module writes through."""

    def write(self, data: str, /) -> int: ...

    def flush(self) -> None: ...


class WorkerTarget(Protocol):
    """A child-process entry point that sends one result back."""

    def __call__(
        self,
        result_sender: Connection[JsonDict, JsonDict],
        api_key: str,
        args: argparse.Namespace,
        job: VisionJob,
        /,
    ) -> None: ...


def call_job(api_key: str, args: argparse.Namespace, job: VisionJob) -> JsonDict:
    provider = getattr(args, "provider", "novita")
    try:
        if provider == "novita":
            return call_novita(
                api_key,
                args.model,
                image_bytes=job["board_presentation"].data,
                prompt=job["prompt"],
                system_prompt=SYSTEM_PROMPT,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
        return call_openrouter(
            api_key,
            args.model,
            image_bytes=job["board_presentation"].data,
            prompt=job["prompt"],
            system_prompt=SYSTEM_PROMPT,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            disable_reasoning=True,
        )
    finally:
        if args.request_interval:
            time.sleep(args.request_interval)


def call_job_in_child(
    result_sender: Connection[JsonDict, JsonDict],
    api_key: str,
    args: argparse.Namespace,
    job: VisionJob,
) -> None:
    try:
        result_sender.send(call_job(api_key, args, job))
    except BaseException as exc:
        result_sender.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        result_sender.close()


def call_job_with_hard_deadline(
    api_key: str,
    args: argparse.Namespace,
    job: VisionJob,
    *,
    worker_target: WorkerTarget | None = None,
) -> JsonDict:
    context = multiprocessing.get_context("spawn")
    result_receiver, result_sender = context.Pipe(duplex=False)
    process = context.Process(
        target=worker_target or call_job_in_child,
        args=(result_sender, api_key, args, job),
    )
    started = time.monotonic()
    process.start()
    result_sender.close()
    process.join(args.timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        result: JsonDict = {
            "error": f"request exceeded hard wall-clock timeout of {args.timeout}s",
            "latency_ms": int((time.monotonic() - started) * 1000),
        }
    elif result_receiver.poll(timeout=1):
        result = obj(result_receiver.recv(), "worker result")
    else:
        result = {
            "error": f"request worker exited without a result (exitcode={process.exitcode})",
            "latency_ms": int((time.monotonic() - started) * 1000),
        }
    result_receiver.close()
    return result


def persist_result(
    output: ResultSink,
    *,
    job: VisionJob,
    result: JsonDict,
    plan: JsonDict,
    accepted: dict[str, JsonDict],
    completed: int,
    pending_count: int,
) -> None:
    record = response_record(job, result, plan)
    output.write(json.dumps(record, sort_keys=True) + "\n")
    output.flush()
    if admissible_record(record, job=job, plan=plan):
        accepted.setdefault(
            text(job["qa"]["id"], "question id"), recompute_record_score(record, job["qa"])
        )
    status = (
        "ERR"
        if record["error"]
        else ("OK" if obj(record["score"], "record score")["correct"] else "MISS")
    )
    print(
        f"[{completed}/{pending_count}] {status} {job['qa']['id']} "
        f"got={text(record['response'], 'response')[:120]!r}"
    )


def _api_key_for(provider: str) -> str:
    api_key_name = "NOVITA_API_KEY" if provider == "novita" else "OPENROUTER_API_KEY"
    api_key = os.getenv(api_key_name)
    if not api_key:
        raise SystemExit(f"{api_key_name} is not set")
    return api_key


def _write_pending(
    args: argparse.Namespace,
    *,
    response_path: Path,
    pending: list[VisionJob],
    plan: JsonDict,
    accepted: dict[str, JsonDict],
    api_key: str,
) -> None:
    mode = "a" if args.resume else "w"
    with response_path.open(mode) as output:
        if args.concurrency == 1:
            for completed, job in enumerate(pending, start=1):
                try:
                    result = call_job_with_hard_deadline(api_key, args, job)
                except Exception as exc:
                    result = {"error": str(exc)}
                persist_result(
                    output,
                    job=job,
                    result=result,
                    plan=plan,
                    accepted=accepted,
                    completed=completed,
                    pending_count=len(pending),
                )
        elif pending:
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
                    persist_result(
                        output,
                        job=job,
                        result=result,
                        plan=plan,
                        accepted=accepted,
                        completed=completed,
                        pending_count=len(pending),
                    )


def run_evaluation(
    args: argparse.Namespace,
    *,
    metadata: JsonDict,
    questions: Sequence[JsonDict],
    jobs: Sequence[VisionJob],
) -> None:
    plan = build_plan(args, metadata=metadata, questions=questions, jobs=jobs)
    validate_or_write_plan(args.output_dir / "plan.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        print(f"\n--- {jobs[0]['qa']['id']} ---\n{jobs[0]['prompt']}")
        return

    provider = getattr(args, "provider", "novita")
    api_key = _api_key_for(provider)
    response_path = args.output_dir / "responses.jsonl"
    existing = read_jsonl(response_path) if args.resume and response_path.exists() else []
    accepted = accepted_records(existing, jobs=jobs, plan=plan)
    pending = [job for job in jobs if text(job["qa"]["id"], "question id") not in accepted]
    print(
        f"Running {len(pending)} pending of {len(jobs)} requests with concurrency={args.concurrency}"
    )

    _write_pending(
        args,
        response_path=response_path,
        pending=pending,
        plan=plan,
        accepted=accepted,
        api_key=api_key,
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
