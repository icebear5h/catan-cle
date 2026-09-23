"""Command line entry point for the CatanBoardBench VLM screening run."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from evals.catan_board_bench.scoring import PROBE_CATEGORIES, PROBE_SYSTEM_PROMPT
from scripts.board_bench.run.eval_catan_board_bench_openrouter.constants import (
    BENCH_DIR,
    DEFAULT_CATEGORIES,
    PROBE_DIR,
    PROBE_QUESTION_DIR,
    PROVIDER_ENV,
    QUESTION_DIR,
    RUNS_DIR,
    SYSTEM_PROMPT,
    resolve_model_specs,
    split_csv,
    timestamp_slug,
)
from scripts.board_bench.run.eval_catan_board_bench_openrouter.plan import build_plan, parse_args
from scripts.board_bench.run.eval_catan_board_bench_openrouter.questions import (
    build_prompt,
    select_questions,
)
from scripts.board_bench.run.eval_catan_board_bench_openrouter.scoring import score_answer
from scripts.board_bench.run.eval_catan_board_bench_openrouter.summary import summarize
from scripts.board_bench.run.eval_catan_board_bench_openrouter.transport import call_provider
from scripts.board_bench.shapes import JsonDict, obj, read_jsonl, text, write_json

__all__ = ["main"]


def _resume_state(
    response_path: Path, jobs: list[JsonDict]
) -> tuple[list[JsonDict], list[JsonDict]]:
    existing = read_jsonl(response_path)
    successful = {
        (text(record["model_id"], "model_id"), text(record["question_id"], "question_id")): record
        for record in existing
        if not record.get("error")
    }
    records = list(successful.values())
    remaining = [
        job
        for job in jobs
        if (
            text(job["model_id"], "model_id"),
            text(obj(job["qa"], "job qa")["id"], "question id"),
        )
        not in successful
    ]
    print(f"Resuming with {len(records)} successful rows; {len(remaining)} requests remain")
    return records, remaining


def _run_jobs(
    args: argparse.Namespace,
    *,
    jobs: list[JsonDict],
    records: list[JsonDict],
    api_key: str,
    bench_dir: Path,
    system_prompt: str,
    provider_order: list[str],
    response_path: Path,
) -> None:
    print(f"Running {len(jobs)} requests with concurrency={args.concurrency}")
    with response_path.open("w") as out:
        for record in records:
            out.write(json.dumps(record, sort_keys=True) + "\n")
        out.flush()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    call_provider,
                    args.provider,
                    api_key,
                    text(job["model_id"], "model_id"),
                    image_bytes=(
                        bench_dir / text(obj(job["qa"], "job qa")["image_path"], "image_path")
                    ).read_bytes(),
                    prompt=text(job["prompt"], "job prompt"),
                    system_prompt=system_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    request_interval=args.request_interval,
                    provider_order=provider_order,
                    allow_provider_fallbacks=args.allow_provider_fallbacks,
                    disable_reasoning=args.disable_reasoning,
                ): job
                for job in jobs
            }

            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                qa = obj(job["qa"], "job qa")
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"error": str(exc)}

                response = text(result.get("response", ""), "response")
                score = score_answer(qa, response)
                record = {
                    "model_key": job["model_key"],
                    "model_id": job["model_id"],
                    "sample_id": qa["sample_id"],
                    "question_id": qa["id"],
                    "category": qa["category"],
                    "question": qa["question"],
                    "expected": qa["answer"],
                    "response": response,
                    "score": score,
                    "latency_ms": result.get("latency_ms"),
                    "usage": result.get("usage", {}),
                    "served_model": result.get("served_model"),
                    "provider": result.get("provider"),
                    "error": result.get("error"),
                }
                if args.save_prompts:
                    record["prompt"] = job["prompt"]
                out.write(json.dumps(record, sort_keys=True) + "\n")
                out.flush()
                records.append(record)

                status = "ERR" if record["error"] else ("OK" if score["correct"] else "MISS")
                print(
                    f"{status} {job['model_key']} {qa['id']} "
                    f"expected={qa['answer']!r} got={response[:80]!r}"
                )


def main() -> None:
    args = parse_args()
    provider_order = split_csv(args.openrouter_provider_order or "")
    if args.provider != "openrouter" and (
        provider_order or args.allow_provider_fallbacks or args.disable_reasoning
    ):
        raise SystemExit("OpenRouter routing/reasoning flags require --provider openrouter")
    if args.allow_provider_fallbacks and not provider_order:
        raise SystemExit("--allow-provider-fallbacks requires --openrouter-provider-order")

    model_specs = resolve_model_specs(split_csv(args.models))
    bench_dir = args.bench_dir or (PROBE_DIR if args.suite == "probe" else BENCH_DIR)
    question_dir = args.question_dir or (
        PROBE_QUESTION_DIR if args.suite == "probe" else QUESTION_DIR
    )
    categories = (
        split_csv(args.categories)
        if args.categories
        else (list(PROBE_CATEGORIES) if args.suite == "probe" else list(DEFAULT_CATEGORIES))
    )
    questions_per_sample = args.questions_per_sample or len(categories)
    use_atlas_prompt = (not args.no_atlas_prompt) and args.suite != "probe"
    system_prompt = PROBE_SYSTEM_PROMPT if args.suite == "probe" else SYSTEM_PROMPT
    selected = select_questions(
        bench_dir,
        question_dir=question_dir,
        categories=categories,
        limit_samples=args.limit_samples,
        questions_per_sample=questions_per_sample,
        max_requests=args.max_requests,
        selection_mode="flat" if args.suite == "probe" else "sample",
    )

    output_dir = args.output_dir or RUNS_DIR / bench_dir.name / "openrouter" / timestamp_slug()
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = build_plan(
        args,
        bench_dir=bench_dir,
        question_dir=question_dir,
        model_specs=model_specs,
        selected=selected,
        categories=categories,
        questions_per_sample=questions_per_sample,
        use_atlas_prompt=use_atlas_prompt,
        provider_order=provider_order,
        output_dir=output_dir,
    )
    write_json(output_dir / "plan.json", plan)

    print(json.dumps(plan, indent=2))

    if args.dry_run:
        if selected:
            item = selected[0]
            prompt = build_prompt(item, use_atlas=use_atlas_prompt)
            print("\n--- first prompt ---")
            print(prompt)
            print("--- expected ---")
            print(item["answer"])
        print(f"\nDry run wrote plan to {output_dir / 'plan.json'}")
        return

    env_var = PROVIDER_ENV[args.provider]
    api_key = os.getenv(env_var)
    if not api_key:
        raise SystemExit(
            f"{env_var} is not set. Put it in .env or export it in the shell; do not paste it into chat."
        )

    response_path = output_dir / "responses.jsonl"
    summary_path = output_dir / "summary.json"

    jobs: list[JsonDict] = [
        {
            "model_key": model_key,
            "model_id": model_id,
            "qa": qa,
            "prompt": build_prompt(qa, use_atlas=use_atlas_prompt),
        }
        for model_key, model_id in model_specs.items()
        for qa in selected
    ]

    records: list[JsonDict] = []
    if args.resume and response_path.exists():
        records, jobs = _resume_state(response_path, jobs)

    _run_jobs(
        args,
        jobs=jobs,
        records=records,
        api_key=api_key,
        bench_dir=bench_dir,
        system_prompt=system_prompt,
        provider_order=provider_order,
        response_path=response_path,
    )

    summary = summarize(records, plan)
    write_json(summary_path, summary)
    print("\nSummary")
    print(json.dumps(summary["models"], indent=2, sort_keys=True))
    print(f"\nResponses: {response_path}")
    print(f"Summary: {summary_path}")
