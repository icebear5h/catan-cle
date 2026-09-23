"""Question selection, representation rendering, and the request loop."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from evals.catan_board_bench.scoring import score_answer, select_questions
from evals.catan_board_bench.text_representations import (
    REPRESENTATION_NAMES,
    BoardFacts,
    fact_digest,
    public_board_facts,
    render_representation,
)
from scripts.board_bench.run.eval_catan_board_bench_text_formats.constants import (
    BENCH_DIR,
    QUESTION_DIR,
    build_prompt,
    model_supports_reasoning_control,
    split_csv,
)
from scripts.board_bench.run.eval_catan_board_bench_text_formats.summary import summarize
from scripts.board_bench.run.eval_catan_board_bench_text_formats.transport import (
    call_openrouter_text,
)
from scripts.board_bench.shapes import JsonDict, obj, text, write_json

__all__ = [
    "RenderedBoards",
    "build_plan",
    "check_representations",
    "render_boards",
    "run_requests",
    "select_text_questions",
]


class RenderedBoards:
    """Per-sample public facts, digests, and rendered representation text."""

    def __init__(self) -> None:
        self.facts_by_sample: dict[str, BoardFacts] = {}
        self.digests: dict[str, str] = {}
        self.texts: dict[tuple[str, str], str] = {}


def render_boards(
    selected: list[JsonDict],
    representations: list[str],
    representation_dir: Path,
) -> RenderedBoards:
    rendered = RenderedBoards()
    for qa in selected:
        sample_id = text(qa["sample_id"], "sample_id")
        if sample_id in rendered.facts_by_sample:
            continue
        facts = public_board_facts(obj(qa["contract"], "question contract"))
        rendered.facts_by_sample[sample_id] = facts
        rendered.digests[sample_id] = fact_digest(facts)
        sample_dir = representation_dir / sample_id
        sample_dir.mkdir(exist_ok=True)
        for representation in representations:
            board_text = render_representation(representation, facts)
            rendered.texts[(sample_id, representation)] = board_text
            (sample_dir / f"{representation}.txt").write_text(board_text + "\n")
    return rendered


def build_plan(
    args: argparse.Namespace,
    representations: list[str],
    categories: list[str],
    selected: list[JsonDict],
    digests: dict[str, str],
) -> JsonDict:
    provider_order = split_csv(args.provider_order)
    return {
        "suite": "catan_board_bench_text_formats",
        "model": args.model,
        "representations": list(representations),
        "bench_dir": str(BENCH_DIR),
        "question_dir": str(QUESTION_DIR),
        "question_count_per_representation": len(selected),
        "request_count": len(selected) * len(representations),
        "question_ids": [qa["id"] for qa in selected],
        "categories": list(categories),
        "limit_samples": args.limit_samples,
        "questions_per_sample": args.questions_per_sample,
        "fact_digests": {key: value for key, value in digests.items()},
        "request_settings": {
            "provider": "openrouter",
            "provider_order": list(provider_order),
            "allow_fallbacks": False if provider_order else None,
            "concurrency": args.concurrency,
            "temperature": args.temperature,
            "requested_max_tokens": args.max_tokens,
            "effective_max_tokens": max(args.max_tokens, 256)
            if model_supports_reasoning_control(args.model)
            else args.max_tokens,
            "reasoning_disabled": model_supports_reasoning_control(args.model),
            "timeout_seconds": args.timeout,
            "image_input": False,
        },
        "output_dir": str(args.output_dir),
    }


def _build_jobs(
    selected: list[JsonDict],
    representations: list[str],
    rendered: RenderedBoards,
) -> list[JsonDict]:
    jobs: list[JsonDict] = []
    for question_index, qa in enumerate(selected):
        rotation = question_index % len(representations)
        rotated = representations[rotation:] + representations[:rotation]
        sample_id = text(qa["sample_id"], "sample_id")
        for representation in rotated:
            jobs.append(
                {
                    "representation": representation,
                    "qa": qa,
                    "prompt": build_prompt(
                        representation,
                        rendered.texts[(sample_id, representation)],
                        qa,
                    ),
                    "fact_digest": rendered.digests[sample_id],
                }
            )
    return jobs


def run_requests(
    args: argparse.Namespace,
    representations: list[str],
    selected: list[JsonDict],
    rendered: RenderedBoards,
    plan: JsonDict,
) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    jobs = _build_jobs(selected, representations, rendered)
    print(f"Running {len(jobs)} text requests with concurrency={args.concurrency}")

    records: list[JsonDict] = []
    response_path = args.output_dir / "responses.jsonl"
    with response_path.open("w") as output:
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
                    provider_order=split_csv(args.provider_order),
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
                record: JsonDict = {
                    "model_id": args.model,
                    "representation": job["representation"],
                    "sample_id": qa["sample_id"],
                    "question_id": qa["id"],
                    "category": qa["category"],
                    "question": qa["question"],
                    "expected": qa["answer"],
                    "response": response,
                    "score": score,
                    "fact_digest": job["fact_digest"],
                    "prompt_characters": len(text(job["prompt"], "job prompt")),
                    "latency_ms": result.get("latency_ms"),
                    "usage": result.get("usage", {}),
                    "served_model": result.get("served_model"),
                    "provider": result.get("provider"),
                    "error": result.get("error"),
                }
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                records.append(record)
                status = "ERR" if record["error"] else ("OK" if score["correct"] else "MISS")
                print(
                    f"{status} {job['representation']} {qa['id']} "
                    f"expected={qa['answer']!r} got={response[:80]!r}"
                )

    summary = summarize(records, plan)
    write_json(args.output_dir / "summary.json", summary)
    print("\nSummary")
    print(json.dumps(summary["representations"], indent=2, sort_keys=True))


def select_text_questions(args: argparse.Namespace, categories: list[str]) -> list[JsonDict]:
    selected: list[JsonDict] = select_questions(
        BENCH_DIR,
        question_dir=QUESTION_DIR,
        categories=categories,
        limit_samples=args.limit_samples,
        questions_per_sample=args.questions_per_sample,
        max_requests=args.max_requests,
        selection_mode="sample",
    )
    if not selected:
        raise SystemExit("No questions selected")
    if any("contract" not in qa for qa in selected):
        raise SystemExit("Every text-format question requires an attached contract")
    return selected


def check_representations(representations: list[str]) -> None:
    unknown = set(representations) - set(REPRESENTATION_NAMES)
    if unknown:
        raise SystemExit(f"Unknown representations: {sorted(unknown)}")
