#!/usr/bin/env python
"""Evaluate information-equivalent Catan board text representations."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import httpx
from dotenv import load_dotenv

from evals.catan_board_bench.scoring import (
    VISUAL_CATEGORIES,
    score_answer,
    select_questions,
    sentinel_hint,
)
from evals.catan_board_bench.text_representations import (
    REPRESENTATION_NAMES,
    fact_digest,
    public_board_facts,
    render_representation,
)


load_dotenv()

BENCH_DIR = Path("evals/catan_board_bench/datasets/catan_board_bench_100")
QUESTION_DIR = BENCH_DIR / "questions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = """You are answering engine-scored questions about an authoritative public Catan board state encoded as text.

Rules:
- Use only the supplied board state.
- Literal tokens such as <T07>, <N18>, and <E03_17> are canonical identifiers.
- In sparse formats, declared EMPTY defaults apply to omitted nodes and edges.
- Derive requested counts from the listed buildings and roads; no hidden state is present.
- Return only the final answer string, with no explanation.
- Preserve exact tokens such as <RED>, <SETTLEMENT>, <WOOD>, and <ORE>."""

JsonDict = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument(
        "--representations",
        default=",".join(REPRESENTATION_NAMES),
    )
    parser.add_argument("--categories", default=",".join(VISUAL_CATEGORIES))
    parser.add_argument("--limit-samples", type=int, default=10)
    parser.add_argument("--questions-per-sample", type=int, default=11)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--provider-order",
        default="AkashML",
        help="Comma-separated OpenRouter providers; empty leaves routing unpinned.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    representations = split_csv(args.representations)
    unknown = set(representations) - set(REPRESENTATION_NAMES)
    if unknown:
        raise SystemExit(f"Unknown representations: {sorted(unknown)}")
    categories = split_csv(args.categories)
    selected = select_questions(
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    representation_dir = args.output_dir / "representations"
    representation_dir.mkdir(exist_ok=True)

    facts_by_sample: dict[str, JsonDict] = {}
    digests: dict[str, str] = {}
    texts: dict[tuple[str, str], str] = {}
    for qa in selected:
        sample_id = qa["sample_id"]
        if sample_id in facts_by_sample:
            continue
        facts = public_board_facts(qa["contract"])
        facts_by_sample[sample_id] = facts
        digests[sample_id] = fact_digest(facts)
        sample_dir = representation_dir / sample_id
        sample_dir.mkdir(exist_ok=True)
        for representation in representations:
            text = render_representation(representation, facts)
            texts[(sample_id, representation)] = text
            (sample_dir / f"{representation}.txt").write_text(text + "\n")

    plan = {
        "suite": "catan_board_bench_text_formats",
        "model": args.model,
        "representations": representations,
        "bench_dir": str(BENCH_DIR),
        "question_dir": str(QUESTION_DIR),
        "question_count_per_representation": len(selected),
        "request_count": len(selected) * len(representations),
        "question_ids": [qa["id"] for qa in selected],
        "categories": categories,
        "limit_samples": args.limit_samples,
        "questions_per_sample": args.questions_per_sample,
        "fact_digests": digests,
        "request_settings": {
            "provider": "openrouter",
            "provider_order": split_csv(args.provider_order),
            "allow_fallbacks": False if split_csv(args.provider_order) else None,
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
    write_json(args.output_dir / "plan.json", plan)
    print(json.dumps(plan, indent=2))

    if args.dry_run:
        first = selected[0]
        for representation in representations:
            print(f"\n--- {representation} ---")
            print(
                build_prompt(
                    representation,
                    texts[(first["sample_id"], representation)],
                    first,
                )[:4000]
            )
        return

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    jobs = []
    for question_index, qa in enumerate(selected):
        rotation = question_index % len(representations)
        rotated = representations[rotation:] + representations[:rotation]
        for representation in rotated:
            jobs.append(
                {
                    "representation": representation,
                    "qa": qa,
                    "prompt": build_prompt(
                        representation,
                        texts[(qa["sample_id"], representation)],
                        qa,
                    ),
                    "fact_digest": digests[qa["sample_id"]],
                }
            )
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
                    prompt=job["prompt"],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    provider_order=split_csv(args.provider_order),
                ): job
                for job in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                qa = job["qa"]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"error": str(exc)}
                response = result.get("response", "")
                score = score_answer(qa, response)
                record = {
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
                    "prompt_characters": len(job["prompt"]),
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


def build_prompt(representation: str, board_text: str, qa: JsonDict) -> str:
    hint = sentinel_hint(qa)
    hint_text = f"{hint}\n\n" if hint else ""
    return (
        f"Representation: {representation}\n"
        "Authoritative public board state:\n"
        f"{board_text}\n\n"
        f"{hint_text}"
        f"Question: {qa['question']}\n\n"
        "Return only the answer."
    )


def call_openrouter_text(
    api_key: str,
    model_id: str,
    *,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    provider_order: Sequence[str],
) -> JsonDict:
    payload: JsonDict = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model_supports_reasoning_control(model_id):
        payload["reasoning"] = {"enabled": False}
        payload["max_tokens"] = max(max_tokens, 256)
    if provider_order:
        payload["provider"] = {
            "order": list(provider_order),
            "allow_fallbacks": False,
        }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/catan-learning",
        "X-Title": "CatanBoardBench Text Format Eval",
    }
    start = time.time()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(OPENROUTER_URL, headers=headers, json=payload)
    latency_ms = int((time.time() - start) * 1000)
    if response.status_code >= 400:
        return {
            "error": f"HTTP {response.status_code}: {response.text[:800]}",
            "latency_ms": latency_ms,
        }
    data = response.json()
    message = data["choices"][0]["message"]
    return {
        "response": extract_message_text(message),
        "latency_ms": latency_ms,
        "usage": data.get("usage", {}),
        "served_model": data.get("model", model_id),
        "provider": data.get("provider"),
    }


def summarize(records: Sequence[JsonDict], plan: JsonDict) -> JsonDict:
    by_representation: dict[str, list[JsonDict]] = defaultdict(list)
    for record in records:
        by_representation[record["representation"]].append(record)
    representations = {name: summarize_group(rows) for name, rows in by_representation.items()}
    for name, rows in by_representation.items():
        by_category: dict[str, list[JsonDict]] = defaultdict(list)
        for row in rows:
            by_category[row["category"]].append(row)
        representations[name]["categories"] = {
            category: summarize_group(category_rows)
            for category, category_rows in sorted(by_category.items())
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "representations": representations,
    }


def summarize_group(records: Sequence[JsonDict]) -> JsonDict:
    attempted = [record for record in records if not record.get("error")]
    component_correct = sum(record["score"]["component_correct"] for record in attempted)
    component_total = sum(record["score"]["component_total"] for record in attempted)
    latencies = [
        record["latency_ms"] for record in attempted if record.get("latency_ms") is not None
    ]
    usage_keys = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
    usage = {
        key: sum((record.get("usage") or {}).get(key, 0) or 0 for record in records)
        for key in usage_keys
    }
    return {
        "requests": len(records),
        "attempted": len(attempted),
        "errors": len(records) - len(attempted),
        "exact": sum(record["score"]["correct"] for record in attempted),
        "exact_accuracy": (
            sum(record["score"]["correct"] for record in attempted) / len(attempted)
            if attempted
            else 0.0
        ),
        "component_correct": component_correct,
        "component_total": component_total,
        "component_accuracy": (component_correct / component_total if component_total else 0.0),
        "avg_latency_ms": statistics.mean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "usage": usage,
        "prompt_characters": sum(record["prompt_characters"] for record in records),
    }


def extract_message_text(message: JsonDict) -> str:
    for field in (
        "content",
        "reasoning",
        "reasoning_content",
        "analysis",
        "summary",
    ):
        value = message.get(field)
        text = extract_content(value)
        if text:
            return text
    return extract_content(message)


def extract_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(text for item in value if (text := extract_content(item))).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning", "summary"):
            text = extract_content(value.get(key))
            if text:
                return text
    return ""


def model_supports_reasoning_control(model_id: str) -> bool:
    model_id_lower = model_id.lower()
    return any(family in model_id_lower for family in ("qwen3.5", "qwen3.6", "qwen3.7", "qwen3.8"))


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_json(path: Path, payload: JsonDict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
