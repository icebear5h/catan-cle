#!/usr/bin/env python
"""Evaluate OpenRouter VLMs on CatanBoardBench engine-scored QA.

This is intended for cheap model screening before SFT. The generated benchmark
uses fixed engine atlas tokens like <T07>, <N18>, and <E03_17>, so this runner
can add local atlas context to make raw public VLMs comparable.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from dotenv import load_dotenv

from cle.game_engine.models.player import Color
from evals.catan_board_bench.scoring import (
    PROBE_CATEGORIES,
    PROBE_SYSTEM_PROMPT,
    score_hex_direction_answer,
)

load_dotenv()


BENCH_DIR = Path("evals/catan_board_bench/datasets/catan_board_bench_100")
QUESTION_DIR = BENCH_DIR / "questions"
RUNS_DIR = Path("artifacts/runs/catan_board_bench")
PROBE_DIR = Path("artifacts/generated/catan_board_bench/piece_recognition")
PROBE_QUESTION_DIR = PROBE_DIR / "questions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NOVITA_URL = "https://api.novita.ai/openai/v1/chat/completions"
MOONDREAM_URL = "https://api.moondream.ai/v1/query"
MOONDREAM_RESOLVE_IPS = ["104.21.54.37", "172.67.223.31"]
MOONDREAM_HOST = "api.moondream.ai"


SMALL_VLM_MODELS: Dict[str, str] = {
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma3-12b": "google/gemma-3-12b-it",
    "gemma3-27b": "google/gemma-3-27b-it",
    "gemma4-26b-a4b-free": "google/gemma-4-26b-a4b-it:free",
    "gemma4-31b-free": "google/gemma-4-31b-it:free",
    "nemotron-3-nano-free": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nemotron-12b-free": "nvidia/nemotron-nano-12b-v2-vl:free",
    "llama-3.2-11b": "meta-llama/llama-3.2-11b-vision-instruct",
    "qwen3-vl-8b": "qwen/qwen3-vl-8b-instruct",
    "qwen3-vl-8b-thinking": "qwen/qwen3-vl-8b-thinking",
    "qwen3.5-9b": "qwen/qwen3.5-9b",
    "qwen3.5-flash": "qwen/qwen3.5-flash-02-23",
    "moondream2": "moondream2",
    "ui-tars-7b": "bytedance/ui-tars-1.5-7b",
    "ministral-3b": "mistralai/ministral-3b-2512",
    "ministral-8b": "mistralai/ministral-8b-2512",
    "mistral-small-3.2-24b": "mistralai/mistral-small-3.2-24b-instruct",
    "mistral-small-24b": "mistralai/mistral-small-3.1-24b-instruct",
    "gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools": "google/gemini-3.1-pro-preview-customtools",
    "gemini-pro-latest": "~google/gemini-pro-latest",
}

DEFAULT_MODELS = [
    "gemma3-4b",
    "qwen3-vl-8b",
    "nemotron-12b-free",
    "ui-tars-7b",
]

DEFAULT_CATEGORIES = [
    "robber_tile",
    "robber_resource_number",
    "tile_resource_number",
    "tile_has_robber",
    "node_occupancy",
    "edge_road_owner",
    "color_road_locations",
    "port_trade_type",
    "port_occupancy",
    "nodes_connected",
    "edge_connects_nodes",
    "color_building_counts",
    "color_road_count",
]

SYSTEM_PROMPT = """You are answering engine-scored questions about a Catan board screenshot.

Rules:
- Use only visible public board information from the image plus the supplied fixed atlas context.
- Do not infer hidden hands or hidden development cards.
- Return only the final answer string, with no explanation.
- Preserve exact tokens such as <T07>, <N18>, <E03_17>, <RED>, and <WOOD>.
- If the answer is empty or absent, use the exact sentinel requested by the question context."""


JsonDict = Dict[str, Any]
COLOR_TOKEN_NAMES = [color.value for color in Color]
CATEGORY_ALIASES = {
    "isolated_tile_resource_number": "tile_resource_number",
    "local_patch_tile_resource_number": "tile_resource_number",
    "isolated_road_owner": "edge_road_owner",
    "local_patch_edge_road_owner": "edge_road_owner",
    "isolated_node_occupancy": "node_occupancy",
    "local_patch_node_occupancy": "node_occupancy",
    "isolated_port_trade_type": "port_trade_type",
    "local_patch_port_trade_type": "port_trade_type",
    "isolated_robber_presence": "robber_presence",
    "local_patch_robber_presence": "robber_presence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=["catan_board_bench", "probe"], default="catan_board_bench"
    )
    parser.add_argument("--bench-dir", type=Path, default=None)
    parser.add_argument("--question-dir", type=Path, default=None)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model keys or raw OpenRouter model ids.",
    )
    parser.add_argument(
        "--provider",
        default="openrouter",
        choices=["openrouter", "novita", "moondream"],
        help="Provider to call: openrouter (default), novita, or moondream.",
    )
    parser.add_argument(
        "--openrouter-provider-order",
        default=None,
        help=(
            "Comma-separated OpenRouter endpoint tags to pin, such as "
            "akashml/bf16. When set, unlisted-provider fallback is disabled by default."
        ),
    )
    parser.add_argument(
        "--allow-provider-fallbacks",
        action="store_true",
        help="Allow OpenRouter to route outside --openrouter-provider-order.",
    )
    parser.add_argument(
        "--disable-reasoning",
        action="store_true",
        help="Request reasoning.enabled=false for every OpenRouter model in this run.",
    )
    parser.add_argument("--categories", default=None)
    parser.add_argument("--limit-samples", type=int, default=10)
    parser.add_argument("--questions-per-sample", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="Minimum delay after each provider call; use 2.1s for a 30 RPM limit with concurrency=1.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful response rows and retry only missing or errored requests.",
    )
    parser.add_argument("--no-atlas-prompt", action="store_true")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


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

    provider_routing: str | JsonDict
    if args.provider == "openrouter" and provider_order:
        provider_routing = {
            "order": provider_order,
            "allow_fallbacks": args.allow_provider_fallbacks,
        }
    else:
        provider_routing = {
            "openrouter": "openrouter_default_unpinned",
            "novita": "novita_direct",
        }.get(args.provider, args.provider)

    reasoning_disabled_for_all = args.disable_reasoning or args.provider == "novita"
    reasoning_disabled_for_models = [
        model_id
        for model_id in model_specs.values()
        if reasoning_disabled_for_all or model_supports_reasoning_control(model_id)
    ]
    if reasoning_disabled_for_all:
        reasoning_mode = "disabled_for_all"
    elif args.provider == "openrouter":
        reasoning_mode = "disabled_for_known_defaults"
    else:
        reasoning_mode = "provider_default"

    plan = {
        "suite": args.suite,
        "bench_dir": str(bench_dir),
        "question_dir": str(question_dir),
        "models": model_specs,
        "question_count": len(selected),
        "categories": categories,
        "limit_samples": args.limit_samples,
        "questions_per_sample": questions_per_sample,
        "atlas_prompt": use_atlas_prompt,
        "request_settings": {
            "provider": args.provider,
            "provider_routing": provider_routing,
            "concurrency": args.concurrency,
            "temperature": args.temperature,
            "requested_max_tokens": args.max_tokens,
            "timeout_seconds": args.timeout,
            "request_interval_seconds": args.request_interval,
            "reasoning_mode": reasoning_mode,
            "reasoning_disabled_for_models": reasoning_disabled_for_models,
        },
        "output_dir": str(output_dir),
    }
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

    provider_env = {
        "moondream": "MOONDREAM_API_KEY",
        "novita": "NOVITA_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_var = provider_env[args.provider]
    api_key = os.getenv(env_var)
    if not api_key:
        raise SystemExit(
            f"{env_var} is not set. Put it in .env or export it in the shell; do not paste it into chat."
        )

    response_path = output_dir / "responses.jsonl"
    summary_path = output_dir / "summary.json"

    jobs = [
        {
            "model_key": model_key,
            "model_id": model_id,
            "qa": qa,
            "prompt": build_prompt(qa, use_atlas=use_atlas_prompt),
        }
        for model_key, model_id in model_specs.items()
        for qa in selected
    ]

    records: List[JsonDict] = []
    if args.resume and response_path.exists():
        existing = [json.loads(line) for line in response_path.read_text().splitlines() if line]
        successful = {
            (record["model_id"], record["question_id"]): record
            for record in existing
            if not record.get("error")
        }
        records = list(successful.values())
        jobs = [job for job in jobs if (job["model_id"], job["qa"]["id"]) not in successful]
        print(f"Resuming with {len(records)} successful rows; {len(jobs)} requests remain")

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
                    job["model_id"],
                    image_bytes=(bench_dir / job["qa"]["image_path"]).read_bytes(),
                    prompt=job["prompt"],
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
                qa = job["qa"]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"error": str(exc)}

                response = result.get("response", "")
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

    summary = summarize(records, plan)
    write_json(summary_path, summary)
    print("\nSummary")
    print(json.dumps(summary["models"], indent=2, sort_keys=True))
    print(f"\nResponses: {response_path}")
    print(f"Summary: {summary_path}")


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_model_specs(keys_or_ids: Sequence[str]) -> Dict[str, str]:
    specs: Dict[str, str] = {}
    for value in keys_or_ids:
        if value in SMALL_VLM_MODELS:
            specs[value] = SMALL_VLM_MODELS[value]
        else:
            specs[sanitize_model_key(value)] = value
    return specs


def sanitize_model_key(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id).strip("_")


def extract_content(value: Any) -> str:
    if isinstance(value, list):
        parts = [extract_content(part) for part in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in (
            "text",
            "content",
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "summary",
        ):
            if key in value:
                text = extract_content(value[key])
                if text:
                    return text
        candidates = [
            extract_content(v) for v in value.values() if isinstance(v, (str, list, dict))
        ]
        return "\n".join(part for part in candidates if part).strip()
    if not isinstance(value, (str, list, dict)):
        return ""
    return ""


def extract_message_text(message: JsonDict) -> str:
    """
    Qwen 3.5 may place returned text in reasoning-oriented fields.
    Check all known locations and fallback through nested payload structures.
    """
    if not isinstance(message, dict):
        return ""
    fields = (
        "content",
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "analysis",
        "summary",
    )
    for field in fields:
        text = extract_content(message.get(field))
        if text:
            return text
    return extract_content(message)


def model_supports_reasoning_control(model_id: str) -> bool:
    # Recent Qwen variants may reason by default on OpenRouter routes.
    # Disable it for short-answer visual scoring and bounded completions.
    model_id_lower = model_id.lower()
    return any(family in model_id_lower for family in ("qwen3.5", "qwen3.6", "qwen3.7", "qwen3.8"))


def select_questions(
    bench_dir: Path,
    *,
    question_dir: Path,
    categories: Sequence[str],
    limit_samples: int,
    questions_per_sample: int,
    max_requests: Optional[int],
    selection_mode: str = "sample",
) -> List[JsonDict]:
    qas = [json.loads(line) for line in (question_dir / "qa.jsonl").read_text().splitlines()]
    contracts: Dict[str, JsonDict] = {}
    selected: List[JsonDict] = []
    category_set = set(categories)
    qas_by_sample: Dict[str, List[JsonDict]] = defaultdict(list)
    sample_order: List[str] = []

    for qa in qas:
        sample_id = qa["sample_id"]
        if sample_id not in qas_by_sample:
            sample_order.append(sample_id)
        if qa["category"] in category_set:
            qas_by_sample[sample_id].append(qa)

    if selection_mode == "flat":
        for category in categories:
            category_qas = [
                qa
                for sample_id in sample_order
                for qa in qas_by_sample[sample_id]
                if qa["category"] == category
            ]
            limit = int(limit_samples)
            for qa in category_qas[:limit] if limit > 0 else category_qas:
                qa = json.loads(json.dumps(qa))
                attach_contract_if_available(qa, bench_dir, contracts)
                selected.append(qa)
                if max_requests is not None and len(selected) >= max_requests:
                    return selected
        return selected

    if selection_mode != "sample":
        raise ValueError(f"unknown question selection_mode {selection_mode!r}")

    for sample_id in sample_order[:limit_samples]:
        sample_qas = qas_by_sample[sample_id]
        by_category: Dict[str, List[JsonDict]] = defaultdict(list)
        for qa in sample_qas:
            by_category[qa["category"]].append(qa)

        per_category_index: Counter[str] = Counter()
        sample_selected = 0
        while sample_selected < questions_per_sample:
            made_progress = False
            for category in categories:
                category_qas = by_category.get(category, [])
                idx = per_category_index[category]
                if idx >= len(category_qas):
                    continue
                qa = json.loads(json.dumps(category_qas[idx]))
                per_category_index[category] += 1

                attach_contract_if_available(qa, bench_dir, contracts)
                selected.append(qa)
                sample_selected += 1
                made_progress = True
                if max_requests is not None and len(selected) >= max_requests:
                    return selected
                if sample_selected >= questions_per_sample:
                    break

            if not made_progress:
                break

    return selected


def attach_contract_if_available(
    qa: JsonDict, bench_dir: Path, contracts: Dict[str, JsonDict]
) -> None:
    contract_ref = qa.get("contract_path")
    if not contract_ref:
        return
    contract_path = resolve_contract_path(bench_dir, contract_ref)
    if contract_path is None:
        return
    contract_key = str(contract_path)
    if contract_key not in contracts:
        contracts[contract_key] = json.loads(contract_path.read_text())
    qa["contract"] = contracts[contract_key]


def resolve_contract_path(bench_dir: Path, contract_ref: str) -> Optional[Path]:
    contract_path = Path(contract_ref)
    if contract_path.is_absolute() and contract_path.exists():
        return contract_path
    bench_candidate = bench_dir / contract_path
    if bench_candidate.exists():
        return bench_candidate
    if contract_path.exists():
        return contract_path
    return None


def build_prompt(qa: JsonDict, *, use_atlas: bool) -> str:
    lines = []
    if use_atlas and qa.get("contract"):
        lines.append("Fixed atlas context:")
        lines.append(tile_layout_text(qa["contract"]))
        local = local_atlas_context(qa)
        if local:
            lines.append(local)
        lines.append("")

    sentinel = sentinel_hint(qa)
    if sentinel:
        lines.append(sentinel)

    lines.extend(
        [
            f"Question: {qa['question']}",
            "",
            "Return only the answer.",
        ]
    )
    return "\n".join(lines)


def tile_layout_text(contract: JsonDict) -> str:
    rows: Dict[int, List[JsonDict]] = defaultdict(list)
    for tile in contract["tiles"]:
        z = tile["coord"][2]
        rows[z].append(tile)
    parts = []
    for row_index, z in enumerate(sorted(rows)):
        tiles = sorted(rows[z], key=lambda t: t["coord"][0])
        parts.append(f"row {row_index} left-to-right: " + " ".join(t["token"] for t in tiles))
    return "Tile rows top-to-bottom: " + "; ".join(parts) + "."


def local_atlas_context(qa: JsonDict) -> str:
    contract = qa.get("contract")
    if not contract:
        return ""
    target = qa["target"]
    category = canonical_category(qa["category"])

    if category in {"tile_resource_number", "tile_has_robber", "tile_occupied_nodes"}:
        tile_token_value = target.get("tile_token")
        tile = find_by_token(contract["tiles"], tile_token_value)
        if tile:
            return (
                f"Local atlas: {tile['token']} touches nodes "
                f"{' '.join(tile['node_tokens'])} and edges {' '.join(tile['edge_tokens'])}."
            )

    if category == "node_occupancy":
        node = find_by_token(contract["nodes"], target.get("node_token"))
        if node:
            port_text = (
                f" ports {' '.join(node['port_tokens'])}" if node["port_tokens"] else " no ports"
            )
            return (
                f"Local atlas: {node['token']} is the intersection of tiles "
                f"{' '.join(node['adjacent_tile_tokens'])}; adjacent edges "
                f"{' '.join(node['adjacent_edge_tokens'])};{port_text}."
            )

    if category == "edge_road_owner":
        edge = find_by_token(contract["edges"], target.get("edge_token"))
        if edge:
            adjacent_tiles = []
            edge_nodes = set(edge["nodes"])
            for tile in contract["tiles"]:
                if edge_nodes.issubset(set(tile["nodes"])):
                    adjacent_tiles.append(tile["token"])
            return (
                f"Local atlas: {edge['token']} connects {' and '.join(edge['node_tokens'])} "
                f"and borders tiles {' '.join(adjacent_tiles)}."
            )

    if category in {"port_trade_type", "port_type_nodes", "port_occupancy"}:
        port = find_by_token(contract["ports"], target.get("port_token"))
        if port:
            return (
                f"Local atlas: {port['token']} is at coord {port['coord']} "
                f"with direction {port['direction']} and touches nodes "
                f"{' '.join(port['attached_node_tokens'])}."
            )

    return ""


def sentinel_hint(qa: JsonDict) -> str:
    category = canonical_category(qa["category"])
    if category in {"longest_road_holder", "largest_army_holder"}:
        return "If no player holds the award, answer exactly NONE."
    if category in {"node_occupancy", "edge_road_owner"}:
        return "If the queried location is unoccupied, answer exactly EMPTY."
    if category == "color_road_locations":
        return "If the player has no roads, answer exactly NONE. Otherwise list only edge tokens."
    if category == "tile_resource_number":
        return "If the tile is desert, answer exactly <DESERT> NO_NUMBER."
    if category == "robber_resource_number":
        return "Answer as: <Txx> <RESOURCE> NUMBER. If the robber is on desert, answer <Txx> <DESERT> NO_NUMBER."
    if category == "tile_has_robber":
        return "Answer exactly YES or NO."
    if category in {"robber_adjacent_buildings", "tile_occupied_nodes"}:
        return "If no buildings touch the tile, answer exactly NONE. Otherwise list node, color, and building tokens."
    if category == "port_trade_type":
        return "If the port is a 3:1 port, answer GENERIC 3:1. Otherwise answer the exact resource token and 2:1."
    if category == "port_type_nodes":
        return "If the port is a 3:1 port, use GENERIC. Otherwise use the exact resource token."
    if category == "port_occupancy":
        return "If no building touches the port, answer exactly NONE. Otherwise list color, building, and node tokens."
    if category in {"nodes_connected", "edge_connects_nodes"}:
        return "Answer exactly YES or NO."
    if category == "robber_presence":
        return "Answer exactly YES or NO."
    return ""


def find_by_token(items: Iterable[JsonDict], token_value: Optional[str]) -> Optional[JsonDict]:
    for item in items:
        if item.get("token") == token_value:
            return item
    return None


def call_openrouter(
    api_key: str,
    model_id: str,
    *,
    image_bytes: bytes,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    provider_order: Sequence[str] = (),
    allow_provider_fallbacks: bool = False,
    disable_reasoning: bool = False,
) -> JsonDict:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    no_system = ":free" in model_id or "gemma" in model_id.lower()
    user_text = prompt if not no_system else f"{system_prompt}\n\n---\n\n{prompt}"
    messages = []
    if not no_system:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": user_text},
            ],
        }
    )
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_reasoning or model_supports_reasoning_control(model_id):
        payload["reasoning"] = {"enabled": False}
        payload["max_tokens"] = max(max_tokens, 256)
    if provider_order:
        payload["provider"] = {
            "order": list(provider_order),
            "allow_fallbacks": allow_provider_fallbacks,
        }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/catan-learning",
        "X-Title": "CatanBoardBench OpenRouter Eval",
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
    content = extract_message_text(message)

    return {
        "response": content.strip(),
        "latency_ms": latency_ms,
        "usage": data.get("usage", {}),
        "served_model": data.get("model", model_id),
        "provider": data.get("provider"),
    }


def call_provider(
    provider: str,
    api_key: str,
    model_id: str,
    *,
    request_interval: float,
    provider_order: Sequence[str] = (),
    allow_provider_fallbacks: bool = False,
    disable_reasoning: bool = False,
    **kwargs: Any,
) -> JsonDict:
    caller = {
        "moondream": call_moondream,
        "novita": call_novita,
        "openrouter": call_openrouter,
    }[provider]
    if provider == "openrouter":
        kwargs.update(
            {
                "provider_order": provider_order,
                "allow_provider_fallbacks": allow_provider_fallbacks,
                "disable_reasoning": disable_reasoning,
            }
        )
    try:
        return caller(api_key, model_id, **kwargs)
    finally:
        if request_interval > 0:
            time.sleep(request_interval)


def call_novita(
    api_key: str,
    model_id: str,
    *,
    image_bytes: bytes,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> JsonDict:
    """Call Novita's OpenAI-compatible multimodal endpoint directly."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start = time.time()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(NOVITA_URL, headers=headers, json=payload)
    latency_ms = int((time.time() - start) * 1000)

    if response.status_code >= 400:
        return {
            "error": f"HTTP {response.status_code}: {response.text[:800]}",
            "latency_ms": latency_ms,
        }

    data = response.json()
    message = data["choices"][0]["message"]
    return {
        "response": extract_message_text(message).strip(),
        "latency_ms": latency_ms,
        "usage": data.get("usage", {}),
        "served_model": data.get("model", model_id),
        "provider": "novita",
    }


def call_moondream(
    api_key: str,
    model_id: str,
    *,
    image_bytes: bytes,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> JsonDict:
    del model_id, temperature, max_tokens
    payload = {
        "image_url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}",
        "question": prompt if not system_prompt else f"{system_prompt}\n\n---\n\n{prompt}",
    }
    start = time.time()
    proc: Optional[subprocess.CompletedProcess[str]]
    proc = None

    def run_with_optional_resolve(
        resolve_ip: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            "curl",
            "-sS",
            "-m",
            str(timeout),
            "-X",
            "POST",
            MOONDREAM_URL,
            "-H",
            "Content-Type: application/json",
            "-H",
            f"X-Moondream-Auth: {api_key}",
        ]
        if resolve_ip:
            cmd += ["--resolve", f"{MOONDREAM_HOST}:443:{resolve_ip}"]
        cmd += ["--data", json.dumps(payload)]
        return subprocess.run(cmd, capture_output=True, text=True)

    attempts = [None] + list(MOONDREAM_RESOLVE_IPS)
    proc: Optional[subprocess.CompletedProcess[str]] = None
    for resolve_ip in attempts:
        proc = run_with_optional_resolve(resolve_ip=resolve_ip)
        if proc.returncode == 0:
            break

    latency_ms = int((time.time() - start) * 1000)
    if proc.returncode != 0:
        return {
            "error": f"curl error {proc.returncode}: {proc.stderr[:800]}",
            "latency_ms": latency_ms,
        }

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "error": f"invalid JSON response: {proc.stdout[:800]}",
            "latency_ms": latency_ms,
        }

    if "error" in data:
        return {
            "error": json.dumps(data["error"])[:800],
            "latency_ms": latency_ms,
        }

    return {
        "response": (data.get("answer") or "").strip(),
        "latency_ms": latency_ms,
        "usage": data.get("metrics", {}),
        "served_model": data.get("model", "moondream"),
    }


def score_answer(qa: JsonDict, response: str) -> JsonDict:
    category = canonical_category(qa["category"])
    target = qa["target"]
    expected = qa["answer"]
    normalized = normalize_text(response)
    expected_normalized = normalize_text(expected)

    if category == "tile_resource_number":
        resource_ok = contains_value(normalized, resource_token(target))
        number = target["number"]
        number_ok = (
            ("NO_NUMBER" in normalized or "<DESERT>" in normalized)
            if number is None
            else str(number) in number_tokens(normalized)
        )
        return component_score(resource_ok, number_ok)

    if category == "robber_resource_number":
        resource_ok = contains_value(normalized, target["resource_token"])
        tile_ok = contains_value(normalized, target["tile_token"])
        number = target["number"]
        number_ok = (
            "NO_NUMBER" in normalized
            if number is None
            else str(number) in number_tokens(normalized)
        )
        return component_score(tile_ok, resource_ok, number_ok)

    if category == "tile_has_robber":
        expected_binary = "YES" if target["has_robber"] else "NO"
        return component_score(expected_binary in normalized)

    if category == "node_occupancy":
        if target["building"] is None:
            return component_score("EMPTY" in normalized)
        return component_score(
            contains_value(normalized, color_token(target)),
            contains_value(normalized, building_token(target)),
        )

    if category == "edge_road_owner":
        road_color = target.get("road_color", target.get("color"))
        if road_color is None:
            return component_score("EMPTY" in normalized)
        return component_score(
            contains_value(normalized, color_token(target, color_key="road_color"))
        )

    if category == "color_road_locations":
        expected_tokens = set(target["road_edge_tokens"])
        if not expected_tokens:
            return component_score("NONE" in normalized)
        found_tokens = edge_tokens(normalized)
        checks = [token in found_tokens for token in sorted(expected_tokens)]
        checks.append(not (found_tokens - expected_tokens))
        return component_score(*checks)

    if category == "port_type_nodes":
        if target["kind"] == "generic":
            resource_ok = "GENERIC" in normalized or "3:1" in normalized
        else:
            resource_ok = contains_value(normalized, target["resource_token"])
        ratio_ok = target["ratio"] in normalized
        node_oks = [contains_value(normalized, token) for token in target["attached_node_tokens"]]
        return component_score(resource_ok, ratio_ok, *node_oks)

    if category == "port_trade_type":
        if target.get("kind") == "generic" or target.get("resource") is None:
            resource_ok = "GENERIC" in normalized or "3:1" in normalized
        else:
            resource_ok = contains_value(normalized, resource_token(target))
        return component_score(resource_ok, target["ratio"] in normalized)

    if category == "robber_presence":
        expected_binary = "YES" if target.get("robber", target.get("has_robber")) else "NO"
        return component_score(expected_binary in normalized)

    if category == "port_occupancy":
        occupied_nodes = target["occupied_nodes"]
        if not occupied_nodes:
            return component_score("NONE" in normalized)
        checks = []
        for node in occupied_nodes:
            checks.extend(
                [
                    contains_value(normalized, node["color_token"]),
                    contains_value(normalized, node["building_token"]),
                    contains_value(normalized, node["node_token"]),
                ]
            )
        return component_score(*checks)

    if category in {"nodes_connected", "edge_connects_nodes"}:
        expected_binary = "YES" if target["connected"] else "NO"
        return component_score(expected_binary in normalized)

    if category in {"robber_adjacent_buildings", "tile_occupied_nodes"}:
        occupied_nodes = target["occupied_nodes"]
        if not occupied_nodes:
            return component_score("NONE" in normalized)
        checks = []
        for node in occupied_nodes:
            checks.extend(
                [
                    contains_value(normalized, node["node_token"]),
                    contains_value(normalized, node["color_token"]),
                    contains_value(normalized, node["building_token"]),
                ]
            )
        return component_score(*checks)

    if category == "color_building_counts":
        return component_score(
            contains_count(normalized, ("SETTLEMENT", "SETTLEMENTS"), target["settlement_count"]),
            contains_count(normalized, ("CITY", "CITIES"), target["city_count"]),
        )

    if category == "color_road_count":
        return component_score(
            contains_count(normalized, ("ROAD", "ROADS"), target["road_count"]),
        )

    if category in {"longest_road_holder", "largest_army_holder"}:
        holder_token = target.get("holder_token")
        if holder_token is None:
            return component_score("NONE" in normalized)
        return component_score(contains_value(normalized, holder_token))

    if category == "current_player":
        return component_score(contains_value(normalized, target["color_token"]))

    if category == "robber_tile":
        return component_score(contains_value(normalized, target["tile_token"]))

    if category in {
        "isolated_hex_direction_to_label",
        "isolated_hex_label_to_direction",
    }:
        return score_hex_direction_answer(qa, response)

    return component_score(expected_normalized == normalized)


def canonical_category(category: str) -> str:
    return CATEGORY_ALIASES.get(category, category)


def resource_token(target: JsonDict) -> str:
    if target.get("resource_token"):
        return target["resource_token"]
    resource = target.get("resource")
    if resource:
        return f"<{resource}>"
    return "<DESERT>"


def color_token(target: JsonDict, *, color_key: str = "color") -> Optional[str]:
    token_key = f"{color_key}_token"
    if target.get(token_key):
        return target[token_key]
    if target.get("color_token") and color_key != "color":
        return target["color_token"]
    color = target.get(color_key, target.get("color"))
    if color:
        return f"<{color}>"
    return None


def building_token(target: JsonDict) -> Optional[str]:
    if target.get("building_token"):
        return target["building_token"]
    building = target.get("building")
    if building:
        return f"<{building}>"
    return None


def component_score(*checks: bool) -> JsonDict:
    total = len(checks)
    correct_components = sum(bool(check) for check in checks)
    return {
        "correct": correct_components == total,
        "component_correct": correct_components,
        "component_total": total,
        "component_accuracy": correct_components / total if total else 0.0,
    }


def normalize_text(value: Any) -> str:
    text = str(value).upper().strip()
    replacements = {
        "NO NUMBER": "NO_NUMBER",
        "NO-NUMBER": "NO_NUMBER",
        "NO PLAYER": "NONE",
        "NO ONE": "NONE",
        "NONE.": "NONE",
        "EMPTY.": "EMPTY",
        "DESERT": "<DESERT>",
        "WOOD": "<WOOD>",
        "BRICK": "<BRICK>",
        "SHEEP": "<SHEEP>",
        "WHEAT": "<WHEAT>",
        "ORE": "<ORE>",
        "SETTLEMENT": "<SETTLEMENT>",
        "CITY": "<CITY>",
        "GEN": "GENERIC",
    }
    replacements.update(
        {
            color_name.replace("_", separator): f"<{color_name}>"
            for color_name in COLOR_TOKEN_NAMES
            if "_" in color_name
            for separator in (" ", "-")
        }
    )
    replacements.update({color_name: f"<{color_name}>" for color_name in COLOR_TOKEN_NAMES})
    for src, dst in replacements.items():
        text = re.sub(rf"(?<![A-Z0-9_<]){re.escape(src)}(?![A-Z0-9_>])", dst, text)
    text = repair_merged_tokens(text)
    text = repair_partial_tokens(text)
    return re.sub(r"\s+", " ", text)


def repair_merged_tokens(text: str) -> str:
    colors = "|".join(re.escape(color_name) for color_name in COLOR_TOKEN_NAMES)
    pieces = "SETTLEMENT|CITY|ROAD"
    return re.sub(rf"<({colors})_({pieces})>", r"<\1> <\2>", text)


def repair_partial_tokens(text: str) -> str:
    named_tokens = [
        "WOOD",
        "BRICK",
        "SHEEP",
        "WHEAT",
        "ORE",
        "DESERT",
        "RED",
        "BLUE",
        "WHITE",
        "BLACK",
        "GREEN",
        "ORANGE",
        "SETTLEMENT",
        "CITY",
        "ROAD",
        *COLOR_TOKEN_NAMES,
    ]
    for token in named_tokens:
        text = re.sub(rf"<{token}(?![A-Z0-9_]*>)", f"<{token}>", text)

    text = re.sub(
        r"<T(\d{1,2})(?![0-9_]*>)",
        lambda match: f"<T{int(match.group(1)):02d}>",
        text,
    )
    text = re.sub(
        r"<N(\d{1,2})(?![0-9_]*>)",
        lambda match: f"<N{int(match.group(1)):02d}>",
        text,
    )
    text = re.sub(
        r"\bT(\d{1,2})\b",
        lambda match: f"<T{int(match.group(1)):02d}>",
        text,
    )
    text = re.sub(
        r"\bN(\d{1,2})\b",
        lambda match: f"<N{int(match.group(1)):02d}>",
        text,
    )
    text = re.sub(
        r"<E(\d{1,2})[_-](\d{1,2})(?![0-9_]*>)",
        lambda match: _edge_token_from_match(match),
        text,
    )
    text = re.sub(
        r"\bE(\d{1,2})[_-](\d{1,2})\b",
        lambda match: _edge_token_from_match(match),
        text,
    )
    return text


def _edge_token_from_match(match: re.Match[str]) -> str:
    a = int(match.group(1))
    b = int(match.group(2))
    lo, hi = sorted((a, b))
    return f"<E{lo:02d}_{hi:02d}>"


def contains_value(normalized_response: str, expected_token: Optional[str]) -> bool:
    if expected_token is None:
        return False
    return normalize_text(expected_token) in normalized_response


def number_tokens(normalized_response: str) -> set[str]:
    return set(re.findall(r"\b(?:2|3|4|5|6|8|9|10|11|12)\b", normalized_response))


def edge_tokens(normalized_response: str) -> set[str]:
    return set(re.findall(r"<E\d{2}_\d{2}>", normalized_response))


def contains_labeled_count(
    normalized_response: str,
    labels: Sequence[str],
    expected_count: int,
) -> bool:
    label_forms = set(labels)
    label_forms.update(normalize_text(label) for label in labels)
    count = str(expected_count)
    for label in label_forms:
        escaped = re.escape(label)
        if re.search(rf"{escaped}\D{{0,20}}\b{count}\b", normalized_response):
            return True
        if re.search(rf"\b{count}\b\D{{0,20}}{escaped}", normalized_response):
            return True
    return False


def contains_count(
    normalized_response: str,
    labels: Sequence[str],
    expected_count: int,
) -> bool:
    return contains_labeled_count(
        normalized_response, labels, expected_count
    ) or contains_bare_count(
        normalized_response,
        expected_count,
    )


def contains_bare_count(normalized_response: str, expected_count: int) -> bool:
    numbers = re.findall(r"\b\d+\b", normalized_response)
    return len(numbers) == 1 and numbers[0] == str(expected_count)


def summarize(records: Sequence[JsonDict], plan: JsonDict) -> JsonDict:
    by_model: Dict[str, List[JsonDict]] = defaultdict(list)
    by_model_category: Dict[Tuple[str, str], List[JsonDict]] = defaultdict(list)
    for record in records:
        by_model[record["model_key"]].append(record)
        by_model_category[(record["model_key"], record["category"])].append(record)

    model_summary = {}
    for model_key, model_records in by_model.items():
        model_summary[model_key] = summarize_records(model_records)
        model_summary[model_key]["categories"] = {
            category: summarize_records(category_records)
            for (key, category), category_records in by_model_category.items()
            if key == model_key
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "models": model_summary,
    }


def summarize_records(records: Sequence[JsonDict]) -> JsonDict:
    attempted = [record for record in records if not record.get("error")]
    errors = len(records) - len(attempted)
    exact = sum(1 for record in attempted if record["score"]["correct"])
    component_correct = sum(record["score"]["component_correct"] for record in attempted)
    component_total = sum(record["score"]["component_total"] for record in attempted)
    latencies = [
        record["latency_ms"] for record in attempted if record.get("latency_ms") is not None
    ]
    return {
        "requests": len(records),
        "attempted": len(attempted),
        "errors": errors,
        "exact_accuracy": exact / len(attempted) if attempted else 0.0,
        "component_accuracy": component_correct / component_total if component_total else 0.0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
    }


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, value: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
