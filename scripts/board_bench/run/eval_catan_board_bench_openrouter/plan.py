"""Argument parsing and the immutable run plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.board_bench.run.eval_catan_board_bench_openrouter.constants import (
    DEFAULT_MODELS,
)
from scripts.board_bench.run.eval_catan_board_bench_openrouter.text_norm import (
    model_supports_reasoning_control,
)
from scripts.board_bench.shapes import JsonDict

# The pre-split module path stays the advertised program name and description.
PROG = "eval_catan_board_bench_openrouter.py"
DESCRIPTION = """Evaluate OpenRouter VLMs on CatanBoardBench engine-scored QA.

This is intended for cheap model screening before SFT. The generated benchmark
uses fixed engine atlas tokens like <T07>, <N18>, and <E03_17>, so this runner
can add local atlas context to make raw public VLMs comparable.
"""

__all__ = ["DESCRIPTION", "PROG", "build_plan", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
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


def _provider_routing(args: argparse.Namespace, provider_order: list[str]) -> JsonDict | str:
    if args.provider == "openrouter" and provider_order:
        return {
            "order": list(provider_order),
            "allow_fallbacks": args.allow_provider_fallbacks,
        }
    fallback: dict[str, str] = {
        "openrouter": "openrouter_default_unpinned",
        "novita": "novita_direct",
    }
    return fallback.get(args.provider, str(args.provider))


def build_plan(
    args: argparse.Namespace,
    *,
    bench_dir: Path,
    question_dir: Path,
    model_specs: dict[str, str],
    selected: list[JsonDict],
    categories: list[str],
    questions_per_sample: int,
    use_atlas_prompt: bool,
    provider_order: list[str],
    output_dir: Path,
) -> JsonDict:
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

    return {
        "suite": args.suite,
        "bench_dir": str(bench_dir),
        "question_dir": str(question_dir),
        "models": {key: value for key, value in model_specs.items()},
        "question_count": len(selected),
        "categories": list(categories),
        "limit_samples": args.limit_samples,
        "questions_per_sample": questions_per_sample,
        "atlas_prompt": use_atlas_prompt,
        "request_settings": {
            "provider": args.provider,
            "provider_routing": _provider_routing(args, provider_order),
            "concurrency": args.concurrency,
            "temperature": args.temperature,
            "requested_max_tokens": args.max_tokens,
            "timeout_seconds": args.timeout,
            "request_interval_seconds": args.request_interval,
            "reasoning_mode": reasoning_mode,
            "reasoning_disabled_for_models": list(reasoning_disabled_for_models),
        },
        "output_dir": str(output_dir),
    }


