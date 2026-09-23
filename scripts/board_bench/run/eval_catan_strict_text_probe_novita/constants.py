"""Suite identity, the retargeting context manager, and selection helpers."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence

from dotenv import load_dotenv

from evals.catan_board_bench import text_format_optimization as optimization
from scripts.board_bench.run import eval_catan_board_bench_full_graph_formats as text_evaluator
from scripts.board_bench.shapes import JsonDict, text

load_dotenv()

EVAL_SCHEMA = "catan_strict_text_novita_eval/v1"
SUITE_NAME = "strict_text_probe_60"
FORMAT_NAME = "indexed_tile_rows"
SYSTEM_PROMPT = text_evaluator.SYSTEM_PROMPT
DEFAULT_DATASET_DIR = optimization.DEFAULT_OUTPUT_DIR

__all__ = [
    "DEFAULT_DATASET_DIR",
    "EVAL_SCHEMA",
    "FORMAT_NAME",
    "SUITE_NAME",
    "SYSTEM_PROMPT",
    "configured_text_evaluator",
    "select_questions",
    "split_csv",
    "validate_cli_args",
]


@contextlib.contextmanager
def configured_text_evaluator() -> Iterator[None]:
    """Retarget the full-graph evaluator at the text-format-optimization dataset.

    The evaluator reads these names from its ``dataset_config`` module at call
    time, so patch that module rather than the package."""
    overrides: dict[str, object] = {
        "DATASET_SCHEMA": optimization.DATASET_SCHEMA,
        "DEFAULT_DATASET_DIR": optimization.DEFAULT_OUTPUT_DIR,
        "FORMAT_NAMES": optimization.FORMAT_NAMES,
        "FORMAT_EXTENSIONS": optimization.FORMAT_EXTENSIONS,
        "parse_full_graph_format": optimization.parse_text_format,
        "EVAL_SCHEMA": optimization.EVAL_SCHEMA,
        "SUITE_NAME": optimization.SUITE_NAME,
    }
    config = text_evaluator.dataset_config
    previous = {name: getattr(config, name) for name in overrides}
    for name, value in overrides.items():
        setattr(config, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(config, name, value)


def validate_cli_args(max_questions: int | None, concurrency: int, max_tokens: int,
                      timeout: float, request_interval: float) -> None:
    if max_questions is not None and max_questions <= 0:
        raise SystemExit("--max-questions must be positive")
    if concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if max_tokens <= 0 or timeout <= 0:
        raise SystemExit("--max-tokens and --timeout must be positive")
    if request_interval < 0:
        raise SystemExit("--request-interval must be non-negative")


def select_questions(
    questions: Sequence[JsonDict],
    *,
    categories: Sequence[str],
    max_questions: int | None,
) -> list[JsonDict]:
    selected = list(questions)
    if categories:
        category_set = set(categories)
        unknown = category_set - {text(row["category"], "category") for row in questions}
        if unknown:
            raise ValueError(f"unknown categories: {sorted(unknown)}")
        selected = [row for row in selected if row["category"] in category_set]
    if max_questions is not None:
        selected = selected[:max_questions]
    if not selected:
        raise ValueError("no questions selected")
    return selected


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
