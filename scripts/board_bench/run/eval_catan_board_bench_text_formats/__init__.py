"""Evaluate information-equivalent Catan board text representations.

Constants, transport, summaries, the request loop, and the CLI live in sibling
modules. Every pre-split name stays importable at this path."""

from __future__ import annotations

from scripts.board_bench.run.eval_catan_board_bench_text_formats.cli import main, parse_args
from scripts.board_bench.run.eval_catan_board_bench_text_formats.constants import (
    BENCH_DIR,
    OPENROUTER_URL,
    QUESTION_DIR,
    SYSTEM_PROMPT,
    build_prompt,
    model_supports_reasoning_control,
    split_csv,
)
from scripts.board_bench.run.eval_catan_board_bench_text_formats.run import (
    RenderedBoards,
    build_plan,
    render_boards,
    run_requests,
    select_text_questions,
)
from scripts.board_bench.run.eval_catan_board_bench_text_formats.summary import (
    summarize,
    summarize_group,
)
from scripts.board_bench.run.eval_catan_board_bench_text_formats.transport import (
    call_openrouter_text,
    extract_content,
    extract_message_text,
)
from scripts.board_bench.shapes import JsonDict, write_json

__all__ = [
    "BENCH_DIR",
    "OPENROUTER_URL",
    "QUESTION_DIR",
    "SYSTEM_PROMPT",
    "JsonDict",
    "RenderedBoards",
    "build_plan",
    "build_prompt",
    "call_openrouter_text",
    "extract_content",
    "extract_message_text",
    "main",
    "model_supports_reasoning_control",
    "parse_args",
    "render_boards",
    "run_requests",
    "select_text_questions",
    "split_csv",
    "summarize",
    "summarize_group",
    "write_json",
]
