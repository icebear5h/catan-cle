"""Evaluate the frozen strict 60-question text projection through Novita.

Constants, transport, the plan, records, and the CLI live in sibling modules.
Every pre-split name stays importable at this path.

``httpx`` is re-exported because tests patch ``<this module>.httpx.Client``."""

from __future__ import annotations

import httpx

from scripts.board_bench.run.eval_catan_strict_text_probe_novita.cli import (
    main,
    parse_args,
    run_evaluation,
)
from scripts.board_bench.run.eval_catan_strict_text_probe_novita.constants import (
    DEFAULT_DATASET_DIR,
    EVAL_SCHEMA,
    FORMAT_NAME,
    SUITE_NAME,
    SYSTEM_PROMPT,
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
from scripts.board_bench.run.eval_catan_strict_text_probe_novita.transport import (
    call_job,
    call_novita_text,
)

__all__ = [
    "DEFAULT_DATASET_DIR",
    "EVAL_SCHEMA",
    "FORMAT_NAME",
    "SUITE_NAME",
    "SYSTEM_PROMPT",
    "accepted_records",
    "admissible_record",
    "build_plan",
    "call_job",
    "call_novita_text",
    "configured_text_evaluator",
    "httpx",
    "main",
    "parse_args",
    "recompute_record_score",
    "response_record",
    "run_evaluation",
    "select_questions",
    "split_csv",
    "summarize",
    "validate_cli_args",
    "validate_or_write_plan",
]
