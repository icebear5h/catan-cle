"""Evaluate hosted VLMs on the strict 60-question raw-image cohort.

Constants, dataset validation, prompts, jobs, records, summaries, execution,
and the CLI live in sibling modules. Every pre-split name stays importable at
this path."""

from __future__ import annotations

from scripts.board_bench.run.eval_catan_strict_vision_probe.cli import (
    main,
    parse_args,
    validate_cli_args,
)
from scripts.board_bench.run.eval_catan_strict_vision_probe.constants import (
    DATASET_SCHEMA,
    DEFAULT_DATASET_DIR,
    EVAL_SCHEMA,
    SUITE_NAME,
    SYSTEM_PROMPT,
    VisionJob,
    acquire_run_lock,
    percentile,
    release_run_lock,
    split_csv,
    usage_reasoning_tokens,
)
from scripts.board_bench.run.eval_catan_strict_vision_probe.dataset import (
    select_questions,
    validate_dataset,
)
from scripts.board_bench.run.eval_catan_strict_vision_probe.execute import (
    call_job,
    call_job_in_child,
    call_job_with_hard_deadline,
    persist_result,
    run_evaluation,
)
from scripts.board_bench.run.eval_catan_strict_vision_probe.jobs import (
    build_jobs,
    build_plan,
    validate_or_write_plan,
)
from scripts.board_bench.run.eval_catan_strict_vision_probe.prompts import (
    build_prompt,
    canonical_atlas_context,
    edge_anchor_text,
    node_anchor_text,
    tile_rows_text,
)
from scripts.board_bench.run.eval_catan_strict_vision_probe.records import (
    accepted_records,
    admissible_record,
    recompute_record_score,
    response_record,
)
from scripts.board_bench.run.eval_catan_strict_vision_probe.summary import (
    summarize,
    summarize_group,
)

__all__ = [
    "DATASET_SCHEMA",
    "DEFAULT_DATASET_DIR",
    "EVAL_SCHEMA",
    "SUITE_NAME",
    "SYSTEM_PROMPT",
    "VisionJob",
    "accepted_records",
    "acquire_run_lock",
    "admissible_record",
    "build_jobs",
    "build_plan",
    "build_prompt",
    "call_job",
    "call_job_in_child",
    "call_job_with_hard_deadline",
    "canonical_atlas_context",
    "edge_anchor_text",
    "main",
    "node_anchor_text",
    "parse_args",
    "percentile",
    "persist_result",
    "recompute_record_score",
    "release_run_lock",
    "response_record",
    "run_evaluation",
    "select_questions",
    "split_csv",
    "summarize",
    "summarize_group",
    "tile_rows_text",
    "usage_reasoning_tokens",
    "validate_cli_args",
    "validate_dataset",
    "validate_or_write_plan",
]
