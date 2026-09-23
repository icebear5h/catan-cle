"""Evaluate six lossless full-graph Catan text formats through OpenRouter.

Dataset config, constants, validation, jobs, records, transport, summaries,
and the CLI live in sibling modules. Every pre-split name stays importable at
this path.

The seven retargetable dataset names live in ``dataset_config`` and are
forwarded here by ``__getattr__`` so a reader always sees the current value.
Retarget the suite by patching ``dataset_config``, never this package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats import dataset_config
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.cli import (
    main,
    parse_args,
    run_evaluation,
)
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.constants import (
    OPENROUTER_URL,
    SYSTEM_PROMPT,
    FormatJob,
    acquire_run_lock,
    extract_message_text,
    job_key,
    model_supports_reasoning_control,
    percentile,
    read_jsonl,
    release_run_lock,
    split_csv,
    usage_reasoning_tokens,
    write_json,
)
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.dataset import (
    validate_dataset_integrity,
    validate_dataset_metadata,
    validate_request_contract,
)
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.jobs import (
    build_jobs,
    build_plan,
    build_prompt,
    validate_or_write_plan,
)
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.records import (
    accepted_records,
    admissible_record,
    recompute_record_score,
    response_record,
)
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.summary import (
    summarize,
    summarize_group,
)
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.transport import (
    call_openrouter_text,
)

_FORWARDED = frozenset(dataset_config.__all__)

if TYPE_CHECKING:
    from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.dataset_config import (
        DATASET_SCHEMA as DATASET_SCHEMA,
    )
    from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.dataset_config import (
        DEFAULT_DATASET_DIR as DEFAULT_DATASET_DIR,
    )
    from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.dataset_config import (
        EVAL_SCHEMA as EVAL_SCHEMA,
    )
    from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.dataset_config import (
        FORMAT_EXTENSIONS as FORMAT_EXTENSIONS,
    )
    from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.dataset_config import (
        FORMAT_NAMES as FORMAT_NAMES,
    )
    from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.dataset_config import (
        SUITE_NAME as SUITE_NAME,
    )
    from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats.dataset_config import (
        parse_full_graph_format as parse_full_graph_format,
    )


def __getattr__(name: str) -> object:
    """Read a retargetable dataset name from ``dataset_config`` at call time."""
    if name in _FORWARDED:
        return getattr(dataset_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DATASET_SCHEMA",
    "DEFAULT_DATASET_DIR",
    "EVAL_SCHEMA",
    "FORMAT_EXTENSIONS",
    "FORMAT_NAMES",
    "OPENROUTER_URL",
    "SUITE_NAME",
    "SYSTEM_PROMPT",
    "FormatJob",
    "accepted_records",
    "acquire_run_lock",
    "admissible_record",
    "build_jobs",
    "build_plan",
    "build_prompt",
    "call_openrouter_text",
    "dataset_config",
    "extract_message_text",
    "job_key",
    "main",
    "model_supports_reasoning_control",
    "parse_args",
    "parse_full_graph_format",
    "percentile",
    "read_jsonl",
    "recompute_record_score",
    "release_run_lock",
    "response_record",
    "run_evaluation",
    "split_csv",
    "summarize",
    "summarize_group",
    "usage_reasoning_tokens",
    "validate_dataset_integrity",
    "validate_dataset_metadata",
    "validate_or_write_plan",
    "validate_request_contract",
    "write_json",
]
