"""The dataset-shaped names this suite is retargeted against.

``scripts.board_bench.run.eval_catan_strict_text_probe_novita`` reuses this
evaluator against the text-format-optimization dataset by swapping these seven
module attributes. Every reader therefore resolves them through this module at
call time; do not ``from ... import`` them into another module's namespace."""

from __future__ import annotations

from evals.catan_board_bench.full_graph_format_probe import (
    DATASET_SCHEMA,
)
from evals.catan_board_bench.full_graph_format_probe import (
    DEFAULT_OUTPUT_DIR as DEFAULT_DATASET_DIR,
)
from evals.catan_board_bench.full_graph_formats import (
    FORMAT_EXTENSIONS,
    FORMAT_NAMES,
    parse_full_graph_format,
)

EVAL_SCHEMA = "catan_full_graph_format_eval/v1"
SUITE_NAME = "full_graph_format_probe"

__all__ = [
    "DATASET_SCHEMA",
    "DEFAULT_DATASET_DIR",
    "EVAL_SCHEMA",
    "FORMAT_EXTENSIONS",
    "FORMAT_NAMES",
    "SUITE_NAME",
    "parse_full_graph_format",
]
