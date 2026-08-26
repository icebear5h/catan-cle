#!/usr/bin/env python
"""Evaluate query-indexed Catan text formats through OpenRouter."""

from __future__ import annotations

from data_pipeline.catan_board_bench import text_format_optimization as optimization
from scripts import eval_catan_board_bench_full_graph_formats as evaluator


def main() -> None:
    evaluator.DATASET_SCHEMA = optimization.DATASET_SCHEMA
    evaluator.DEFAULT_DATASET_DIR = optimization.DEFAULT_OUTPUT_DIR
    evaluator.FORMAT_NAMES = optimization.FORMAT_NAMES
    evaluator.FORMAT_EXTENSIONS = optimization.FORMAT_EXTENSIONS
    evaluator.parse_full_graph_format = optimization.parse_text_format
    evaluator.EVAL_SCHEMA = optimization.EVAL_SCHEMA
    evaluator.SUITE_NAME = optimization.SUITE_NAME
    evaluator.main()


if __name__ == "__main__":
    main()
