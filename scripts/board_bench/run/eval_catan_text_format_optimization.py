#!/usr/bin/env python
"""Evaluate query-indexed Catan text formats through OpenRouter."""

from __future__ import annotations

from scripts.board_bench.run import eval_catan_board_bench_full_graph_formats as evaluator
from scripts.board_bench.run.eval_catan_strict_text_probe_novita.constants import (
    configured_text_evaluator,
)


def main() -> None:
    # The evaluator reads its retargetable names from ``dataset_config`` at call
    # time, so the suite is swapped by patching that module, not this package.
    with configured_text_evaluator():
        evaluator.main()


if __name__ == "__main__":
    main()
