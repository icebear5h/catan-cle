"""OpenBench plugin metadata for CatanBoardBench."""

from __future__ import annotations

import importlib


def get_benchmark_metadata():
    benchmark_metadata = importlib.import_module("openbench.utils").BenchmarkMetadata

    return benchmark_metadata(
        name="CatanBoardBench",
        description=(
            "Engine-scored Catan public-board perception and grounded reasoning "
            "across image and text representations."
        ),
        category="multimodal",
        tags=[
            "catan",
            "vlm",
            "board-state",
            "visual-grounding",
            "spatial-reasoning",
        ],
        module_path="evals.catan_board_bench.benchmark",
        function_name="catan_board_bench",
    )
