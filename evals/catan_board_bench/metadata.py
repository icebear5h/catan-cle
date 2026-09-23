"""OpenBench plugin metadata for CatanBoardBench."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openbench.utils import BenchmarkMetadata


def get_benchmark_metadata() -> BenchmarkMetadata:
    benchmark_metadata: type[BenchmarkMetadata] = importlib.import_module("openbench.utils").BenchmarkMetadata

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
