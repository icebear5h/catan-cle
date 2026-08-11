"""OpenBench plugin metadata for CatanBench."""

from __future__ import annotations


def get_benchmark_metadata():
    from openbench.utils import BenchmarkMetadata

    return BenchmarkMetadata(
        name="CatanBench",
        description="Engine-scored multimodal Catan public board QA over fixed board screenshots.",
        category="multimodal",
        tags=["catan", "vlm", "board-state", "symbolic-reasoning"],
        module_path="catanbench.eval.benchmark",
        function_name="catanbench",
    )
