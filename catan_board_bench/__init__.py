"""CatanBoardBench domain package (compatibility boundary).

This package is the external entrypoint for CatanBoardBench code that was previously
hosted under `data_pipeline.catan_board_bench`. Imports stay on stable top-level names
while code currently remains in the existing package until a later full move is
performed.
"""

from __future__ import annotations

from importlib import import_module
import sys


def _alias(module_name: str) -> None:
    sys.modules[f"{__name__}.{module_name}"] = import_module(
        f"data_pipeline.catan_board_bench.{module_name}"
    )


_alias("annotations")
_alias("builder")
_alias("hex_direction_probe")
_alias("render")
_alias("scoring")
_alias("text_representations")
_alias("tokens")
_alias("eval")
_alias("eval.benchmark")
_alias("eval.metadata")

__all__ = []
