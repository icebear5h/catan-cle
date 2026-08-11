"""Training pipeline domain package (compatibility boundary).

This package currently forwards imports to `data_pipeline.training` while the
actual implementation stays in place.
"""

from __future__ import annotations

from importlib import import_module
import sys


def _alias(module_name: str) -> None:
    sys.modules[f"{__name__}.{module_name}"] = import_module(f"data_pipeline.training.{module_name}")


_alias("schema")
_alias("pretraining")
_alias("pretraining.build_corpus")
_alias("pretraining.sources")
_alias("pretraining.sources.forums")
_alias("pretraining.sources.youtube")

__all__ = []
