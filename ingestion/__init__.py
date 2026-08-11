"""External source ingestion domain package (compatibility boundary).

This package currently forwards imports to `data_pipeline.ingestion` while the
actual implementation remains in the original location.
"""

from __future__ import annotations

from importlib import import_module
import sys


sys.modules[f"{__name__}.youtube_scraper"] = import_module("data_pipeline.ingestion.youtube_scraper")

__all__ = []
