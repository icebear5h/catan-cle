"""Replay pipeline domain package (compatibility boundary).

This package currently forwards imports to `data_pipeline.bootstrapping` while the
actual domain extraction/replay code remains in place.
"""

from __future__ import annotations

from importlib import import_module
import sys


def _alias(module_name: str) -> None:
    sys.modules[f"{__name__}.{module_name}"] = import_module(f"data_pipeline.bootstrapping.{module_name}")


_alias("db")
_alias("replay_decoder")
_alias("generate_training_data")
_alias("scrapers")
_alias("scrapers.colonist_api")
_alias("scrapers.replay_api_scraper")
_alias("scrapers.replay_playwright_scraper")
_alias("scrapers.scrape_top_players")
_alias("scrapers.colonist_schema")
_alias("colonist")
_alias("colonist.board_layout")

__all__ = []
