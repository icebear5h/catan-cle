"""
Bootstrapping Module - Generate training data from expert Colonist.io replays.

Main components:
- generate_training_data: Convert replays to observation-action pairs
- db: Supabase storage for replays and training data
- scrapers/: Scraping tools for Colonist.io

Usage:
    # Scrape replays
    COLONIST_JWT="<token>" python -m data_pipeline.bootstrapping.scrapers.replay_api_scraper --max-games 100

    # Generate training data
    python -m data_pipeline.bootstrapping.generate_training_data --input-dir data/raw_replays

    # With Supabase
    python -m data_pipeline.bootstrapping.generate_training_data --supabase --no-file
"""

from .generate_training_data import process_replay, process_all_replays, TrainingExample
from .db import (
    save_replay,
    get_replay,
    save_training_examples,
    get_training_examples,
    is_configured,
)

__all__ = [
    "process_replay",
    "process_all_replays",
    "TrainingExample",
    "save_replay",
    "get_replay",
    "save_training_examples",
    "get_training_examples",
    "is_configured",
]
