"""Paths, logging, and Colonist enum mappings for the direct replay API scraper."""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("data_pipeline.bootstrapping.scrapers.replay_api_scraper")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INDEX_FILE = (
    PROJECT_ROOT / "artifacts" / "raw" / "colonist" / "indexes" / "4p_games_top100.json"
)
DEFAULT_STAGING_DIR = PROJECT_ROOT / "artifacts" / "staging" / "colonist" / "replays"


# Resource enum mapping (from Colonist)
RESOURCE_ENUM = {
    1: "wood",
    2: "brick",
    3: "sheep",
    4: "wheat",
    5: "ore",
}

# Building type mapping
BUILDING_TYPE = {
    1: "settlement",
    2: "city",
}

# Edge type mapping
EDGE_TYPE = {
    1: "road",
    2: "ship",
}

# Action state mapping
ACTION_STATE = {
    0: "roll_dice",
    1: "place_settlement",
    2: "place_city",
    3: "place_road",
    4: "play_turn",
    5: "discard",
    6: "move_robber",
    7: "steal",
}

# Game log type mapping (partial)
LOG_TYPE = {
    4: "place_piece",
    5: "build_piece",
    10: "roll_dice",
    44: "end_turn",
    47: "receive_resources",
    118: "trade_offer",
}

__all__ = [
    "ACTION_STATE",
    "BUILDING_TYPE",
    "DEFAULT_INDEX_FILE",
    "DEFAULT_STAGING_DIR",
    "EDGE_TYPE",
    "LOG_TYPE",
    "PROJECT_ROOT",
    "RESOURCE_ENUM",
    "logger",
]
