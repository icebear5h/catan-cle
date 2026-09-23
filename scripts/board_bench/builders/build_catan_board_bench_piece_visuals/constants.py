"""Paths and the visual-primitive vocabularies."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts" / "generated" / "catan_board_bench" / "piece_recognition"
)
DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "artifacts"
    / "fixtures"
    / "sft"
    / "render_contracts"
    / "colonist_dummy_setup.json"
)
DEFAULT_PROMPT_PREFIX = "Answer exactly using Catan visual tokens. Do not explain."

RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
NUMBERS = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
COLORS = ["RED", "BLUE", "ORANGE", "WHITE", "BLACK"]
PORTS = [None, "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]

__all__ = [
    "COLORS",
    "DEFAULT_CONTRACT",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PROMPT_PREFIX",
    "NUMBERS",
    "PORTS",
    "PROJECT_ROOT",
    "RESOURCES",
]
