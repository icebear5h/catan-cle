"""Shared helpers for board recognition eval suite build, cli, and validation contracts."""
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5] / "artifacts/generated/board_recognition/replay_v1"


CORRECTED_SOURCE = "spatial_robber_choice_order_v1"

JsonRow = dict[str, Any]
