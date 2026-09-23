"""Schema constants shared by the Qwen SFT exporter and its validator."""

from __future__ import annotations

from data_pipeline.json_types import JsonDict

SFT_EXPORT_SCHEMA = "catan_board_recognition_qwen_sft/v1"
SFT_ROW_KEYS = {"image", "conversations"}
SPLITS = ("train", "validation", "test")

__all__ = ["SFT_EXPORT_SCHEMA", "SFT_ROW_KEYS", "SPLITS", "JsonDict"]
