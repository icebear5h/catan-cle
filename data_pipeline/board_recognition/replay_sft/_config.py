"""Schema constants and the exporter's error type."""


from __future__ import annotations

from data_pipeline.json_types import JsonDict

EXPORT_SCHEMA = "catan_board_recognition_qwen_sft/v2"


QUERIES_PER_STATE = 8


SFT_ROW_KEYS = {"image", "conversations"}


class ReplaySftExportError(RuntimeError):
    """Raised when compact replay_v1 projection invariants fail."""


__all__ = ["EXPORT_SCHEMA", "JsonDict", "QUERIES_PER_STATE", "ReplaySftExportError", "SFT_ROW_KEYS"]
