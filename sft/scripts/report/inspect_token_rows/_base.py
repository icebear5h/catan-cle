"""Keys, defaults and thresholds shared by the token-row report."""

from __future__ import annotations

from pathlib import Path

from sft.json_types import JsonDict as JsonDict

INPUT_ROWS_KEY = (
    "base_model.model.model.language_model.embed_tokens.token_adapter.trainable_tokens_delta"
)

OUTPUT_ROWS_KEY = "base_model.model.lm_head.token_adapter.trainable_tokens_delta"

SIDE_KEYS = {"input": INPUT_ROWS_KEY, "output": OUTPUT_ROWS_KEY}

DEFAULT_TOKEN_INVENTORY = Path(
    "artifacts/generated/board_recognition/replay_v1",
    "ms_swift_bidirectional_v1/trainable_tokens.json",
)

FAMILIES = ("N", "E", "T", "P")

FAMILY_PAIRS = ("NE", "NT", "NP", "ET", "EP", "TP")

FOCUS_TOKEN = "<T10>"

DEFAULT_TWIN_THRESHOLD = 0.25

DEFAULT_FAMILY_FLOOR = 0.05

TABLE_FLOOR_PREVIEW = 8
