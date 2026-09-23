"""Report the geometry of the trainable atlas-token rows in a PEFT adapter.

Training gates use this to catch entangled rows (near-duplicate directions, "twins") and
rows that point away from their own family (node, edge, tile, port) or away from the shared
mean direction. Both the input embedding rows and the lm_head rows are inspected. Every
cosine is computed over L2-normalized float32 rows; a family centroid is the normalized mean
of that family's raw rows, and the mean direction is the normalized mean of all rows.
"""

from __future__ import annotations

from ._base import (
    DEFAULT_FAMILY_FLOOR,
    DEFAULT_TOKEN_INVENTORY,
    DEFAULT_TWIN_THRESHOLD,
    FAMILIES,
    FAMILY_PAIRS,
    FOCUS_TOKEN,
    INPUT_ROWS_KEY,
    OUTPUT_ROWS_KEY,
    SIDE_KEYS,
    TABLE_FLOOR_PREVIEW,
    JsonDict,
)
from ._cli import build_parser, main, parse_adapter_spec
from ._report import build_report, format_table, inspect_adapter
from ._rows import inspect_rows, load_token_inventory, load_token_rows, token_family

__all__ = [
    "DEFAULT_FAMILY_FLOOR",
    "DEFAULT_TOKEN_INVENTORY",
    "DEFAULT_TWIN_THRESHOLD",
    "FAMILIES",
    "FAMILY_PAIRS",
    "FOCUS_TOKEN",
    "INPUT_ROWS_KEY",
    "JsonDict",
    "OUTPUT_ROWS_KEY",
    "SIDE_KEYS",
    "TABLE_FLOOR_PREVIEW",
    "build_parser",
    "build_report",
    "format_table",
    "inspect_adapter",
    "inspect_rows",
    "load_token_inventory",
    "load_token_rows",
    "main",
    "parse_adapter_spec",
    "token_family",
]
