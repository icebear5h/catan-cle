"""Probe schemas, format names, default locations, and the dataset README."""

from __future__ import annotations

from evals.catan_board_bench.paths import RELATIVE_DATASETS_DIR
from evals.json_types import JsonDict as JsonDict

DATASET_SCHEMA = "catan_text_format_optimization_probe/v1"
QUERY_INDEX_SCHEMA = "catan_query_index/v1"
EVAL_SCHEMA = "catan_text_format_optimization_eval/v1"
SUITE_NAME = "text_format_optimization_probe"
DEFAULT_SOURCE_DIR = RELATIVE_DATASETS_DIR / "ascii_variation_probe"
DEFAULT_OUTPUT_DIR = RELATIVE_DATASETS_DIR / "text_format_optimization_probe"
FORMAT_NAMES = (
    "tile_rows",
    "indexed_records",
    "indexed_tile_rows",
    "indexed_json",
)
FORMAT_EXTENSIONS = {
    "tile_rows": ".txt",
    "indexed_records": ".txt",
    "indexed_tile_rows": ".txt",
    "indexed_json": ".json",
}

_README = """# Catan Text-Format Optimization Probe

A strict comparison between the incumbent `tile_rows` representation and three
query-indexed, lossless projections of the same complete public graph.

The derived indexes expose named tile neighbors, tile-corner state, port-endpoint
state, roll-to-tile/source lookup, and player-owned entity-ID lists. They do not
expose player entity counts, aggregated roll payouts, or any question-specific
answer.
"""

