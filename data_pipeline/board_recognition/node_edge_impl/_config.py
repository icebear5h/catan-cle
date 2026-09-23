"""Schemas, families, prompts, and the empty-negative mix."""

from __future__ import annotations

from data_pipeline.json_types import JsonDict

__all__ = ["JsonDict"]


EXPORT_SCHEMA = "catan_node_edge_readout/v1"
ROW_SCHEMA = "catan_node_edge_readout_row/v1"
DEFAULT_OUTPUT_NAME = "node_edge_readout_v1"
GROUNDING_STAGE = "node_edge_readout"
TASK_FAMILY = "node_edge_readout"
SPLITS = ("train", "validation", "test", "color_diagnostic")
EVAL_SPLITS = ("validation", "test", "color_diagnostic")
# Coverage per image: "capped" samples rows_per_family occupied and empties (train);
# "balanced" keeps every occupied location and as many empties, hardest first
# (eval); "full" writes every location, seven in ten of them empty (pools and
# diagnostics only: exact accuracy on a full split rewards answering empty).
COVERAGE_MODES = ("capped", "balanced", "full")
FAMILIES = ("node", "edge")
CATEGORY = {"node": "node.occupancy", "edge": "edge.owner"}
TASK_TYPE = {"node": "node_occupancy", "edge": "edge_owner"}
READOUT_CATEGORY = {"node": "node.readout", "edge": "edge.readout"}
READOUT_TASK_TYPE = {"node": "node_readout", "edge": "edge_readout"}
NODE_COUNT = 54
EDGE_COUNT = 72
NODE_READOUT_PROMPT = (
    "List every node <N00> to <N53> as \"token building\", where building is \"empty\", "
    "\"colour settlement\" or \"colour city\", in token order, separated by \"; \"."
)
EDGE_READOUT_PROMPT = (
    "List every edge <E00_01> to <E52_53> as \"token road\", where road is \"empty\" or "
    "\"colour road\", in token order, separated by \"; \"."
)
READOUT_PROMPT = {"node": NODE_READOUT_PROMPT, "edge": EDGE_READOUT_PROMPT}
ROWS_PER_FAMILY = 4
READOUTS_PER_FAMILY = 1
# Empty locations are drawn by kind in these shares (largest remainder), so
# the confusable kinds dominate without the far ones vanishing; leftovers fill
# from the ranked pool in kind order when a kind runs short.
EMPTY_KINDS = ("adjacent", "cross_type", "hop2", "hop3", "far")
EMPTY_SHARES = {"adjacent": 0.5, "cross_type": 0.15, "hop2": 0.1, "hop3": 0.05, "far": 0.2}
EMPTY_ANSWER = "empty"
