"""Compact ordered JSON representation of the minimal public graph."""

from __future__ import annotations

import json
from collections.abc import Sequence

from evals.catan_board_bench.ascii_variations.facts import FullFacts
from evals.catan_board_bench.full_graph_formats.graph import (
    expand_minimal_graph,
    minimal_graph_facts,
)


def render_full_graph_json(facts: FullFacts) -> str:
    return json.dumps(
        minimal_graph_facts(facts),
        separators=(",", ":"),
        sort_keys=False,
    )


def parse_full_graph_json(text: str) -> FullFacts:
    payload = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(payload, dict):
        raise ValueError("full-graph JSON must be one object")
    return expand_minimal_graph(payload)


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
