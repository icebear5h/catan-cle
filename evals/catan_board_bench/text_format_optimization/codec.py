"""Rendering and parsing of the four indexed text-format candidates."""

from __future__ import annotations

import json

from evals.catan_board_bench.ascii_variations import (
    parse_ascii_variant,
    render_ascii_variant,
)
from evals.catan_board_bench.ascii_variations.facts import FullFacts
from evals.catan_board_bench.full_graph_formats import (
    expand_minimal_graph,
    minimal_graph_facts,
)
from evals.catan_board_bench.text_format_optimization.indexes import (
    _append_record_indexes,
    _query_index_lines,
    build_query_indexes,
)
from evals.catan_board_bench.text_format_optimization.schema import QUERY_INDEX_SCHEMA
from evals.catan_board_bench.text_format_optimization.support import (
    _reject_duplicate_pairs,
)


def render_text_format(name: str, facts: FullFacts, *, sample_id: str) -> str:
    """Render one optimization candidate."""

    if name == "tile_rows":
        return render_ascii_variant("tile_rows", facts, sample_id=sample_id)
    if name == "indexed_records":
        base = render_ascii_variant("sectioned", facts, sample_id=sample_id)
        return _append_record_indexes(base, facts)
    if name == "indexed_tile_rows":
        base = render_ascii_variant("tile_rows", facts, sample_id=sample_id)
        return _append_record_indexes(base, facts)
    if name == "indexed_json":
        return json.dumps(
            {
                "schema": QUERY_INDEX_SCHEMA,
                "graph": minimal_graph_facts(facts),
                "indexes": build_query_indexes(facts),
            },
            separators=(",", ":"),
        )
    raise ValueError(f"unknown text format: {name}")


def parse_text_format(name: str, text: str) -> FullFacts:
    """Parse and verify one optimization candidate."""

    if name == "tile_rows":
        return parse_ascii_variant(text)
    if name in {"indexed_records", "indexed_tile_rows"}:
        facts = parse_ascii_variant(text)
        actual = [line for line in text.splitlines() if line.startswith("QI|")]
        expected = _query_index_lines(facts)
        if actual != expected:
            raise ValueError("query-index sidecar is missing, reordered, or inconsistent")
        return facts
    if name == "indexed_json":
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
        if not isinstance(payload, dict) or payload.get("schema") != QUERY_INDEX_SCHEMA:
            raise ValueError("unexpected query-indexed JSON schema")
        graph = payload.get("graph")
        if not isinstance(graph, dict):
            raise ValueError("minimal graph must be an object")
        facts = expand_minimal_graph(graph)
        if payload.get("indexes") != build_query_indexes(facts):
            raise ValueError("query-indexed JSON contains inconsistent derived indexes")
        return facts
    raise ValueError(f"unknown text format: {name}")

