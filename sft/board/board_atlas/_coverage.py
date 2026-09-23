"""Coverage reports, topology invariant guards, and JSONL writing."""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, Iterable

import networkx as nx

from cle.game_engine.models.board import base_map
from cle.game_engine.models.map import NUM_EDGES, NUM_NODES, NUM_TILES
from sft.board.board_atlas._examples import AtlasExample, is_empty_fact
from sft.board.board_atlas._tables import FactTable
from sft.board.board_atlas._tokens import LAND_GRAPH, LAND_NODES, SCHEMA
from sft.json_types import JsonLikeDict

# --- coverage --------------------------------------------------------------


def answer_shape_report(
    tables: Dict[str, FactTable], examples: Iterable[AtlasExample]
) -> JsonLikeDict:
    """NONE share of emitted entries, per table and overall.

    The number to watch: r04 trained at 35% NONE and evaluated at 9%, and its
    dominant set-answer failure was over-predicting NONE. This makes the sampler's
    bias visible before a run instead of after one.
    """
    emitted: Dict[str, Counter[str]] = {name: Counter() for name in tables}
    for example in examples:
        table = tables[example.table]
        for key in example.keys:
            emitted[example.table]["empty" if is_empty_fact(table, key) else "filled"] += 1

    by_table: Dict[str, Dict[str, float]] = {}
    total_entries = 0
    total_empty = 0
    for name, counts in emitted.items():
        total = counts["empty"] + counts["filled"]
        available = sum(1 for k in tables[name].facts if is_empty_fact(tables[name], k))
        by_table[name] = {
            "entries": total,
            "empty": counts["empty"],
            "none_share": counts["empty"] / total if total else 0.0,
            "none_share_of_facts": available / len(tables[name]) if len(tables[name]) else 0.0,
        }
        total_entries += total
        total_empty += counts["empty"]
    return {
        "entries": total_entries,
        "empty": total_empty,
        "none_share": total_empty / total_entries if total_entries else 0.0,
        "by_table": by_table,
    }


def coverage_report(
    tables: Dict[str, FactTable], examples: Iterable[AtlasExample]
) -> JsonLikeDict:
    """Per-table and per-key emission counts. The atlas is a closed world, so
    uncovered keys are a generation bug, not a sampling outcome."""
    seen: Dict[str, Counter[str]] = {name: Counter() for name in tables}
    for ex in examples:
        seen[ex.table].update(ex.keys)

    report: Dict[str, JsonLikeDict] = {}
    total = 0
    covered = 0
    for name, table in tables.items():
        counts = seen[name]
        uncovered = sorted(k for k in table.facts if not counts[k])
        occurrences = [counts[k] for k in table.facts]
        report[name] = {
            "facts": len(table.facts),
            "covered": len(table.facts) - len(uncovered),
            "uncovered": uncovered,
            "min_occurrences": min(occurrences) if occurrences else 0,
            "max_occurrences": max(occurrences) if occurrences else 0,
        }
        total += len(table.facts)
        covered += len(table.facts) - len(uncovered)
    return {
        "schema": SCHEMA,
        "total_facts": total,
        "covered_facts": covered,
        "fully_covered": covered == total,
        "by_table": report,
    }


def assert_topology_invariants() -> None:
    """Guard the assumptions the corpus is built on."""
    if len(LAND_NODES) != NUM_NODES:
        raise AssertionError(f"expected {NUM_NODES} land nodes, got {len(LAND_NODES)}")
    if LAND_GRAPH.number_of_edges() != NUM_EDGES:
        raise AssertionError(
            f"expected {NUM_EDGES} land edges, got {LAND_GRAPH.number_of_edges()}"
        )
    if len(base_map.tiles_by_id) != NUM_TILES:
        raise AssertionError(
            f"expected {NUM_TILES} land tiles, got {len(base_map.tiles_by_id)}"
        )
    if not nx.is_connected(LAND_GRAPH):
        raise AssertionError("land graph is not connected; distance table would be partial")
    degrees = {d for _, d in LAND_GRAPH.degree()}
    if not degrees <= {2, 3}:
        raise AssertionError(f"unexpected node degrees on the land graph: {sorted(degrees)}")


def write_jsonl(examples: Iterable[AtlasExample], path: str) -> int:
    written = 0
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_row(), separators=(",", ":")) + "\n")
            written += 1
    return written
