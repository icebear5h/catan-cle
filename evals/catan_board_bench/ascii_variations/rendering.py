"""ASCII layout selection and spatial sidecars over canonical records."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Sequence

from evals.catan_board_bench.ascii_variations.codec import _csv, _nullable
from evals.catan_board_bench.ascii_variations.diagram import _topology_diagram
from evals.catan_board_bench.ascii_variations.facts import FullFacts, FullTile
from evals.catan_board_bench.ascii_variations.graph import canonicalize_full_facts
from evals.catan_board_bench.ascii_variations.records import fact_record_lines
from evals.catan_board_bench.ascii_variations.schema import (
    ASCII_VARIANTS,
    CORNER_ORDER,
    RECORD_KINDS,
    SIDE_ORDER,
)


def render_ascii_variant(
    variant: str,
    facts: FullFacts,
    *,
    sample_id: str,
) -> str:
    if variant not in ASCII_VARIANTS:
        raise ValueError(f"unknown ASCII variant: {variant}")
    facts = canonicalize_full_facts(facts)
    records = fact_record_lines(facts)
    header = _ascii_header(variant)

    if variant == "flat_sorted":
        body = records
    elif variant == "flat_shuffled":
        body = list(records)
        seed = int.from_bytes(
            hashlib.sha256(f"{sample_id}:{variant}".encode()).digest()[:8],
            "big",
        )
        random.Random(seed).shuffle(body)
    elif variant == "sectioned":
        body = _sectioned_records(records)
    elif variant == "tile_rows":
        body = [
            "BOARD ROWS (TOP TO BOTTOM)",
            *_tile_row_sidecar(facts),
            "",
            "FULL FACT RECORDS",
            *_sectioned_records(records),
        ]
    elif variant == "local_blocks":
        body = [
            "LOCAL TILE BLOCKS",
            *_local_tile_blocks(facts),
            "",
            "FULL FACT RECORDS",
            *_sectioned_records(records),
        ]
    else:
        body = [
            "NODE / EDGE / TILE TOPOLOGY DIAGRAM",
            *_topology_diagram(facts),
            "",
            "FULL FACT RECORDS",
            *_sectioned_records(records),
        ]
    return "\n".join([*header, *body])


def _ascii_header(variant: str) -> list[str]:
    return [
        "CATAN FULL PUBLIC GRAPH V1",
        f"ASCII VARIANT={variant}",
        "ENTITY IDS ARE OPAQUE, PERMUTED, AND BOARD-LOCAL.",
        "ALL 19 TILES, 54 NODES, 72 EDGES, AND 9 PORTS ARE EXPLICIT.",
        "A dash (-) means no number, color, building, road, port, or adjacent tile.",
        "RECORDS: T=tile N=node E=edge P=port; lists are comma-separated.",
    ]


def _sectioned_records(records: Sequence[str]) -> list[str]:
    result = []
    labels = {"T": "TILES", "N": "NODES", "E": "EDGES", "P": "PORTS"}
    for kind in RECORD_KINDS:
        result.append(labels[kind])
        result.extend(line for line in records if line.startswith(f"{kind}|"))
    return result


def _tile_row_sidecar(facts: FullFacts) -> list[str]:
    rows: dict[int, list[FullTile]] = defaultdict(list)
    for tile in facts["tiles"]:
        rows[int(tile["cube"][2])].append(tile)
    max_count = max(len(row) for row in rows.values())
    result = []
    for row_index, z in enumerate(sorted(rows)):
        row = sorted(rows[z], key=lambda item: item["cube"][0])
        indent = "    " * (max_count - len(row))
        cells = []
        for tile in row:
            number = _nullable(tile["number"])
            robber = "*" if tile["robber"] else ""
            cells.append(f"{tile['id']}@({_csv(tile['cube'])})={tile['resource']}/{number}{robber}")
        result.append(f"ROW{row_index} {indent}" + "  ".join(cells))
    return result


def _local_tile_blocks(facts: FullFacts) -> list[str]:
    nodes = {item["id"]: item for item in facts["nodes"]}
    edges = {item["id"]: item for item in facts["edges"]}
    result = []
    for tile in sorted(facts["tiles"], key=lambda item: (item["cube"][2], item["cube"][0])):
        result.append(
            f"[{tile['id']} cube=({_csv(tile['cube'])}) "
            f"{tile['resource']}/{_nullable(tile['number'])} "
            f"robber={int(tile['robber'])}]"
        )
        corner_cells = []
        for direction in CORNER_ORDER:
            node = nodes[tile["corners"][direction]]
            state = "EMPTY" if node["building"] is None else f"{node['color']}/{node['building']}"
            corner_cells.append(f"{direction}:{node['id']}={state}")
        result.append("  corners " + " ".join(corner_cells))
        side_cells = []
        for direction in SIDE_ORDER:
            edge = edges[tile["sides"][direction]]
            side_cells.append(f"{direction}:{edge['id']}={edge['road'] or 'EMPTY'}")
        result.append("  sides " + " ".join(side_cells))
    return result
