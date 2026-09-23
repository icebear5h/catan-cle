"""Integrated dynamic-state glyphs and the static topology record codec."""

from __future__ import annotations

import re

from evals.catan_board_bench.ascii_variations import CORNER_ORDER, SIDE_ORDER
from evals.catan_board_bench.ascii_variations.facts import FullFacts
from evals.catan_board_bench.full_graph_formats.codec import (
    _csv,
    _insert_unique,
    _mapping,
    _parse_csv,
    _parse_int_csv,
    _parse_mapping,
    _record_fields,
)
from evals.catan_board_bench.full_graph_formats.diagram import _integrated_diagram
from evals.catan_board_bench.full_graph_formats.graph import (
    _validated_full_facts,
    expand_minimal_graph,
)
from evals.catan_board_bench.full_graph_formats.schema import (
    _CODE_TO_BUILDING,
    _CODE_TO_COLOR,
    _COLOR_TO_CODE,
    _EXPECTED_COUNTS,
    MINIMAL_SCHEMA,
)


def render_integrated_ascii(facts: FullFacts) -> str:
    facts = _validated_full_facts(facts)
    color_legend = " ".join(f"{code}={color}" for color, code in sorted(_COLOR_TO_CODE.items()))
    lines = [
        "CATAN OCCUPATION-INTEGRATED BOARD V1",
        "Tile label: ID[RESOURCE/NUMBER*], where *=robber.",
        "Node label: ID[-] or ID[COLOR/S|C], where S=SETTLEMENT C=CITY.",
        "Edge label: ID[-] or ID[COLOR] for road ownership.",
        f"COLOR CODES: {color_legend}",
        "GLOBAL POINTY-TOP BOARD",
        *_integrated_diagram(facts),
        "",
        "STATIC TOPOLOGY (dynamic tile/node/edge state occurs only above)",
    ]
    for tile in facts["tiles"]:
        lines.append(
            "|".join(
                (
                    "IT",
                    tile["id"],
                    f"cube={_csv(tile['cube'])}",
                    f"corners={_mapping(tile['corners'], CORNER_ORDER)}",
                    f"sides={_mapping(tile['sides'], SIDE_ORDER)}",
                )
            )
        )
    for edge in facts["edges"]:
        lines.append(
            "|".join(
                (
                    "IE",
                    edge["id"],
                    f"nodes={_csv(edge['nodes'])}",
                )
            )
        )
    for port in facts["ports"]:
        lines.append(
            "|".join(
                (
                    "IP",
                    port["id"],
                    f"cube={_csv(port['cube'])}",
                    f"direction={port['direction']}",
                    f"resource={port['resource']}",
                    f"ratio={port['ratio']}",
                    f"nodes={_csv(port['nodes'])}",
                )
            )
        )
    return "\n".join(lines)


def parse_integrated_ascii(text: str) -> FullFacts:
    diagram_marker = "GLOBAL POINTY-TOP BOARD\n"
    static_marker = "\n\nSTATIC TOPOLOGY (dynamic tile/node/edge state occurs only above)\n"
    if text.count(diagram_marker) != 1 or text.count(static_marker) != 1:
        raise ValueError("integrated ASCII section markers must appear exactly once")
    _header, remainder = text.split(diagram_marker, 1)
    diagram_text, static_text = remainder.split(static_marker, 1)

    tile_states: dict[str, tuple[str, int | None, bool]] = {}
    node_states: dict[str, tuple[str | None, str | None]] = {}
    edge_states: dict[str, str | None] = {}

    tile_pattern = re.compile(r"\b(T\d{2})\[([A-Z_]+)/(-|\d+)(\*)?\]")
    node_pattern = re.compile(r"\b(N\d{2})\[(-|[A-Z]{2}/[SC])\]")
    edge_pattern = re.compile(r"\b(E\d{2})\[(-|[A-Z]{2})\]")
    if any(pattern.search(static_text) for pattern in (tile_pattern, node_pattern, edge_pattern)):
        raise ValueError("dynamic state glyph found outside integrated diagram")
    for match in tile_pattern.finditer(diagram_text):
        tile_value = (
            match.group(2),
            None if match.group(3) == "-" else int(match.group(3)),
            bool(match.group(4)),
        )
        _insert_unique(tile_states, match.group(1), tile_value, "tile state")
    for match in node_pattern.finditer(diagram_text):
        encoded = match.group(2)
        node_value: tuple[str | None, str | None]
        if encoded == "-":
            node_value = (None, None)
        else:
            color_code, building_code = encoded.split("/", 1)
            if color_code not in _CODE_TO_COLOR or building_code not in _CODE_TO_BUILDING:
                raise ValueError(f"unknown integrated node state: {encoded}")
            node_value = (_CODE_TO_COLOR[color_code], _CODE_TO_BUILDING[building_code])
        _insert_unique(node_states, match.group(1), node_value, "node state")
    for match in edge_pattern.finditer(diagram_text):
        encoded = match.group(2)
        edge_value: str | None
        if encoded == "-":
            edge_value = None
        else:
            if encoded not in _CODE_TO_COLOR:
                raise ValueError(f"unknown integrated edge state: {encoded}")
            edge_value = _CODE_TO_COLOR[encoded]
        _insert_unique(edge_states, match.group(1), edge_value, "edge state")

    tile_static: dict[str, dict[str, object]] = {}
    edge_static: dict[str, dict[str, list[str]]] = {}
    ports: list[dict[str, object]] = []
    for line in static_text.splitlines():
        if line.startswith("IT|"):
            parts = _record_fields(line)
            tile_id = parts.pop("id")
            topology: dict[str, object] = {
                "cube": _parse_int_csv(parts["cube"]),
                "corners": _parse_mapping(parts["corners"]),
                "sides": _parse_mapping(parts["sides"]),
            }
            _insert_unique(tile_static, tile_id, topology, "integrated tile topology")
        elif line.startswith("IE|"):
            parts = _record_fields(line)
            edge_id = parts.pop("id")
            _insert_unique(
                edge_static,
                edge_id,
                {"nodes": _parse_csv(parts["nodes"])},
                "integrated edge topology",
            )
        elif line.startswith("IP|"):
            parts = _record_fields(line)
            ports.append(
                {
                    "id": parts["id"],
                    "cube": _parse_int_csv(parts["cube"]),
                    "direction": parts["direction"],
                    "resource": parts["resource"],
                    "ratio": parts["ratio"],
                    "nodes": _parse_csv(parts["nodes"]),
                }
            )

    if (
        len(tile_states),
        len(node_states),
        len(edge_states),
        len(ports),
    ) != _EXPECTED_COUNTS:
        raise ValueError(
            "incomplete integrated state glyphs: "
            f"{(len(tile_states), len(node_states), len(edge_states), len(ports))}"
        )
    if set(tile_static) != set(tile_states):
        raise ValueError("integrated tile state/topology IDs disagree")
    if set(edge_static) != set(edge_states):
        raise ValueError("integrated edge state/topology IDs disagree")

    tiles = []
    for tile_id, static in tile_static.items():
        resource, number, robber = tile_states[tile_id]
        tiles.append(
            {
                "id": tile_id,
                **static,
                "resource": resource,
                "number": number,
                "robber": robber,
            }
        )
    nodes = [
        {"id": node_id, "color": state[0], "building": state[1]}
        for node_id, state in node_states.items()
    ]
    edges = [
        {
            "id": edge_id,
            "nodes": edge_static[edge_id]["nodes"],
            "road": road,
        }
        for edge_id, road in edge_states.items()
    ]
    return expand_minimal_graph(
        {
            "schema": MINIMAL_SCHEMA,
            "tiles": tiles,
            "nodes": nodes,
            "edges": edges,
            "ports": ports,
        }
    )
