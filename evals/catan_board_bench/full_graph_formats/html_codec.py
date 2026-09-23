"""Optimized HTML projection and reconstruction of the minimal graph."""

from __future__ import annotations

from evals.catan_board_bench.ascii_variations import CORNER_ORDER, SIDE_ORDER
from evals.catan_board_bench.ascii_variations.facts import FullFacts
from evals.catan_board_bench.full_graph_formats.codec import (
    _null_marker,
    _parse_binary,
    _parse_null_marker,
    _parse_optional_int,
)
from evals.catan_board_bench.full_graph_formats.graph import (
    expand_minimal_graph,
    minimal_graph_facts,
)
from evals.catan_board_bench.full_graph_formats.html_tables import (
    _CompactHtmlTableParser,
    _html_table,
    _require_headers,
)
from evals.catan_board_bench.full_graph_formats.schema import MINIMAL_SCHEMA


def render_optimized_html(facts: FullFacts) -> str:
    payload = minimal_graph_facts(facts)
    lines = [f'<main data-schema="{MINIMAL_SCHEMA}">']
    lines.extend(
        _html_table(
            "tiles",
            (
                "id",
                "q",
                "r",
                "s",
                "resource",
                "number",
                "robber",
                *[f"corner_{value.lower()}" for value in CORNER_ORDER],
                *[f"side_{value.lower()}" for value in SIDE_ORDER],
            ),
            (
                (
                    tile["id"],
                    *tile["cube"],
                    tile["resource"],
                    _null_marker(tile["number"]),
                    int(tile["robber"]),
                    *[tile["corners"][value] for value in CORNER_ORDER],
                    *[tile["sides"][value] for value in SIDE_ORDER],
                )
                for tile in payload["tiles"]
            ),
        )
    )
    lines.extend(
        _html_table(
            "nodes",
            ("id", "color", "building"),
            (
                (
                    node["id"],
                    _null_marker(node["color"]),
                    _null_marker(node["building"]),
                )
                for node in payload["nodes"]
            ),
        )
    )
    lines.extend(
        _html_table(
            "edges",
            ("id", "node_1", "node_2", "road"),
            (
                (
                    edge["id"],
                    edge["nodes"][0],
                    edge["nodes"][1],
                    _null_marker(edge["road"]),
                )
                for edge in payload["edges"]
            ),
        )
    )
    lines.extend(
        _html_table(
            "ports",
            (
                "id",
                "q",
                "r",
                "s",
                "direction",
                "resource",
                "ratio",
                "node_1",
                "node_2",
            ),
            (
                (
                    port["id"],
                    *port["cube"],
                    port["direction"],
                    port["resource"],
                    port["ratio"],
                    port["nodes"][0],
                    port["nodes"][1],
                )
                for port in payload["ports"]
            ),
        )
    )
    lines.append("</main>")
    return "\n".join(lines)


def parse_optimized_html(text: str) -> FullFacts:
    parser = _CompactHtmlTableParser()
    parser.feed(text)
    parser.close()
    if parser.schema != MINIMAL_SCHEMA:
        raise ValueError(f"unexpected HTML schema: {parser.schema!r}")
    expected_tables = {"tiles", "nodes", "edges", "ports"}
    if set(parser.tables) != expected_tables:
        raise ValueError(f"unexpected HTML tables: {sorted(parser.tables)}")

    tile_headers = (
        "id",
        "q",
        "r",
        "s",
        "resource",
        "number",
        "robber",
        *[f"corner_{value.lower()}" for value in CORNER_ORDER],
        *[f"side_{value.lower()}" for value in SIDE_ORDER],
    )
    _require_headers(parser.tables["tiles"], tile_headers)
    _require_headers(parser.tables["nodes"], ("id", "color", "building"))
    _require_headers(parser.tables["edges"], ("id", "node_1", "node_2", "road"))
    _require_headers(
        parser.tables["ports"],
        (
            "id",
            "q",
            "r",
            "s",
            "direction",
            "resource",
            "ratio",
            "node_1",
            "node_2",
        ),
    )

    tiles = []
    for row in parser.tables["tiles"][1:]:
        values = dict(zip(tile_headers, row))
        tiles.append(
            {
                "id": values["id"],
                "cube": [int(values[key]) for key in ("q", "r", "s")],
                "resource": values["resource"],
                "number": _parse_optional_int(values["number"]),
                "robber": _parse_binary(values["robber"]),
                "corners": {
                    direction: values[f"corner_{direction.lower()}"] for direction in CORNER_ORDER
                },
                "sides": {
                    direction: values[f"side_{direction.lower()}"] for direction in SIDE_ORDER
                },
            }
        )
    nodes = [
        {
            "id": row[0],
            "color": _parse_null_marker(row[1]),
            "building": _parse_null_marker(row[2]),
        }
        for row in parser.tables["nodes"][1:]
    ]
    edges = [
        {
            "id": row[0],
            "nodes": [row[1], row[2]],
            "road": _parse_null_marker(row[3]),
        }
        for row in parser.tables["edges"][1:]
    ]
    ports = [
        {
            "id": row[0],
            "cube": [int(row[index]) for index in (1, 2, 3)],
            "direction": row[4],
            "resource": row[5],
            "ratio": row[6],
            "nodes": [row[7], row[8]],
        }
        for row in parser.tables["ports"][1:]
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
