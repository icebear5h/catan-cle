"""Lossless full-graph Catan renderings for the strict format search."""

from __future__ import annotations

import ast
import html
import json
import math
import re
import sqlite3
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any, Callable, Dict, Iterable, Sequence

from evals.catan_board_bench.ascii_variations import (
    CORNER_ORDER,
    FACT_SCHEMA,
    SIDE_ORDER,
    canonicalize_full_facts,
    parse_ascii_variant,
    render_ascii_variant,
)


JsonDict = Dict[str, Any]
MINIMAL_SCHEMA = "catan_full_public_graph_minimal/v1"
FORMAT_NAMES = (
    "optimized_html",
    "full_graph_json",
    "datalog",
    "sql_relational",
    "integrated_ascii",
    "tile_rows",
)
FORMAT_EXTENSIONS = {
    "optimized_html": ".html",
    "full_graph_json": ".json",
    "datalog": ".dl",
    "sql_relational": ".sql",
    "integrated_ascii": ".txt",
    "tile_rows": ".txt",
}

_COLOR_TO_CODE = {
    "BLACK": "BK",
    "BLUE": "BL",
    "BRONZE": "BZ",
    "GOLD": "GD",
    "GREEN": "GR",
    "MYSTIC_BLUE": "MB",
    "ORANGE": "OR",
    "PINK": "PK",
    "RED": "RD",
    "SILVER": "SV",
    "WHITE": "WH",
}
_CODE_TO_COLOR = {code: color for color, code in _COLOR_TO_CODE.items()}
_BUILDING_TO_CODE = {"SETTLEMENT": "S", "CITY": "C"}
_CODE_TO_BUILDING = {code: value for value, code in _BUILDING_TO_CODE.items()}
_SIDE_ENDPOINTS = {
    "EAST": ("NORTHEAST", "SOUTHEAST"),
    "SOUTHEAST": ("SOUTHEAST", "SOUTH"),
    "SOUTHWEST": ("SOUTH", "SOUTHWEST"),
    "WEST": ("SOUTHWEST", "NORTHWEST"),
    "NORTHWEST": ("NORTHWEST", "NORTH"),
    "NORTHEAST": ("NORTH", "NORTHEAST"),
}
_EXPECTED_COUNTS = (19, 54, 72, 9)


def render_full_graph_format(
    name: str,
    facts: JsonDict,
    *,
    sample_id: str,
) -> str:
    """Render one full public graph in the selected representation."""

    canonical = _validated_full_facts(facts)
    if name == "tile_rows":
        return render_ascii_variant("tile_rows", canonical, sample_id=sample_id)
    try:
        renderer = _RENDERERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown full-graph format: {name}") from exc
    return renderer(canonical)


def parse_full_graph_format(name: str, text: str) -> JsonDict:
    """Parse one representation back to the canonical public fact graph."""

    try:
        parser = _PARSERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown full-graph format: {name}") from exc
    parsed = parser(text)
    return _validated_full_facts(parsed)


def minimal_graph_facts(facts: JsonDict) -> JsonDict:
    """Remove redundant incidence lists while retaining all source information."""

    facts = _validated_full_facts(facts)
    return {
        "schema": MINIMAL_SCHEMA,
        "tiles": [
            {
                "id": tile["id"],
                "cube": list(tile["cube"]),
                "resource": tile["resource"],
                "number": tile["number"],
                "robber": tile["robber"],
                "corners": {direction: tile["corners"][direction] for direction in CORNER_ORDER},
                "sides": {direction: tile["sides"][direction] for direction in SIDE_ORDER},
            }
            for tile in facts["tiles"]
        ],
        "nodes": [
            {
                "id": node["id"],
                "color": node["color"],
                "building": node["building"],
            }
            for node in facts["nodes"]
        ],
        "edges": [
            {
                "id": edge["id"],
                "nodes": list(edge["nodes"]),
                "road": edge["road"],
            }
            for edge in facts["edges"]
        ],
        "ports": [
            {
                "id": port["id"],
                "cube": list(port["cube"]),
                "direction": port["direction"],
                "resource": port["resource"],
                "ratio": port["ratio"],
                "nodes": list(port["nodes"]),
            }
            for port in facts["ports"]
        ],
    }


def expand_minimal_graph(payload: JsonDict) -> JsonDict:
    """Reconstruct redundant incidence fields from a minimal typed graph."""

    _validate_minimal_payload(payload)
    tiles = payload["tiles"]
    nodes = payload["nodes"]
    edges = payload["edges"]
    ports = payload["ports"]

    node_ids = {node["id"] for node in nodes}
    edge_ids = {edge["id"] for edge in edges}
    node_tiles: dict[str, set[str]] = defaultdict(set)
    node_edges: dict[str, set[str]] = defaultdict(set)
    node_ports: dict[str, set[str]] = defaultdict(set)
    edge_tiles: dict[str, set[str]] = defaultdict(set)

    for tile in tiles:
        for node_id in tile["corners"].values():
            if node_id not in node_ids:
                raise ValueError(f"tile {tile['id']} references unknown node {node_id}")
            node_tiles[node_id].add(tile["id"])
        for direction, edge_id in tile["sides"].items():
            if edge_id not in edge_ids:
                raise ValueError(f"tile {tile['id']} references unknown edge {edge_id}")
            edge_tiles[edge_id].add(tile["id"])
            endpoint_directions = _SIDE_ENDPOINTS[direction]
            expected_nodes = {
                tile["corners"][endpoint_directions[0]],
                tile["corners"][endpoint_directions[1]],
            }
            edge = next(item for item in edges if item["id"] == edge_id)
            if set(edge["nodes"]) != expected_nodes:
                raise ValueError(f"edge {edge_id} endpoints disagree with tile {tile['id']}")

    for edge in edges:
        for node_id in edge["nodes"]:
            if node_id not in node_ids:
                raise ValueError(f"edge {edge['id']} references unknown node {node_id}")
            node_edges[node_id].add(edge["id"])
    for port in ports:
        for node_id in port["nodes"]:
            if node_id not in node_ids:
                raise ValueError(f"port {port['id']} references unknown node {node_id}")
            node_ports[node_id].add(port["id"])

    facts = {
        "schema": FACT_SCHEMA,
        "tiles": tiles,
        "nodes": [
            {
                **node,
                "tiles": sorted(node_tiles[node["id"]]),
                "edges": sorted(node_edges[node["id"]]),
                "ports": sorted(node_ports[node["id"]]),
            }
            for node in nodes
        ],
        "edges": [{**edge, "tiles": sorted(edge_tiles[edge["id"]])} for edge in edges],
        "ports": ports,
    }
    return _validated_full_facts(facts)


def render_full_graph_json(facts: JsonDict) -> str:
    return json.dumps(
        minimal_graph_facts(facts),
        separators=(",", ":"),
        sort_keys=False,
    )


def parse_full_graph_json(text: str) -> JsonDict:
    payload = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(payload, dict):
        raise ValueError("full-graph JSON must be one object")
    return expand_minimal_graph(payload)


def render_optimized_html(facts: JsonDict) -> str:
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


def parse_optimized_html(text: str) -> JsonDict:
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


def render_datalog(facts: JsonDict) -> str:
    payload = minimal_graph_facts(facts)
    lines = [f'schema("{MINIMAL_SCHEMA}").']
    for tile in payload["tiles"]:
        number = "none" if tile["number"] is None else tile["number"]
        lines.append(
            _datalog_fact(
                "tile",
                (
                    tile["id"],
                    *tile["cube"],
                    tile["resource"],
                    number,
                    "true" if tile["robber"] else "false",
                ),
            )
        )
        lines.extend(
            _datalog_fact("tile_corner", (tile["id"], direction, node_id))
            for direction, node_id in tile["corners"].items()
        )
        lines.extend(
            _datalog_fact("tile_side", (tile["id"], direction, edge_id))
            for direction, edge_id in tile["sides"].items()
        )
    lines.extend(
        _datalog_fact(
            "node",
            (
                node["id"],
                node["color"] or "none",
                node["building"] or "none",
            ),
        )
        for node in payload["nodes"]
    )
    lines.extend(
        _datalog_fact(
            "edge",
            (
                edge["id"],
                edge["nodes"][0],
                edge["nodes"][1],
                edge["road"] or "none",
            ),
        )
        for edge in payload["edges"]
    )
    lines.extend(
        _datalog_fact(
            "port",
            (
                port["id"],
                *port["cube"],
                port["direction"],
                port["resource"],
                port["ratio"],
                port["nodes"][0],
                port["nodes"][1],
            ),
        )
        for port in payload["ports"]
    )
    return "\n".join(lines)


def parse_datalog(text: str) -> JsonDict:
    rows: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    pattern = re.compile(r"([a-z_]+)\((.*)\)\.")
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = pattern.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"invalid Datalog line {line_number}: {line!r}")
        predicate = match.group(1)
        if predicate not in {
            "schema",
            "tile",
            "tile_corner",
            "tile_side",
            "node",
            "edge",
            "port",
        }:
            raise ValueError(f"unknown Datalog predicate: {predicate}")
        try:
            arguments = ast.literal_eval(f"({match.group(2)},)")
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid Datalog arguments on line {line_number}") from exc
        rows[predicate].append(arguments)

    if rows["schema"] != [(MINIMAL_SCHEMA,)]:
        raise ValueError("Datalog schema fact must appear exactly once")
    _require_unique_rows(rows)
    _require_arity(rows, "tile", 7)
    _require_arity(rows, "tile_corner", 3)
    _require_arity(rows, "tile_side", 3)
    _require_arity(rows, "node", 3)
    _require_arity(rows, "edge", 4)
    _require_arity(rows, "port", 9)

    corners: dict[str, dict[str, str]] = defaultdict(dict)
    sides: dict[str, dict[str, str]] = defaultdict(dict)
    for tile_id, direction, node_id in rows["tile_corner"]:
        if direction in corners[tile_id]:
            raise ValueError(f"duplicate Datalog corner {tile_id}/{direction}")
        corners[tile_id][direction] = node_id
    for tile_id, direction, edge_id in rows["tile_side"]:
        if direction in sides[tile_id]:
            raise ValueError(f"duplicate Datalog side {tile_id}/{direction}")
        sides[tile_id][direction] = edge_id

    tiles = []
    for row in rows["tile"]:
        tile_id, q, r, s, resource, number, robber = row
        tiles.append(
            {
                "id": tile_id,
                "cube": [q, r, s],
                "resource": resource,
                "number": None if number == "none" else number,
                "robber": _parse_boolean_atom(robber),
                "corners": corners[tile_id],
                "sides": sides[tile_id],
            }
        )
    payload = {
        "schema": MINIMAL_SCHEMA,
        "tiles": tiles,
        "nodes": [
            {
                "id": row[0],
                "color": None if row[1] == "none" else row[1],
                "building": None if row[2] == "none" else row[2],
            }
            for row in rows["node"]
        ],
        "edges": [
            {
                "id": row[0],
                "nodes": [row[1], row[2]],
                "road": None if row[3] == "none" else row[3],
            }
            for row in rows["edge"]
        ],
        "ports": [
            {
                "id": row[0],
                "cube": list(row[1:4]),
                "direction": row[4],
                "resource": row[5],
                "ratio": row[6],
                "nodes": [row[7], row[8]],
            }
            for row in rows["port"]
        ],
    }
    return expand_minimal_graph(payload)


def render_sql_relational(facts: JsonDict) -> str:
    payload = minimal_graph_facts(facts)
    lines = [
        "BEGIN;",
        "CREATE TABLE metadata(schema TEXT PRIMARY KEY);",
        "CREATE TABLE tiles(id TEXT PRIMARY KEY,q INTEGER,r INTEGER,s INTEGER,resource TEXT,number INTEGER,robber INTEGER);",
        "CREATE TABLE tile_corners(tile TEXT,direction TEXT,node TEXT,PRIMARY KEY(tile,direction));",
        "CREATE TABLE tile_sides(tile TEXT,direction TEXT,edge TEXT,PRIMARY KEY(tile,direction));",
        "CREATE TABLE nodes(id TEXT PRIMARY KEY,color TEXT,building TEXT);",
        "CREATE TABLE edges(id TEXT PRIMARY KEY,node_1 TEXT,node_2 TEXT,road TEXT);",
        "CREATE TABLE ports(id TEXT PRIMARY KEY,q INTEGER,r INTEGER,s INTEGER,direction TEXT,resource TEXT,ratio TEXT,node_1 TEXT,node_2 TEXT);",
        f"INSERT INTO metadata VALUES ({_sql_value(MINIMAL_SCHEMA)});",
    ]
    lines.extend(
        _sql_insert(
            "tiles",
            (
                (
                    tile["id"],
                    *tile["cube"],
                    tile["resource"],
                    tile["number"],
                    int(tile["robber"]),
                )
                for tile in payload["tiles"]
            ),
        )
    )
    lines.extend(
        _sql_insert(
            "tile_corners",
            (
                (tile["id"], direction, tile["corners"][direction])
                for tile in payload["tiles"]
                for direction in CORNER_ORDER
            ),
        )
    )
    lines.extend(
        _sql_insert(
            "tile_sides",
            (
                (tile["id"], direction, tile["sides"][direction])
                for tile in payload["tiles"]
                for direction in SIDE_ORDER
            ),
        )
    )
    lines.extend(
        _sql_insert(
            "nodes",
            ((node["id"], node["color"], node["building"]) for node in payload["nodes"]),
        )
    )
    lines.extend(
        _sql_insert(
            "edges",
            (
                (
                    edge["id"],
                    edge["nodes"][0],
                    edge["nodes"][1],
                    edge["road"],
                )
                for edge in payload["edges"]
            ),
        )
    )
    lines.extend(
        _sql_insert(
            "ports",
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
    lines.append("COMMIT;")
    return "\n".join(lines)


def parse_sql_relational(text: str) -> JsonDict:
    allowed = re.compile(
        r"(?:BEGIN|COMMIT|CREATE TABLE (?:metadata|tiles|tile_corners|tile_sides|nodes|edges|ports)\b|INSERT INTO (?:metadata|tiles|tile_corners|tile_sides|nodes|edges|ports)\b)",
        re.IGNORECASE,
    )
    statements = [statement.strip() for statement in text.split(";") if statement.strip()]
    for statement in statements:
        if allowed.match(statement) is None:
            raise ValueError(f"unsupported SQL statement: {statement[:80]!r}")

    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(text)
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [row[0] for row in table_rows]
        expected_tables = sorted(
            ("metadata", "tiles", "tile_corners", "tile_sides", "nodes", "edges", "ports")
        )
        if table_names != expected_tables:
            raise ValueError(f"unexpected SQL tables: {table_names}")
        schema_rows = connection.execute("SELECT schema FROM metadata").fetchall()
        if schema_rows != [(MINIMAL_SCHEMA,)]:
            raise ValueError("SQL metadata schema must appear exactly once")

        corner_rows = connection.execute(
            "SELECT tile,direction,node FROM tile_corners ORDER BY tile,direction"
        ).fetchall()
        side_rows = connection.execute(
            "SELECT tile,direction,edge FROM tile_sides ORDER BY tile,direction"
        ).fetchall()
        corners: dict[str, dict[str, str]] = defaultdict(dict)
        sides: dict[str, dict[str, str]] = defaultdict(dict)
        for tile_id, direction, node_id in corner_rows:
            corners[tile_id][direction] = node_id
        for tile_id, direction, edge_id in side_rows:
            sides[tile_id][direction] = edge_id

        tiles = [
            {
                "id": row[0],
                "cube": list(row[1:4]),
                "resource": row[4],
                "number": row[5],
                "robber": bool(row[6]),
                "corners": {direction: corners[row[0]][direction] for direction in CORNER_ORDER},
                "sides": {direction: sides[row[0]][direction] for direction in SIDE_ORDER},
            }
            for row in connection.execute(
                "SELECT id,q,r,s,resource,number,robber FROM tiles ORDER BY id"
            )
        ]
        nodes = [
            {"id": row[0], "color": row[1], "building": row[2]}
            for row in connection.execute("SELECT id,color,building FROM nodes ORDER BY id")
        ]
        edges = [
            {"id": row[0], "nodes": [row[1], row[2]], "road": row[3]}
            for row in connection.execute("SELECT id,node_1,node_2,road FROM edges ORDER BY id")
        ]
        ports = [
            {
                "id": row[0],
                "cube": list(row[1:4]),
                "direction": row[4],
                "resource": row[5],
                "ratio": row[6],
                "nodes": [row[7], row[8]],
            }
            for row in connection.execute(
                "SELECT id,q,r,s,direction,resource,ratio,node_1,node_2 FROM ports ORDER BY id"
            )
        ]
    except sqlite3.Error as exc:
        raise ValueError(f"invalid relational SQL: {exc}") from exc
    finally:
        connection.close()
    return expand_minimal_graph(
        {
            "schema": MINIMAL_SCHEMA,
            "tiles": tiles,
            "nodes": nodes,
            "edges": edges,
            "ports": ports,
        }
    )


def render_integrated_ascii(facts: JsonDict) -> str:
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


def parse_integrated_ascii(text: str) -> JsonDict:
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
        value = (
            match.group(2),
            None if match.group(3) == "-" else int(match.group(3)),
            bool(match.group(4)),
        )
        _insert_unique(tile_states, match.group(1), value, "tile state")
    for match in node_pattern.finditer(diagram_text):
        encoded = match.group(2)
        if encoded == "-":
            value = (None, None)
        else:
            color_code, building_code = encoded.split("/", 1)
            if color_code not in _CODE_TO_COLOR or building_code not in _CODE_TO_BUILDING:
                raise ValueError(f"unknown integrated node state: {encoded}")
            value = (_CODE_TO_COLOR[color_code], _CODE_TO_BUILDING[building_code])
        _insert_unique(node_states, match.group(1), value, "node state")
    for match in edge_pattern.finditer(diagram_text):
        encoded = match.group(2)
        if encoded == "-":
            value = None
        else:
            if encoded not in _CODE_TO_COLOR:
                raise ValueError(f"unknown integrated edge state: {encoded}")
            value = _CODE_TO_COLOR[encoded]
        _insert_unique(edge_states, match.group(1), value, "edge state")

    tile_static: dict[str, JsonDict] = {}
    edge_static: dict[str, JsonDict] = {}
    ports: list[JsonDict] = []
    for line in static_text.splitlines():
        if line.startswith("IT|"):
            parts = _record_fields(line)
            tile_id = parts.pop("id")
            _insert_unique(
                tile_static,
                tile_id,
                {
                    "cube": _parse_int_csv(parts["cube"]),
                    "corners": _parse_mapping(parts["corners"]),
                    "sides": _parse_mapping(parts["sides"]),
                },
                "integrated tile topology",
            )
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


def _validated_full_facts(facts: JsonDict) -> JsonDict:
    canonical = canonicalize_full_facts(facts)
    counts = tuple(len(canonical[key]) for key in ("tiles", "nodes", "edges", "ports"))
    if counts != _EXPECTED_COUNTS:
        raise ValueError(f"expected full 19/54/72/9 graph, found {counts}")
    if canonical.get("schema") != FACT_SCHEMA:
        raise ValueError(f"unexpected canonical fact schema: {canonical.get('schema')}")
    for collection in ("tiles", "nodes", "edges", "ports"):
        ids = [item["id"] for item in canonical[collection]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate IDs in {collection}")
    return canonical


def _validate_minimal_payload(payload: JsonDict) -> None:
    if not isinstance(payload, dict):
        raise ValueError("minimal graph must be an object")
    if set(payload) != {"schema", "tiles", "nodes", "edges", "ports"}:
        raise ValueError(f"unexpected minimal graph keys: {sorted(payload)}")
    if payload["schema"] != MINIMAL_SCHEMA:
        raise ValueError(f"unexpected minimal graph schema: {payload['schema']!r}")
    counts = tuple(len(payload[key]) for key in ("tiles", "nodes", "edges", "ports"))
    if counts != _EXPECTED_COUNTS:
        raise ValueError(f"incomplete minimal graph: {counts}")
    expected_keys = {
        "tiles": {"id", "cube", "resource", "number", "robber", "corners", "sides"},
        "nodes": {"id", "color", "building"},
        "edges": {"id", "nodes", "road"},
        "ports": {"id", "cube", "direction", "resource", "ratio", "nodes"},
    }
    for collection, keys in expected_keys.items():
        ids = []
        for item in payload[collection]:
            if not isinstance(item, dict) or set(item) != keys:
                raise ValueError(f"invalid {collection} item keys")
            ids.append(item["id"])
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate minimal IDs in {collection}")
    for tile in payload["tiles"]:
        if tuple(tile["corners"]) != CORNER_ORDER:
            raise ValueError(f"tile {tile['id']} has invalid corner mapping")
        if tuple(tile["sides"]) != SIDE_ORDER:
            raise ValueError(f"tile {tile['id']} has invalid side mapping")
        if len(tile["cube"]) != 3 or sum(tile["cube"]) != 0:
            raise ValueError(f"tile {tile['id']} has invalid cube coordinate")
    for edge in payload["edges"]:
        if len(edge["nodes"]) != 2 or edge["nodes"][0] == edge["nodes"][1]:
            raise ValueError(f"edge {edge['id']} has invalid endpoints")
    for port in payload["ports"]:
        if len(port["cube"]) != 3 or sum(port["cube"]) != 0:
            raise ValueError(f"port {port['id']} has invalid cube coordinate")
        if len(port["nodes"]) != 2 or port["nodes"][0] == port["nodes"][1]:
            raise ValueError(f"port {port['id']} has invalid nodes")


def _html_table(
    table_id: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> list[str]:
    lines = [f'<table id="{table_id}">', _html_row("th", headers)]
    lines.extend(_html_row("td", row) for row in rows)
    lines.append("</table>")
    return lines


def _html_row(tag: str, values: Sequence[Any]) -> str:
    return "<tr>" + "".join(f"<{tag}>{html.escape(str(value), quote=True)}" for value in values)


class _CompactHtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.schema: str | None = None
        self.tables: dict[str, list[list[str]]] = {}
        self._table: str | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag == "main":
            if self.schema is not None:
                raise ValueError("duplicate HTML main element")
            self.schema = attributes.get("data-schema")
        elif tag == "table":
            self._finish_table()
            table_id = attributes.get("id")
            if not table_id or table_id in self.tables:
                raise ValueError(f"invalid or duplicate HTML table: {table_id!r}")
            self._table = table_id
            self.tables[table_id] = []
        elif tag == "tr":
            self._finish_row()
            if self._table is None:
                raise ValueError("HTML row outside table")
            self._row = []
        elif tag in {"th", "td"}:
            self._finish_cell()
            if self._row is None:
                raise ValueError("HTML cell outside row")
            self._cell = []
        elif tag not in {"meta"}:
            raise ValueError(f"unsupported HTML tag: {tag}")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"}:
            self._finish_cell()
        elif tag == "tr":
            self._finish_row()
        elif tag == "table":
            self._finish_table()
        elif tag == "main":
            self._finish_table()
        else:
            raise ValueError(f"unsupported HTML closing tag: {tag}")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)
        elif data.strip():
            raise ValueError(f"unexpected HTML text: {data!r}")

    def close(self) -> None:
        super().close()
        self._finish_table()

    def _finish_cell(self) -> None:
        if self._cell is not None:
            if self._row is None:
                raise ValueError("HTML cell lost its row")
            self._row.append("".join(self._cell).strip())
            self._cell = None

    def _finish_row(self) -> None:
        self._finish_cell()
        if self._row is not None:
            if self._table is None:
                raise ValueError("HTML row lost its table")
            self.tables[self._table].append(self._row)
            self._row = None

    def _finish_table(self) -> None:
        self._finish_row()
        self._table = None


def _require_headers(rows: Sequence[Sequence[str]], expected: Sequence[str]) -> None:
    if not rows or tuple(rows[0]) != tuple(expected):
        raise ValueError(f"unexpected HTML headers: {rows[0] if rows else None}")
    for row in rows[1:]:
        if len(row) != len(expected):
            raise ValueError("HTML table row has wrong field count")


def _datalog_fact(predicate: str, values: Sequence[Any]) -> str:
    return f"{predicate}(" + ",".join(_datalog_value(value) for value in values) + ")."


def _datalog_value(value: Any) -> str:
    if isinstance(value, bool):
        return '"true"' if value else '"false"'
    if isinstance(value, int):
        return str(value)
    return json.dumps(str(value))


def _require_unique_rows(rows: dict[str, list[tuple[Any, ...]]]) -> None:
    for predicate, values in rows.items():
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate Datalog facts for {predicate}")


def _require_arity(
    rows: dict[str, list[tuple[Any, ...]]],
    predicate: str,
    arity: int,
) -> None:
    if any(len(row) != arity for row in rows[predicate]):
        raise ValueError(f"invalid {predicate} arity")


def _sql_insert(table: str, rows: Iterable[Sequence[Any]]) -> list[str]:
    values = ["(" + ",".join(_sql_value(value) for value in row) + ")" for row in rows]
    if not values:
        return []
    return [f"INSERT INTO {table} VALUES", ",\n".join(values) + ";"]


def _sql_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _integrated_diagram(facts: JsonDict) -> list[str]:
    tile_centers = {tile["id"]: _cube_point(tile["cube"]) for tile in facts["tiles"]}
    corner_offsets = {
        "NORTH": (0.0, -1.0),
        "NORTHEAST": (math.sqrt(3) / 2, -0.5),
        "SOUTHEAST": (math.sqrt(3) / 2, 0.5),
        "SOUTH": (0.0, 1.0),
        "SOUTHWEST": (-math.sqrt(3) / 2, 0.5),
        "NORTHWEST": (-math.sqrt(3) / 2, -0.5),
    }
    node_points: dict[str, tuple[float, float]] = {}
    for tile in facts["tiles"]:
        center = tile_centers[tile["id"]]
        for direction, node_id in tile["corners"].items():
            offset = corner_offsets[direction]
            point = (center[0] + offset[0], center[1] + offset[1])
            previous = node_points.get(node_id)
            if previous is not None and (
                abs(previous[0] - point[0]) > 1e-6 or abs(previous[1] - point[1]) > 1e-6
            ):
                raise ValueError(f"inconsistent diagram position for {node_id}")
            node_points[node_id] = point

    all_points = [*node_points.values(), *tile_centers.values()]
    min_x = min(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)

    def grid(point: tuple[float, float]) -> tuple[int, int]:
        return (
            round((point[0] - min_x) * 22) + 18,
            round((point[1] - min_y) * 9) + 3,
        )

    grid_nodes = {node_id: grid(point) for node_id, point in node_points.items()}
    grid_tiles = {tile_id: grid(point) for tile_id, point in tile_centers.items()}
    max_x = max(point[0] for point in [*grid_nodes.values(), *grid_tiles.values()]) + 18
    max_y = max(point[1] for point in [*grid_nodes.values(), *grid_tiles.values()]) + 4
    canvas = [[" " for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    node_by_id = {node["id"]: node for node in facts["nodes"]}
    for edge in facts["edges"]:
        start = grid_nodes[edge["nodes"][0]]
        end = grid_nodes[edge["nodes"][1]]
        _draw_line(canvas, start, end)
    for edge in facts["edges"]:
        start = grid_nodes[edge["nodes"][0]]
        end = grid_nodes[edge["nodes"][1]]
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        state = "-" if edge["road"] is None else _COLOR_TO_CODE[edge["road"]]
        _overlay_strict(canvas, midpoint, f"{edge['id']}[{state}]")
    for tile in facts["tiles"]:
        number = "-" if tile["number"] is None else tile["number"]
        robber = "*" if tile["robber"] else ""
        label = f"{tile['id']}[{tile['resource']}/{number}{robber}]"
        _overlay_strict(canvas, grid_tiles[tile["id"]], label)
    for node_id, point in grid_nodes.items():
        node = node_by_id[node_id]
        if node["building"] is None:
            state = "-"
        else:
            state = f"{_COLOR_TO_CODE[node['color']]}/{_BUILDING_TO_CODE[node['building']]}"
        _overlay_strict(canvas, point, f"{node_id}[{state}]")

    return [line.rstrip() for line in ("".join(row) for row in canvas) if line.rstrip()]


def _cube_point(cube: Sequence[int]) -> tuple[float, float]:
    q = cube[0]
    r = cube[2]
    return (math.sqrt(3) * (q + r / 2), 1.5 * r)


def _draw_line(
    canvas: list[list[str]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0))
    if steps == 0:
        return
    glyph = "-" if y0 == y1 else ("\\" if (x1 - x0) * (y1 - y0) > 0 else "/")
    for step in range(1, steps):
        x = round(x0 + (x1 - x0) * step / steps)
        y = round(y0 + (y1 - y0) * step / steps)
        if canvas[y][x] == " ":
            canvas[y][x] = glyph


def _overlay_strict(
    canvas: list[list[str]],
    center: tuple[int, int],
    label: str,
) -> None:
    start_x = center[0] - len(label) // 2
    y = center[1]
    for offset, character in enumerate(label):
        x = start_x + offset
        if not (0 <= y < len(canvas) and 0 <= x < len(canvas[y])):
            raise ValueError(f"diagram label out of bounds: {label}")
        existing = canvas[y][x]
        if existing not in {" ", "-", "/", "\\"}:
            raise ValueError(f"diagram label collision for {label!r} at {(x, y)} with {existing!r}")
        canvas[y][x] = character


def _record_fields(line: str) -> dict[str, str]:
    parts = line.split("|")
    result = {"id": parts[1]}
    for part in parts[2:]:
        key, value = part.split("=", 1)
        if key in result:
            raise ValueError(f"duplicate integrated record field: {key}")
        result[key] = value
    return result


def _insert_unique(
    mapping: dict[str, Any],
    key: str,
    value: Any,
    label: str,
) -> None:
    if key in mapping:
        raise ValueError(f"duplicate {label}: {key}")
    mapping[key] = value


def _mapping(mapping: JsonDict, order: Sequence[str]) -> str:
    return ",".join(f"{key}:{mapping[key]}" for key in order)


def _parse_mapping(value: str) -> dict[str, str]:
    return dict(item.split(":", 1) for item in value.split(","))


def _csv(values: Sequence[Any]) -> str:
    return "-" if not values else ",".join(str(value) for value in values)


def _parse_csv(value: str) -> list[str]:
    return [] if value == "-" else value.split(",")


def _parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _null_marker(value: Any) -> str:
    return "-" if value is None else str(value)


def _parse_null_marker(value: str) -> str | None:
    return None if value == "-" else value


def _parse_optional_int(value: str) -> int | None:
    return None if value == "-" else int(value)


def _parse_binary(value: str) -> bool:
    if value not in {"0", "1"}:
        raise ValueError(f"expected binary value, found {value!r}")
    return value == "1"


def _parse_boolean_atom(value: str) -> bool:
    if value not in {"true", "false"}:
        raise ValueError(f"expected Datalog boolean atom, found {value!r}")
    return value == "true"


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> JsonDict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


_RENDERERS: dict[str, Callable[[JsonDict], str]] = {
    "optimized_html": render_optimized_html,
    "full_graph_json": render_full_graph_json,
    "datalog": render_datalog,
    "sql_relational": render_sql_relational,
    "integrated_ascii": render_integrated_ascii,
}
_PARSERS: dict[str, Callable[[str], JsonDict]] = {
    "optimized_html": parse_optimized_html,
    "full_graph_json": parse_full_graph_json,
    "datalog": parse_datalog,
    "sql_relational": parse_sql_relational,
    "integrated_ascii": parse_integrated_ascii,
    "tile_rows": parse_ascii_variant,
}
