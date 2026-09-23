"""Relational SQL rendering and in-memory reconstruction of the public graph."""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence

from evals.catan_board_bench.ascii_variations import CORNER_ORDER, SIDE_ORDER
from evals.catan_board_bench.ascii_variations.facts import FullFacts
from evals.catan_board_bench.full_graph_formats.graph import (
    expand_minimal_graph,
    minimal_graph_facts,
)
from evals.catan_board_bench.full_graph_formats.schema import MINIMAL_SCHEMA


def render_sql_relational(facts: FullFacts) -> str:
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


def parse_sql_relational(text: str) -> FullFacts:
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


def _sql_insert(table: str, rows: Iterable[Sequence[object]]) -> list[str]:
    values = ["(" + ",".join(_sql_value(value) for value in row) + ")" for row in rows]
    if not values:
        return []
    return [f"INSERT INTO {table} VALUES", ",\n".join(values) + ";"]


def _sql_value(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"
