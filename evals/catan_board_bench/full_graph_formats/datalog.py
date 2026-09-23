"""Datalog fact rendering and strict predicate/arity reconstruction."""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from collections.abc import Sequence

from evals.catan_board_bench.ascii_variations.facts import FullFacts
from evals.catan_board_bench.full_graph_formats.codec import _parse_boolean_atom
from evals.catan_board_bench.full_graph_formats.graph import (
    expand_minimal_graph,
    minimal_graph_facts,
)
from evals.catan_board_bench.full_graph_formats.schema import MINIMAL_SCHEMA


def render_datalog(facts: FullFacts) -> str:
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


def parse_datalog(text: str) -> FullFacts:
    rows: dict[str, list[tuple[object, ...]]] = defaultdict(list)
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
    for tile_id, direction, node_id in (_atoms(row) for row in rows["tile_corner"]):
        if direction in corners[tile_id]:
            raise ValueError(f"duplicate Datalog corner {tile_id}/{direction}")
        corners[tile_id][direction] = node_id
    for tile_id, direction, edge_id in (_atoms(row) for row in rows["tile_side"]):
        if direction in sides[tile_id]:
            raise ValueError(f"duplicate Datalog side {tile_id}/{direction}")
        sides[tile_id][direction] = edge_id

    tiles: list[dict[str, object]] = []
    for row in rows["tile"]:
        row_tile_id, q, r, s, resource, number, robber = row
        tiles.append(
            {
                "id": row_tile_id,
                "cube": [q, r, s],
                "resource": resource,
                "number": None if number == "none" else number,
                "robber": _parse_boolean_atom(robber),
                "corners": corners[_atom(row_tile_id)],
                "sides": sides[_atom(row_tile_id)],
            }
        )
    payload: dict[str, object] = {
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


def _datalog_fact(predicate: str, values: Sequence[object]) -> str:
    return f"{predicate}(" + ",".join(_datalog_value(value) for value in values) + ")."


def _datalog_value(value: object) -> str:
    if isinstance(value, bool):
        return '"true"' if value else '"false"'
    if isinstance(value, int):
        return str(value)
    return json.dumps(str(value))


def _atom(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"expected Datalog string atom, found {value!r}")
    return value


def _atoms(row: tuple[object, ...]) -> tuple[str, str, str]:
    first, second, third = row
    return _atom(first), _atom(second), _atom(third)


def _require_unique_rows(rows: dict[str, list[tuple[object, ...]]]) -> None:
    for predicate, values in rows.items():
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate Datalog facts for {predicate}")


def _require_arity(
    rows: dict[str, list[tuple[object, ...]]],
    predicate: str,
    arity: int,
) -> None:
    if any(len(row) != arity for row in rows[predicate]):
        raise ValueError(f"invalid {predicate} arity")
