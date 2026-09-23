"""Lossless canonical T/N/E/P record codec shared by every ASCII layout."""

from __future__ import annotations

from typing import TypeVar

from evals.catan_board_bench.ascii_variations.codec import (
    _csv,
    _mapping,
    _nullable,
    _parse_csv,
    _parse_int_csv,
    _parse_mapping,
    _parse_nullable,
    _parse_nullable_int,
)
from evals.catan_board_bench.ascii_variations.facts import (
    FullEdge,
    FullFacts,
    FullNode,
    FullPort,
    FullTile,
)
from evals.catan_board_bench.ascii_variations.graph import canonicalize_full_facts
from evals.catan_board_bench.ascii_variations.schema import (
    CORNER_ORDER,
    FACT_SCHEMA,
    RECORD_KINDS,
    SIDE_ORDER,
)

_Record = TypeVar("_Record")


def parse_ascii_variant(text: str) -> FullFacts:
    """Parse canonical record lines embedded in every lossless variation."""

    tiles: dict[str, FullTile] = {}
    nodes: dict[str, FullNode] = {}
    edges: dict[str, FullEdge] = {}
    ports: dict[str, FullPort] = {}
    for line in text.splitlines():
        if len(line) < 3 or line[0] not in RECORD_KINDS or line[1] != "|":
            continue
        kind, item_id, fields = _record_parts(line)
        if kind == "T":
            _keep_unique(tiles, kind, item_id, _tile_record(item_id, fields))
        elif kind == "N":
            _keep_unique(nodes, kind, item_id, _node_record(item_id, fields))
        elif kind == "E":
            _keep_unique(edges, kind, item_id, _edge_record(item_id, fields))
        else:
            _keep_unique(ports, kind, item_id, _port_record(item_id, fields))

    facts = canonicalize_full_facts(
        {
            "schema": FACT_SCHEMA,
            "tiles": list(tiles.values()),
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
            "ports": list(ports.values()),
        }
    )
    counts = (
        len(facts["tiles"]),
        len(facts["nodes"]),
        len(facts["edges"]),
        len(facts["ports"]),
    )
    if counts != (19, 54, 72, 9):
        raise ValueError(f"incomplete full graph records: {counts}")
    return facts


def _keep_unique(bucket: dict[str, _Record], kind: str, item_id: str, parsed: _Record) -> None:
    existing = bucket.get(item_id)
    if existing is not None and existing != parsed:
        raise ValueError(f"conflicting duplicate record {kind}/{item_id}")
    bucket[item_id] = parsed


def fact_record_lines(facts: FullFacts) -> list[str]:
    lines: list[str] = []
    for tile in facts["tiles"]:
        lines.append(
            "|".join(
                (
                    "T",
                    tile["id"],
                    f"cube={_csv(tile['cube'])}",
                    f"resource={tile['resource']}",
                    f"number={_nullable(tile['number'])}",
                    f"robber={int(tile['robber'])}",
                    f"corners={_mapping(tile['corners'], CORNER_ORDER)}",
                    f"sides={_mapping(tile['sides'], SIDE_ORDER)}",
                )
            )
        )
    for node in facts["nodes"]:
        lines.append(
            "|".join(
                (
                    "N",
                    node["id"],
                    f"color={_nullable(node['color'])}",
                    f"building={_nullable(node['building'])}",
                    f"tiles={_csv(node['tiles'])}",
                    f"edges={_csv(node['edges'])}",
                    f"ports={_csv(node['ports'])}",
                )
            )
        )
    for edge in facts["edges"]:
        lines.append(
            "|".join(
                (
                    "E",
                    edge["id"],
                    f"nodes={_csv(edge['nodes'])}",
                    f"road={_nullable(edge['road'])}",
                    f"tiles={_csv(edge['tiles'])}",
                )
            )
        )
    for port in facts["ports"]:
        lines.append(
            "|".join(
                (
                    "P",
                    port["id"],
                    f"cube={_csv(port['cube'])}",
                    f"direction={port['direction']}",
                    f"resource={port['resource']}",
                    f"ratio={port['ratio']}",
                    f"nodes={_csv(port['nodes'])}",
                )
            )
        )
    return lines


def _parse_record(line: str) -> FullTile | FullNode | FullEdge | FullPort:
    kind, item_id, fields = _record_parts(line)
    if kind == "T":
        return _tile_record(item_id, fields)
    if kind == "N":
        return _node_record(item_id, fields)
    if kind == "E":
        return _edge_record(item_id, fields)
    if kind == "P":
        return _port_record(item_id, fields)
    raise ValueError(f"unknown record kind: {kind}")


def _record_parts(line: str) -> tuple[str, str, dict[str, str]]:
    parts = line.split("|")
    kind = parts[0]
    item_id = parts[1]
    fields = {}
    for part in parts[2:]:
        key, value = part.split("=", 1)
        fields[key] = value
    return kind, item_id, fields


def _tile_record(item_id: str, fields: dict[str, str]) -> FullTile:
    return {
        "id": item_id,
        "cube": _parse_int_csv(fields["cube"]),
        "resource": fields["resource"],
        "number": _parse_nullable_int(fields["number"]),
        "robber": fields["robber"] == "1",
        "corners": _parse_mapping(fields["corners"]),
        "sides": _parse_mapping(fields["sides"]),
    }


def _node_record(item_id: str, fields: dict[str, str]) -> FullNode:
    return {
        "id": item_id,
        "color": _parse_nullable(fields["color"]),
        "building": _parse_nullable(fields["building"]),
        "tiles": _parse_csv(fields["tiles"]),
        "edges": _parse_csv(fields["edges"]),
        "ports": _parse_csv(fields["ports"]),
    }


def _edge_record(item_id: str, fields: dict[str, str]) -> FullEdge:
    return {
        "id": item_id,
        "nodes": _parse_csv(fields["nodes"]),
        "road": _parse_nullable(fields["road"]),
        "tiles": _parse_csv(fields["tiles"]),
    }


def _port_record(item_id: str, fields: dict[str, str]) -> FullPort:
    return {
        "id": item_id,
        "cube": _parse_int_csv(fields["cube"]),
        "direction": fields["direction"],
        "resource": fields["resource"],
        "ratio": fields["ratio"],
        "nodes": _parse_csv(fields["nodes"]),
    }
