"""Lossless text representations of target-neutral public Catan board facts."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Dict, Iterable, Optional, Sequence


JsonDict = Dict[str, Any]
REPRESENTATION_NAMES = (
    "verbose_json",
    "compact_json",
    "graph_dsl",
    "spatial_ascii",
)
FACT_SCHEMA = "catan_public_board_facts/v1"


def public_board_facts(contract: JsonDict) -> JsonDict:
    """Normalize the common fact set needed by the visual question suite.

    Nodes and edges are sparse dynamic state. Explicit defaults make omission
    equivalent to EMPTY while avoiding full fixed-atlas topology in only one
    representation.
    """
    tiles = [
        {
            "token": tile["token"],
            "coord": list(tile["coord"]),
            "resource": tile.get("resource") or "DESERT",
            "number": tile.get("number"),
            "has_robber": bool(tile.get("has_robber")),
        }
        for tile in contract["tiles"]
    ]
    nodes = [
        {
            "token": node["token"],
            "color": node["color"],
            "building": node["building"],
        }
        for node in contract["nodes"]
        if node.get("building") is not None
    ]
    edges = [
        {
            "token": edge["token"],
            "nodes": list(edge["node_tokens"]),
            "road_color": edge["road_color"],
        }
        for edge in contract["edges"]
        if edge.get("road_color") is not None
    ]
    ports = [
        {
            "token": port["token"],
            "coord": list(port["coord"]),
            "direction": port["direction"],
            "resource": port.get("resource") or "GENERIC",
            "ratio": port["ratio"],
            "nodes": list(port["attached_node_tokens"]),
        }
        for port in contract["ports"]
    ]
    return canonicalize_facts(
        {
            "schema": FACT_SCHEMA,
            "defaults": {"node": "EMPTY", "edge": "EMPTY"},
            "tiles": tiles,
            "nodes": nodes,
            "edges": edges,
            "ports": ports,
        }
    )


def canonicalize_facts(facts: JsonDict) -> JsonDict:
    return {
        "schema": facts["schema"],
        "defaults": {
            "node": facts["defaults"]["node"],
            "edge": facts["defaults"]["edge"],
        },
        "tiles": sorted(facts["tiles"], key=lambda item: item["token"]),
        "nodes": sorted(facts["nodes"], key=lambda item: item["token"]),
        "edges": sorted(facts["edges"], key=lambda item: item["token"]),
        "ports": sorted(facts["ports"], key=lambda item: item["token"]),
    }


def fact_digest(facts: JsonDict) -> str:
    encoded = json.dumps(
        canonicalize_facts(facts),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_representation(name: str, facts: JsonDict) -> str:
    try:
        renderer = _RENDERERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown representation: {name}") from exc
    return renderer(canonicalize_facts(facts))


def parse_representation(name: str, text: str) -> JsonDict:
    try:
        parser = _PARSERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown representation: {name}") from exc
    return canonicalize_facts(parser(text))


def render_verbose_json(facts: JsonDict) -> str:
    return json.dumps(facts, indent=2, sort_keys=False)


def parse_verbose_json(text: str) -> JsonDict:
    return json.loads(text)


def render_compact_json(facts: JsonDict) -> str:
    payload = {
        "v": 1,
        "d": [facts["defaults"]["node"], facts["defaults"]["edge"]],
        "legend": {
            "t": "[tile,coord,resource,number,robber]",
            "n": "[node,color,building]",
            "e": "[edge,nodes,color]",
            "p": "[port,coord,direction,resource,ratio,nodes]",
        },
        "t": [
            [
                tile["token"],
                tile["coord"],
                tile["resource"],
                tile["number"],
                int(tile["has_robber"]),
            ]
            for tile in facts["tiles"]
        ],
        "n": [[node["token"], node["color"], node["building"]] for node in facts["nodes"]],
        "e": [[edge["token"], edge["nodes"], edge["road_color"]] for edge in facts["edges"]],
        "p": [
            [
                port["token"],
                port["coord"],
                port["direction"],
                port["resource"],
                port["ratio"],
                port["nodes"],
            ]
            for port in facts["ports"]
        ],
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=False)


def parse_compact_json(text: str) -> JsonDict:
    payload = json.loads(text)
    return {
        "schema": FACT_SCHEMA,
        "defaults": {"node": payload["d"][0], "edge": payload["d"][1]},
        "tiles": [
            {
                "token": item[0],
                "coord": item[1],
                "resource": item[2],
                "number": item[3],
                "has_robber": bool(item[4]),
            }
            for item in payload["t"]
        ],
        "nodes": [
            {"token": item[0], "color": item[1], "building": item[2]} for item in payload["n"]
        ],
        "edges": [
            {"token": item[0], "nodes": item[1], "road_color": item[2]} for item in payload["e"]
        ],
        "ports": [
            {
                "token": item[0],
                "coord": item[1],
                "direction": item[2],
                "resource": item[3],
                "ratio": item[4],
                "nodes": item[5],
            }
            for item in payload["p"]
        ],
    }


def render_graph_dsl(facts: JsonDict) -> str:
    lines = [
        "CATAN_GRAPH_V1",
        f"DEFAULT|NODE|{facts['defaults']['node']}",
        f"DEFAULT|EDGE|{facts['defaults']['edge']}",
    ]
    lines.extend(
        "|".join(
            (
                "T",
                tile["token"],
                _coord(tile["coord"]),
                tile["resource"],
                "-" if tile["number"] is None else str(tile["number"]),
                "1" if tile["has_robber"] else "0",
            )
        )
        for tile in facts["tiles"]
    )
    lines.extend(f"N|{node['token']}|{node['color']}|{node['building']}" for node in facts["nodes"])
    lines.extend(
        f"E|{edge['token']}|{_join(edge['nodes'])}|{edge['road_color']}" for edge in facts["edges"]
    )
    lines.extend(
        "|".join(
            (
                "P",
                port["token"],
                _coord(port["coord"]),
                port["direction"],
                port["resource"],
                port["ratio"],
                _join(port["nodes"]),
            )
        )
        for port in facts["ports"]
    )
    return "\n".join(lines)


def parse_graph_dsl(text: str) -> JsonDict:
    facts: JsonDict = {
        "schema": FACT_SCHEMA,
        "defaults": {},
        "tiles": [],
        "nodes": [],
        "edges": [],
        "ports": [],
    }
    for line in text.splitlines()[1:]:
        parts = line.split("|")
        kind = parts[0]
        if kind == "DEFAULT":
            facts["defaults"][parts[1].lower()] = parts[2]
        elif kind == "T":
            facts["tiles"].append(
                {
                    "token": parts[1],
                    "coord": _parse_coord(parts[2]),
                    "resource": parts[3],
                    "number": None if parts[4] == "-" else int(parts[4]),
                    "has_robber": parts[5] == "1",
                }
            )
        elif kind == "N":
            facts["nodes"].append({"token": parts[1], "color": parts[2], "building": parts[3]})
        elif kind == "E":
            facts["edges"].append(
                {
                    "token": parts[1],
                    "nodes": parts[2].split(","),
                    "road_color": parts[3],
                }
            )
        elif kind == "P":
            facts["ports"].append(
                {
                    "token": parts[1],
                    "coord": _parse_coord(parts[2]),
                    "direction": parts[3],
                    "resource": parts[4],
                    "ratio": parts[5],
                    "nodes": parts[6].split(","),
                }
            )
    return facts


def render_spatial_ascii(facts: JsonDict) -> str:
    rows: dict[int, list[JsonDict]] = {}
    for tile in facts["tiles"]:
        rows.setdefault(int(tile["coord"][2]), []).append(tile)

    lines = [
        "CATAN BOARD — LAND ROWS TOP TO BOTTOM",
        f"DEFAULT NODE={facts['defaults']['node']} EDGE={facts['defaults']['edge']}",
        "Tile cell: token@cube:resource/number; *=robber",
    ]
    max_count = max(len(row) for row in rows.values())
    for row_number, z in enumerate(sorted(rows)):
        row = sorted(rows[z], key=lambda tile: tile["coord"][0])
        indent = "    " * (max_count - len(row))
        cells = []
        for tile in row:
            number = "-" if tile["number"] is None else str(tile["number"])
            robber = "*" if tile["has_robber"] else ""
            cells.append(
                f"{tile['token']}@{_coord(tile['coord'])}:{tile['resource']}/{number}{robber}"
            )
        lines.append(f"ROW{row_number} {indent}" + "  ".join(cells))

    lines.append("\nOCCUPIED NODES")
    lines.append(
        "; ".join(f"{node['token']}={node['color']}/{node['building']}" for node in facts["nodes"])
        or "NONE"
    )
    lines.append("\nROADS")
    lines.append(
        "; ".join(
            f"{edge['token']}({_join(edge['nodes'])})={edge['road_color']}"
            for edge in facts["edges"]
        )
        or "NONE"
    )
    lines.append("\nPORTS")
    lines.extend(
        f"{port['token']}@{_coord(port['coord'])}:{port['direction']}="
        f"{port['resource']}/{port['ratio']} nodes={_join(port['nodes'])}"
        for port in facts["ports"]
    )
    return "\n".join(lines)


def parse_spatial_ascii(text: str) -> JsonDict:
    lines = text.splitlines()
    default_match = re.fullmatch(r"DEFAULT NODE=(\w+) EDGE=(\w+)", lines[1])
    if default_match is None:
        raise ValueError("Missing ASCII defaults")
    facts: JsonDict = {
        "schema": FACT_SCHEMA,
        "defaults": {"node": default_match.group(1), "edge": default_match.group(2)},
        "tiles": [],
        "nodes": [],
        "edges": [],
        "ports": [],
    }
    section: Optional[str] = None
    tile_pattern = re.compile(
        r"(<T\d{2}>)@\((-?\d+),(-?\d+),(-?\d+)\):"
        r"([A-Z]+)/(\d+|-)(\*)?"
    )
    node_pattern = re.compile(r"(<N\d{2}>)=([A-Z_]+)/([A-Z_]+)")
    edge_pattern = re.compile(r"(<E\d{2}_\d{2}>)\((<N\d{2}>),(<N\d{2}>)\)=([A-Z_]+)")
    port_pattern = re.compile(
        r"(<P\d{2}>)@\((-?\d+),(-?\d+),(-?\d+)\):([A-Z]+)="
        r"([A-Z]+)/([23]:1) nodes=(<N\d{2}>),(<N\d{2}>)"
    )
    for line in lines[3:]:
        if line == "OCCUPIED NODES":
            section = "nodes"
            continue
        if line == "ROADS":
            section = "edges"
            continue
        if line == "PORTS":
            section = "ports"
            continue
        if line.startswith("ROW"):
            for match in tile_pattern.finditer(line):
                facts["tiles"].append(
                    {
                        "token": match.group(1),
                        "coord": [int(match.group(i)) for i in (2, 3, 4)],
                        "resource": match.group(5),
                        "number": None if match.group(6) == "-" else int(match.group(6)),
                        "has_robber": bool(match.group(7)),
                    }
                )
        elif section == "nodes" and line != "NONE" and line:
            for match in node_pattern.finditer(line):
                facts["nodes"].append(
                    {
                        "token": match.group(1),
                        "color": match.group(2),
                        "building": match.group(3),
                    }
                )
        elif section == "edges" and line != "NONE" and line:
            for match in edge_pattern.finditer(line):
                facts["edges"].append(
                    {
                        "token": match.group(1),
                        "nodes": [match.group(2), match.group(3)],
                        "road_color": match.group(4),
                    }
                )
        elif section == "ports" and line:
            match = port_pattern.fullmatch(line)
            if match is None:
                raise ValueError(f"Invalid ASCII port line: {line}")
            facts["ports"].append(
                {
                    "token": match.group(1),
                    "coord": [int(match.group(i)) for i in (2, 3, 4)],
                    "direction": match.group(5),
                    "resource": match.group(6),
                    "ratio": match.group(7),
                    "nodes": [match.group(8), match.group(9)],
                }
            )
    return facts


def representation_metrics(facts: JsonDict) -> dict[str, dict[str, int]]:
    return {
        name: {
            "characters": len(render_representation(name, facts)),
            "lines": render_representation(name, facts).count("\n") + 1,
        }
        for name in REPRESENTATION_NAMES
    }


def _coord(values: Sequence[int]) -> str:
    return "(" + ",".join(str(value) for value in values) + ")"


def _parse_coord(value: str) -> list[int]:
    return [int(part) for part in value.strip("()").split(",")]


def _join(values: Iterable[str]) -> str:
    return ",".join(values)


_RENDERERS: dict[str, Callable[[JsonDict], str]] = {
    "verbose_json": render_verbose_json,
    "compact_json": render_compact_json,
    "graph_dsl": render_graph_dsl,
    "spatial_ascii": render_spatial_ascii,
}
_PARSERS: dict[str, Callable[[str], JsonDict]] = {
    "verbose_json": parse_verbose_json,
    "compact_json": parse_compact_json,
    "graph_dsl": parse_graph_dsl,
    "spatial_ascii": parse_spatial_ascii,
}
