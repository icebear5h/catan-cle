"""The spatial ASCII representation and its parser."""

from __future__ import annotations

import re
from typing import Optional

from evals.catan_board_bench.text_representations.schema import (
    FACT_SCHEMA,
    BoardFacts,
    BoardFactTile,
    _coord,
    _join,
)


def render_spatial_ascii(facts: BoardFacts) -> str:
    rows: dict[int, list[BoardFactTile]] = {}
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


def parse_spatial_ascii(text: str) -> BoardFacts:
    lines = text.splitlines()
    default_match = re.fullmatch(r"DEFAULT NODE=(\w+) EDGE=(\w+)", lines[1])
    if default_match is None:
        raise ValueError("Missing ASCII defaults")
    facts: BoardFacts = {
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
            port_match = port_pattern.fullmatch(line)
            if port_match is None:
                raise ValueError(f"Invalid ASCII port line: {line}")
            facts["ports"].append(
                {
                    "token": port_match.group(1),
                    "coord": [int(port_match.group(i)) for i in (2, 3, 4)],
                    "direction": port_match.group(5),
                    "resource": port_match.group(6),
                    "ratio": port_match.group(7),
                    "nodes": [port_match.group(8), port_match.group(9)],
                }
            )
    return facts


