"""The line-oriented graph DSL representation and its parser."""

from __future__ import annotations

from evals.catan_board_bench.text_representations.schema import (
    FACT_SCHEMA,
    BoardFacts,
    _coord,
    _join,
    _parse_coord,
)


def render_graph_dsl(facts: BoardFacts) -> str:
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


def parse_graph_dsl(text: str) -> BoardFacts:
    facts: BoardFacts = {
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

