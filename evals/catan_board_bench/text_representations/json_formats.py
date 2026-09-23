"""Verbose and compact JSON representations of public board facts."""

from __future__ import annotations

import json

from evals.catan_board_bench.text_representations.schema import (
    FACT_SCHEMA,
    BoardFacts,
)


def render_verbose_json(facts: BoardFacts) -> str:
    return json.dumps(facts, indent=2, sort_keys=False)


def parse_verbose_json(text: str) -> BoardFacts:
    parsed: BoardFacts = json.loads(text)
    return parsed


def render_compact_json(facts: BoardFacts) -> str:
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


def parse_compact_json(text: str) -> BoardFacts:
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

