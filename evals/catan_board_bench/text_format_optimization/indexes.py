"""Question-independent query indexes derived from the canonical graph."""

from __future__ import annotations

from typing import Sequence, TypedDict

from cle.game_engine.models.coordinate_system import UNIT_VECTORS
from cle.game_engine.models.player import Color
from evals.catan_board_bench.ascii_variations import (
    CORNER_ORDER,
    DIRECTION_ORDER,
    SCREEN_DIRECTIONS,
)
from evals.catan_board_bench.ascii_variations.facts import FullFacts, FullNode, FullTile


class NodeState(TypedDict):
    node: str
    color: str | None
    building: str | None


class OccupiedCorner(NodeState):
    units: int


class RollSource(TypedDict):
    tile: str
    resource: str
    robber: bool
    occupied_corners: list[OccupiedCorner]


class PlayerHoldings(TypedDict):
    settlements: list[str]
    cities: list[str]
    roads: list[str]


class QueryIndexes(TypedDict):
    tile_neighbors: dict[str, dict[str, str | None]]
    tile_corners: dict[str, dict[str, NodeState]]
    port_nodes: dict[str, list[NodeState]]
    players: dict[str, PlayerHoldings]
    roll_tiles: dict[str, list[str]]
    roll_sources: dict[str, list[RollSource]]


def build_query_indexes(facts: FullFacts) -> QueryIndexes:
    """Build deterministic query indexes without precomputing scored answers."""

    tile_by_cube = {tuple(tile["cube"]): tile for tile in facts["tiles"]}
    nodes = {node["id"]: node for node in facts["nodes"]}

    tile_neighbors: dict[str, dict[str, str | None]] = {}
    tile_corners: dict[str, dict[str, NodeState]] = {}
    for tile in facts["tiles"]:
        origin = tile["cube"]
        tile_neighbors[tile["id"]] = {
            SCREEN_DIRECTIONS[direction]: _tile_id_or_none(
                tile_by_cube.get(
                    tuple(origin[index] + UNIT_VECTORS[direction][index] for index in range(3))
                )
            )
            for direction in DIRECTION_ORDER
        }
        tile_corners[tile["id"]] = {
            corner: _node_state(nodes[tile["corners"][corner]]) for corner in CORNER_ORDER
        }

    port_nodes = {
        port["id"]: [_node_state(nodes[node_id]) for node_id in port["nodes"]]
        for port in facts["ports"]
    }

    colors = sorted(
        {color.value for color in Color}
        | {color for node in facts["nodes"] if (color := node["color"])}
        | {road for edge in facts["edges"] if (road := edge["road"])}
    )
    players: dict[str, PlayerHoldings] = {}
    for color in colors:
        players[color] = {
            "settlements": sorted(
                node["id"]
                for node in facts["nodes"]
                if node["color"] == color and node["building"] == "SETTLEMENT"
            ),
            "cities": sorted(
                node["id"]
                for node in facts["nodes"]
                if node["color"] == color and node["building"] == "CITY"
            ),
            "roads": sorted(edge["id"] for edge in facts["edges"] if edge["road"] == color),
        }

    roll_tiles = {
        str(number): sorted(
            tile["id"] for tile in facts["tiles"] if tile["number"] == number
        )
        for number in (*range(2, 7), *range(8, 13))
    }
    roll_sources: dict[str, list[RollSource]] = {
        str(number): [
            {
                "tile": tile["id"],
                "resource": tile["resource"],
                "robber": tile["robber"],
                "occupied_corners": [
                    {
                        **state,
                        "units": 1 if state["building"] == "SETTLEMENT" else 2,
                    }
                    for state in tile_corners[tile["id"]].values()
                    if state["building"] is not None
                ],
            }
            for tile in sorted(facts["tiles"], key=lambda value: value["id"])
            if tile["number"] == number
        ]
        for number in (*range(2, 7), *range(8, 13))
    }
    return {
        "tile_neighbors": tile_neighbors,
        "tile_corners": tile_corners,
        "port_nodes": port_nodes,
        "players": players,
        "roll_tiles": roll_tiles,
        "roll_sources": roll_sources,
    }



def _append_record_indexes(base: str, facts: FullFacts) -> str:
    return "\n".join(
        (
            base,
            "",
            "QUERY INDEXES (DERIVED; PLAYER COUNTS AND AGGREGATED PAYOUTS ARE NOT PRECOMPUTED)",
            "QI RULE|PORT_NODES: occupants are only endpoints whose building is not '-'",
            "QI RULE|ROLL_SOURCE: ignore blocked=1 terms; otherwise sum units by color/resource",
            *_query_index_lines(facts),
        )
    )


def _query_index_lines(facts: FullFacts) -> list[str]:
    indexes = build_query_indexes(facts)
    lines: list[str] = []
    for tile_id, neighbors in indexes["tile_neighbors"].items():
        lines.append(
            "QI|TILE_NEIGHBORS|"
            + tile_id
            + "|"
            + "|".join(f"{key}={value or '-'}" for key, value in neighbors.items())
        )
    for tile_id, corners in indexes["tile_corners"].items():
        lines.append(
            "QI|TILE_CORNERS|"
            + tile_id
            + "|"
            + "|".join(
                f"{corner}={state['node']}/{state['color'] or '-'}/{state['building'] or '-'}"
                for corner, state in corners.items()
            )
        )
    for port_id, states in indexes["port_nodes"].items():
        lines.append(
            "QI|PORT_NODES|"
            + port_id
            + "|"
            + "|".join(
                f"{state['node']}/{state['color'] or '-'}/{state['building'] or '-'}"
                for state in states
            )
        )
    for color, owned in indexes["players"].items():
        lines.append(
            "QI|PLAYER|"
            + color
            + f"|settlements={_csv_or_dash(owned['settlements'])}"
            + f"|cities={_csv_or_dash(owned['cities'])}"
            + f"|roads={_csv_or_dash(owned['roads'])}"
        )
    for roll, tiles in indexes["roll_tiles"].items():
        lines.append(f"QI|ROLL_TILES|{roll}|tiles={_csv_or_dash(tiles)}")
    for roll, sources in indexes["roll_sources"].items():
        for source in sources:
            common = (
                f"QI|ROLL_SOURCE|{roll}|tile={source['tile']}"
                f"|resource={source['resource']}|blocked={int(source['robber'])}"
            )
            if not source["occupied_corners"]:
                lines.append(common + "|node=-|color=-|building=-|units=0")
                continue
            for state in source["occupied_corners"]:
                lines.append(
                    common
                    + f"|node={state['node']}|color={state['color']}"
                    + f"|building={state['building']}|units={state['units']}"
                )
    return lines


def _node_state(node: FullNode) -> NodeState:
    return {"node": node["id"], "color": node["color"], "building": node["building"]}


def _tile_id_or_none(tile: FullTile | None) -> str | None:
    return None if tile is None else tile.get("id")


def _csv_or_dash(values: Sequence[str]) -> str:
    return ",".join(values) if values else "-"

