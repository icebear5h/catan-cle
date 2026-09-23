"""Material-board validation and detached state decoding."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from functools import lru_cache
from typing import cast

from cle.game_engine.models.board import Board
from cle.game_engine.models.board.graph import NodeGraph
from cle.game_engine.models.coordinate_system import Coordinate
from cle.game_engine.models.enums import CITY, RESOURCES, SETTLEMENT, FastBuildingType, FastResource
from cle.game_engine.models.map import CatanMap
from cle.game_engine.models.player import Color
from cle.game_engine.public_board import PUBLIC_BOARD_CONTRACT_SCHEMA
from data_pipeline.board_recognition.full_board_readout import board_answer
from evals.catan_board_bench.tokens import base_catan_map
from sft.board.symbolic_board_tasks._constants import (
    RESOURCES_LOWER,
    PhysicalStateError,
    _keys,
    _require,
)
from sft.board.symbolic_board_tasks._geometry import (
    _atlas,
    _participants,
    _rows,
    _same,
    _token,
)
from sft.board.symbolic_board_tasks._types import Atlas, DecodedState, StatePayload
from sft.json_types import (
    JsonDict,
    JsonLikeDict,
    JsonList,
    as_dict,
    as_int,
    as_list,
    as_str,
)


def validate_contract(contract: JsonDict) -> StatePayload:
    """Validate exact canonical identity, redundant topology/facts and physical state.

    This certifies the supplied material board, not its reachable game history.
    Historical award/length caches are deliberately neither inputs nor authority.
    """
    _require(isinstance(contract, dict) and contract.get("schema") == PUBLIC_BOARD_CONTRACT_SCHEMA,
             "invalid public board schema")
    if "identity_space" in contract:
        _same(contract["identity_space"], "canonical_engine_ids", "identity space")
    players = contract.get("players")
    _require(isinstance(players, list) and all(isinstance(p, dict) for p in players),
             "invalid players")
    player_rows = [cast("JsonDict", p) for p in cast("JsonList", players)]
    colors = _participants([p.get("color") for p in player_rows])
    atlas: Atlas = _atlas()
    for family in ("tiles", "nodes", "edges", "ports"):
        gold_rows = [as_dict(entry) for entry in as_list(as_dict(atlas["raw"])[family])]
        for row, gold in zip(_rows(contract, family, gold_rows), gold_rows, strict=True):
            token = as_str(gold["token"])
            expected: JsonLikeDict = {}
            if family == "tiles":
                ns = sorted(as_int(v) for v in as_dict(gold["nodes"]).values())
                es = sorted([as_int(x) for x in as_list(v)]
                            for v in as_dict(gold["edges"]).values())
                expected = dict(coord=gold["coord"], nodes=ns, edges=es,
                                node_tokens=[f"<N{n:02d}>" for n in ns],
                                edge_tokens=[f"<E{a:02d}_{b:02d}>" for a, b in es])
                _require(type(row.get("has_robber")) is bool, "invalid robber flag")
                _require(row.get("resource") is None or row["resource"] in RESOURCES,
                         "invalid tile resource")
                resource = row.get("resource")
                number = row.get("number")
                _require((resource is None and number is None) or
                         (resource is not None and type(number) is int
                          and number in (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)),
                         "invalid typed tile number")
                expected["resource_token"] = "<DESERT>" if resource is None else f"<{resource}>"
            elif family == "nodes":
                expected = dict(
                    adjacent_tiles=sorted(int(t[2:-1]) for t in atlas["node_tiles"][token]),
                    adjacent_tile_tokens=sorted(atlas["node_tiles"][token]),
                    adjacent_edges=sorted([int(n[2:-1]) for n in atlas["edges"][e]]
                                          for e in atlas["node_edges"][token]),
                    adjacent_edge_tokens=sorted(atlas["node_edges"][token]),
                    port_ids=sorted(int(p[2:-1]) for p in atlas["node_ports"].get(token, [])),
                    port_tokens=sorted(atlas["node_ports"].get(token, [])),
                )
                _require({"building", "color"} <= row.keys(), "missing building ownership")
                owner, piece = row["color"], row["building"]
                _require((owner is None and piece is None) or
                         (owner in colors and piece in (SETTLEMENT, CITY)), "invalid building ownership")
                expected.update(color_token=f"<{owner}>" if owner else None,
                                building_token=f"<{piece}>" if piece else None)
            elif family == "edges":
                expected = dict(nodes=gold["id"], node_tokens=list(atlas["edges"][token]))
                _require("road_color" in row and (row["road_color"] is None or row["road_color"] in colors),
                         "invalid road ownership")
                owner = row["road_color"]
                expected["road_color_token"] = f"<{owner}>" if owner else None
            else:
                resource = row.get("resource")
                _require(resource is None or resource in RESOURCES, "invalid port resource")
                attached = [as_int(n) for n in as_list(gold["attached_nodes"])]
                expected = dict(coord=gold["coord"], direction=gold["direction"],
                                attached_nodes=attached,
                                attached_node_tokens=[f"<N{n:02d}>" for n in attached],
                                kind="generic" if resource is None else "resource",
                                ratio="3:1" if resource is None else "2:1",
                                resource_token=None if resource is None else f"<{resource}>")
            for key, value in expected.items():
                _same(row.get(key), value, f"{token} {key}")
    robber = contract.get("robber")
    _require(isinstance(robber, dict), "missing robber")
    robber_row = cast("JsonDict", robber)
    marked = [t for t in (as_dict(e) for e in as_list(contract["tiles"])) if t["has_robber"]]
    _require(len(marked) == 1, "expected one robber")
    for key, value in (("tile_id", marked[0]["id"]), ("tile_token", marked[0]["token"]),
                       ("coord", marked[0]["coord"])):
        _same(robber_row.get(key), value, "robber " + key)
    state = StatePayload(board=board_answer(contract), colors=list(colors))
    values = decode_state(state)
    for player in player_rows:
        color = player["color"]
        for field, count in (
            ("settlement_count", sum(v == (color, "settlement") for v in values["buildings"].values())),
            ("city_count", sum(v == (color, "city") for v in values["buildings"].values())),
            ("road_count", sum(v == color for v in values["roads"].values())),
        ):
            if field in player:
                _same(player[field], count, "player " + field)
        if "color_token" in player:
            _same(player["color_token"], f"<{color}>", "player color token")
    return state


@lru_cache(maxsize=256)
def _decode(board: str, colors: tuple[str, ...]) -> DecodedState:
    keys = [t for f in "TNEP" for t in sorted(_atlas()["tokens"]) if t[1] == f] + ["robber"]
    entries = board.split(";")
    _require(len(entries) == 155, "board must contain exactly 155 entries")
    values: dict[str, str] = {}
    for key, entry in zip(keys, entries, strict=True):
        parts = entry.strip().split(maxsplit=1)
        _require(len(parts) == 2 and parts[0] == key, "board identity/order/duplicate error")
        values[key] = parts[1]
    result = DecodedState(tiles={}, ports={}, buildings={}, roads={}, colors=colors,
                          robber=_token(values["robber"], "T"))
    color_names = {c.lower().replace("_", " "): c for c in colors}
    for token, value in values.items():
        if token.startswith("<T"):
            parts = value.split()
            _require(len(parts) == 2 and parts[0] in RESOURCES_LOWER, "invalid terrain")
            resource, number = parts
            _require((resource == "desert" and number == "none") or
                     (resource != "desert" and number in {"2", "3", "4", "5", "6", "8", "9", "10", "11", "12"}),
                     "invalid terrain number")
            result["tiles"][token] = (resource, None if number == "none" else int(number))
        elif token.startswith("<P"):
            _require(value in {"3:1 port", *(r + " port" for r in RESOURCES_LOWER - {"desert"})},
                     "invalid port")
            result["ports"][token] = value.removesuffix(" port")
        elif token.startswith(("<N", "<E")) and value != "empty":
            parts = value.rsplit(" ", 1)
            _require(len(parts) == 2 and parts[0] in color_names, "invalid piece color")
            color, piece = color_names[parts[0]], parts[1]
            if token[1] == "N":
                _require(piece in ("settlement", "city"), "invalid node piece")
                result["buildings"][token] = (color, piece)
            else:
                _require(piece == "road", "invalid edge piece")
                result["roads"][token] = color
    _require(sum(r == "desert" for r, _ in result["tiles"].values()) == 1, "expected one desert")
    for node in result["buildings"]:
        if _atlas()["graph"][node] & result["buildings"].keys():
            raise PhysicalStateError(f"building_distance_conflict:{node}")
    for color in colors:
        for piece, cap in (("settlement", 5), ("city", 4), ("road", 15)):
            count = (sum(owner == color for owner in result["roads"].values()) if piece == "road"
                     else sum(v == (color, piece) for v in result["buildings"].values()))
            if count > cap:
                raise PhysicalStateError(f"piece_supply:{color}:{piece}:{count}>{cap}")
    return result


def decode_state(state: Mapping[str, object]) -> DecodedState:
    """Strictly parse a complete minimal state, returning detached facts."""
    _keys(state, {"board", "colors"}, "state")
    _require(isinstance(state["board"], str), "board must be text")
    return copy.deepcopy(_decode(cast("str", state["board"]), _participants(state["colors"])))


def detached_board(state: Mapping[str, object]) -> Board:
    """Reconstruct fresh map, bidirectional roads, distance cache and road components."""
    data = decode_state(state)
    board_map = base_catan_map()
    for tile in board_map.tiles_by_id.values():
        resource, number = data["tiles"][f"<T{tile.id:02d}>"]
        tile.resource = None if resource == "desert" else cast("FastResource", resource.upper())
        tile.number = number
    for port in board_map.ports_by_id.values():
        resource = data["ports"][f"<P{port.id:02d}>"]
        port.resource = None if resource == "3:1" else cast("FastResource", resource.upper())
    board = Board(CatanMap.from_tiles(board_map.tiles))
    board.buildable_subgraph = cast("NodeGraph", board.buildable_subgraph).copy()
    robber_tile = next(t for t in (as_dict(e) for e in as_list(as_dict(_atlas()["raw"])["tiles"]))
                       if t["token"] == data["robber"])
    board.robber_coordinate = cast(
        "Coordinate", tuple(as_int(c) for c in as_list(robber_tile["coord"]))
    )
    for node, (color, piece) in data["buildings"].items():
        nid = int(node[2:-1])
        board.buildings[nid] = (Color(color), cast("FastBuildingType", piece.upper()))
        board.board_buildable_ids.difference_update({nid, *board.buildable_subgraph.neighbors(nid)})
    for edge, color in data["roads"].items():
        a, b = (int(n[2:-1]) for n in _atlas()["edges"][edge])
        board.roads[a, b] = board.roads[b, a] = Color(color)
    board.recompute_road_state()
    return board
