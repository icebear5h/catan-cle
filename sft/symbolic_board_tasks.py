"""Text-only atlas tasks, detached engine oracles, and recomputing strict scoring.

Metadata target is exactly ``{"state": state_or_null, "query": query}``.
A state contains only ``board`` (the canonical 155-entry board_answer string)
and four participant ``colors``. Geometry lives in this module, never in model
input. Known malformed targets raise ValueError; cached expected text is ignored.
Settlement and edge-simple longest-trail tasks are compositional transfer only.
"""

from __future__ import annotations

import copy
import json
from collections import deque
from functools import lru_cache

from cle.game_engine.models.board import Board, STATIC_GRAPH
from cle.game_engine.models.enums import CITY, RESOURCES, SETTLEMENT
from cle.game_engine.models.map import CatanMap
from cle.game_engine.models.player import Color
from cle.game_engine.public_board import PUBLIC_BOARD_CONTRACT_SCHEMA
from data_pipeline.board_recognition.full_board_readout import board_answer
from evals.catan_board_bench.tokens import atlas_metadata, atlas_tokens, base_catan_map

STATIC_TASKS = frozenset({
    "symbolic_direction", "symbolic_direction_choice", "symbolic_neighbors",
    "symbolic_incidence", "symbolic_oriented_step",
})
TRAIN_TASKS = STATIC_TASKS | frozenset({
    "symbolic_owned_nodes", "symbolic_owned_roads", "symbolic_piece_owner",
    "symbolic_owned_incident_roads", "symbolic_reachable", "symbolic_shortest_route",
    "symbolic_near_nodes", "symbolic_near", "symbolic_local_constraint",
    "symbolic_scene_tiles",
})
TRANSFER_TASKS = frozenset({
    "symbolic_settlement_locations", "symbolic_longest_lengths",
    "symbolic_longest_leaders", "symbolic_longest_award",
})
SYMBOLIC_TASKS = TRAIN_TASKS | TRANSFER_TASKS
COLORS = frozenset(c.value for c in Color)
RESOURCES_LOWER = frozenset(r.lower() for r in RESOURCES) | {"desert"}
OFFSETS = dict(zip(
    ("NORTH", "NORTHEAST", "SOUTHEAST", "SOUTH", "SOUTHWEST", "NORTHWEST"),
    ((0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1)), strict=True,
))
DIRECTIONS = ("left", "right", "above", "below")
SET_FORMAT = "Output only the exact unordered set, separated by spaces; NONE if empty. No duplicates or explanation."
ROUTE_RULES = (
    "Use only existing roads of that color. Opponent buildings may be endpoints, "
    "including the start, but never interior transit vertices. "
)
TRAIL_RULES = (
    "A trail uses only one player's existing roads. Count edges, with no undirected "
    "edge reused; cycles and revisited vertices are allowed. Opponent buildings block "
    "interior transit but may be endpoints, including the start. "
)
SETTLEMENT_RULES = (
    "A board-legal settlement location is empty and has no building of any color at "
    "any adjacent node. In setup no road connection is required; in normal play at "
    "least one incident existing road must be owned by the queried player. Ignore "
    "hand, piece supply, and whose turn it is. "
)


class PhysicalStateError(ValueError):
    """Validly encoded source fails material supply or building-distance admission."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _keys(value: object, keys: set[str], label: str) -> None:
    _require(isinstance(value, dict) and set(value) == keys, f"invalid {label} keys")


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_number(value: str) -> None:
    raise ValueError(f"non-integer JSON number: {value}")


def strict_json(text: str) -> object:
    """JSON without duplicate keys, floating numbers, NaN, or trailing prose."""
    _require(isinstance(text, str), "expected JSON text")
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_float=_bad_number, parse_constant=_bad_number)


@lru_cache(maxsize=1)
def _atlas() -> dict:
    atlas = atlas_metadata()
    positions, node_tiles, node_edges, node_ports = {}, {}, {}, {}
    nodes = {n["id"]: n["token"] for n in atlas["nodes"]}
    edges = {e["token"]: tuple(nodes[n] for n in e["id"]) for e in atlas["edges"]}
    touching = {token: set() for token in atlas_tokens()}
    for tile in atlas["tiles"]:
        q, _, r = tile["coord"]
        center = (2 * q + r, 3 * r)
        positions[tile["token"]] = center
        for ref, nid in tile["nodes"].items():
            dx, dy = OFFSETS[ref]
            point = (center[0] + dx, center[1] + dy)
            token = nodes[nid]
            _require(positions.setdefault(token, point) == point, "inconsistent atlas geometry")
            node_tiles.setdefault(token, set()).add(tile["token"])
            touching[token].add(tile["token"])
            touching[tile["token"]].add(token)
        for endpoints in tile["edges"].values():
            edge = f"<E{min(endpoints):02d}_{max(endpoints):02d}>"
            touching[edge].add(tile["token"])
            touching[tile["token"]].add(edge)
    graph = {token: set() for token in nodes.values()}
    for edge, (a, b) in edges.items():
        graph[a].add(b)
        graph[b].add(a)
        for node in (a, b):
            node_edges.setdefault(node, set()).add(edge)
            touching[node].add(edge)
            touching[edge].add(node)
    for port in atlas["ports"]:
        for nid in port["attached_nodes"]:
            node = nodes[nid]
            node_ports.setdefault(node, set()).add(port["token"])
            touching[node].add(port["token"])
            touching[port["token"]].add(node)
    tile_neighbors = {
        t["token"]: {u["token"] for u in atlas["tiles"] if u != t and
                     len(set(t["nodes"].values()) & set(u["nodes"].values())) == 2}
        for t in atlas["tiles"]
    }
    _require(len(positions) == 73 and len(set(positions.values())) == 73,
             "atlas positions must be unique")
    canonical_edges = {tuple(e["id"]) for e in atlas["edges"]}
    _require({tuple(sorted(e)) for e in STATIC_GRAPH.subgraph(range(54)).edges} == canonical_edges,
             "engine graph differs from canonical atlas")
    return dict(raw=atlas, positions=positions, graph=graph, edges=edges,
                touching=touching, tile_neighbors=tile_neighbors, node_tiles=node_tiles,
                node_edges=node_edges, node_ports=node_ports,
                tokens=tuple(atlas_tokens()))


def atlas_geometry() -> dict:
    """Detached oracle-only integer geometry/topology; never serialize into prompts."""
    return copy.deepcopy(_atlas())


def _token(value: object, families: str) -> str:
    _require(isinstance(value, str) and value in _atlas()["tokens"]
             and value[1] in families, f"invalid canonical {families} token: {value!r}")
    return value


def _participants(colors: object) -> tuple[str, ...]:
    _require(isinstance(colors, list) and len(colors) == 4
             and all(isinstance(c, str) and c in COLORS for c in colors)
             and len(set(colors)) == 4, "expected four unique participant colors")
    return tuple(colors)


def _same(actual: object, expected: object, label: str) -> None:
    # JSON equality with type distinctions (True must not masquerade as node 1).
    _require(json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True),
             f"noncanonical {label}")


def _rows(contract: dict, family: str, expected: list) -> list:
    rows = contract.get(family)
    _require(isinstance(rows, list) and len(rows) == len(expected), f"incomplete {family}")
    for row, gold in zip(rows, expected, strict=True):
        _require(isinstance(row, dict), f"invalid {family} row")
        _same(row.get("id"), gold["id"], family + " identity/order")
        _same(row.get("token"), gold["token"], family + " token")
    return rows


def validate_contract(contract: dict) -> dict:
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
    colors = _participants([p.get("color") for p in players])
    atlas = _atlas()
    for family in ("tiles", "nodes", "edges", "ports"):
        for row, gold in zip(_rows(contract, family, atlas["raw"][family]),
                             atlas["raw"][family], strict=True):
            token = gold["token"]
            expected = {}
            if family == "tiles":
                ns = sorted(gold["nodes"].values())
                es = sorted(gold["edges"].values())
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
                expected = dict(coord=gold["coord"], direction=gold["direction"],
                                attached_nodes=gold["attached_nodes"],
                                attached_node_tokens=[f"<N{n:02d}>" for n in gold["attached_nodes"]],
                                kind="generic" if resource is None else "resource",
                                ratio="3:1" if resource is None else "2:1",
                                resource_token=None if resource is None else f"<{resource}>")
            for key, value in expected.items():
                _same(row.get(key), value, f"{token} {key}")
    robber = contract.get("robber")
    _require(isinstance(robber, dict), "missing robber")
    marked = [t for t in contract["tiles"] if t["has_robber"]]
    _require(len(marked) == 1, "expected one robber")
    for key, value in (("tile_id", marked[0]["id"]), ("tile_token", marked[0]["token"]),
                       ("coord", marked[0]["coord"])):
        _same(robber.get(key), value, "robber " + key)
    state = {"board": board_answer(contract), "colors": list(colors)}
    values = decode_state(state)
    for player in players:
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
def _decode(board: str, colors: tuple[str, ...]) -> dict:
    keys = [t for f in "TNEP" for t in sorted(_atlas()["tokens"]) if t[1] == f] + ["robber"]
    entries = board.split(";")
    _require(len(entries) == 155, "board must contain exactly 155 entries")
    values = {}
    for key, entry in zip(keys, entries, strict=True):
        parts = entry.strip().split(maxsplit=1)
        _require(len(parts) == 2 and parts[0] == key, "board identity/order/duplicate error")
        values[key] = parts[1]
    result = dict(tiles={}, ports={}, buildings={}, roads={}, colors=colors,
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


def decode_state(state: dict) -> dict:
    """Strictly parse a complete minimal state, returning detached facts."""
    _keys(state, {"board", "colors"}, "state")
    _require(isinstance(state["board"], str), "board must be text")
    return copy.deepcopy(_decode(state["board"], _participants(state["colors"])))


def detached_board(state: dict) -> Board:
    """Reconstruct fresh map, bidirectional roads, distance cache and road components."""
    data = decode_state(state)
    board_map = base_catan_map()
    for tile in board_map.tiles_by_id.values():
        resource, number = data["tiles"][f"<T{tile.id:02d}>"]
        tile.resource = None if resource == "desert" else resource.upper()
        tile.number = number
    for port in board_map.ports_by_id.values():
        resource = data["ports"][f"<P{port.id:02d}>"]
        port.resource = None if resource == "3:1" else resource.upper()
    board = Board(CatanMap.from_tiles(board_map.tiles))
    board.buildable_subgraph = board.buildable_subgraph.copy()
    board.robber_coordinate = tuple(next(t["coord"] for t in _atlas()["raw"]["tiles"]
                                         if t["token"] == data["robber"]))
    for node, (color, piece) in data["buildings"].items():
        nid = int(node[2:-1])
        board.buildings[nid] = (Color(color), piece.upper())
        board.board_buildable_ids.difference_update({nid, *board.buildable_subgraph.neighbors(nid)})
    for edge, color in data["roads"].items():
        a, b = (int(n[2:-1]) for n in _atlas()["edges"][edge])
        board.roads[a, b] = board.roads[b, a] = Color(color)
    board.recompute_road_state()
    return board


def _near(selector: object, data: dict) -> set[str]:
    _keys(selector, {"kind", "value"}, "near selector")
    kind, value = selector["kind"], selector["value"]
    if kind in ("tile", "port"):
        token = _token(value, "T" if kind == "tile" else "P")
        return {n for n in _atlas()["touching"][token] if n[1] == "N"}
    _require(kind == "resource" and isinstance(value, str) and value in RESOURCES_LOWER,
             "invalid near selector")
    return {n for n, tiles in _atlas()["node_tiles"].items()
            if any(data["tiles"][t][0] == value for t in tiles)}


def _relation(a: str, b: str, direction: str) -> bool:
    _token(a, "NT")
    _token(b, "NT")
    _require(a != b and a[1] == b[1] and direction in DIRECTIONS, "invalid direction query")
    x, y = _atlas()["positions"][a]
    u, v = _atlas()["positions"][b]
    return {"left": x < u, "right": x > u, "above": y < v, "below": y > v}[direction]


def owned_route(state: dict, color: str, start: str, end: str) -> dict:
    """BFS on supplied ownership, symmetric even when either endpoint is blocked."""
    data = decode_state(state)
    _require(color in data["colors"], "nonparticipant color")
    _token(start, "N")
    _token(end, "N")
    graph = {n: [] for n in _atlas()["graph"]}
    for edge, owner in data["roads"].items():
        if owner == color:
            a, b = _atlas()["edges"][edge]
            graph[a].append((b, edge))
            graph[b].append((a, edge))
    parents, queue = {start: None}, deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            nodes, edges = [node], []
            while parents[node] is not None:
                node, edge = parents[node]
                nodes.append(node)
                edges.append(edge)
            return {"nodes": nodes[::-1], "edges": edges[::-1]}
        if node != start and node in data["buildings"] and data["buildings"][node][0] != color:
            continue
        for neighbor, edge in sorted(graph[node]):
            if neighbor not in parents:
                parents[neighbor] = (node, edge)
                queue.append(neighbor)
    return {"nodes": None, "edges": None}


def symbolic_task_role(task: str, split: str) -> str:
    """Canonical dataset eligibility: train, component_eval, or transfer."""
    if task in TRAIN_TASKS and split in ("train", "validation", "test"):
        return "train" if split == "train" else "component_eval"
    if task in TRANSFER_TASKS and split in ("transfer_validation", "transfer_test"):
        return "transfer"
    raise ValueError(f"task/split role violation: {task}/{split}")


@lru_cache(maxsize=256)
def _transfer_facts(board_text: str, colors: tuple) -> dict:
    # Cache immutable-source computations, never an engine Board exposed to callers.
    board = detached_board({"board": board_text, "colors": list(colors)})
    return {
        "lengths": {c: board.road_lengths[Color(c)] for c in colors},
        "award": board.road_color.value if board.road_color else None,
        "settlements": {
            (color, phase): frozenset(f"<N{n:02d}>" for n in board.buildable_node_ids(
                Color(color), initial_build_phase=phase == "setup"))
            for color in colors for phase in ("setup", "normal")
        },
    }


def _solve(task: str, target: dict) -> tuple[str, object]:
    _keys(target, {"state", "query"}, "target")
    query, state = target["query"], target["state"]
    if task in STATIC_TASKS:
        _require(state is None, "static task must not carry dynamic state")
        if task in ("symbolic_direction", "symbolic_direction_choice"):
            _keys(query, {"a", "b", "direction"} | ({"choices"} if task.endswith("choice") else set()), "query")
            result = _relation(query["a"], query["b"], query["direction"])
            if task.endswith("choice"):
                _require(isinstance(query["choices"], list) and len(query["choices"]) == 2
                         and set(query["choices"]) == {query["a"], query["b"]}, "invalid choices")
                reverse = _relation(query["b"], query["a"], query["direction"])
                _require(result != reverse, "direction choice cannot be tied")
                return "choice", query["a"] if result else query["b"]
            return "bool", result
        if task == "symbolic_neighbors":
            _keys(query, {"token"}, "query")
            token = _token(query["token"], "NT")
            return "set", set(_atlas()["graph" if token[1] == "N" else "tile_neighbors"][token])
        if task == "symbolic_incidence":
            _keys(query, {"token", "family"}, "query")
            token = _token(query["token"], "NTEP")
            _require(query["family"] in ("N", "T", "E", "P") and query["family"] != token[1],
                     "invalid incidence family")
            return "set", {n for n in _atlas()["touching"][token] if n[1] == query["family"]}
        _keys(query, {"token", "direction"}, "query")
        token = _token(query["token"], "NT")
        valid = (set(OFFSETS) if token[1] == "N" else
                 {"EAST", "WEST", "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST"})
        _require(query["direction"] in valid, "invalid oriented direction")
        x, y = _atlas()["positions"][token]
        neighbors = _atlas()["graph" if token[1] == "N" else "tile_neighbors"][token]
        result = set()
        for n in neighbors:
            u, v = _atlas()["positions"][n]
            direction = (("NORTH" if v < y else "SOUTH") if u == x else
                         ("EAST" if u > x else "WEST") if v == y else
                         ("NORTH" if v < y else "SOUTH") + ("EAST" if u > x else "WEST"))
            if direction == query["direction"]:
                result.add(n)
        _require(len(result) <= 1, "ambiguous oriented step")
        return "set", result
    data = decode_state(state)
    schemas = {
        "symbolic_owned_nodes": {"color", "piece"}, "symbolic_owned_roads": {"color"},
        "symbolic_piece_owner": {"token"}, "symbolic_owned_incident_roads": {"color", "node"},
        "symbolic_reachable": {"color", "start", "end"},
        "symbolic_shortest_route": {"color", "start", "end"},
        "symbolic_near_nodes": {"near"}, "symbolic_near": {"node", "near"},
        "symbolic_local_constraint": {"color", "node", "predicate"},
        "symbolic_scene_tiles": {"resource"},
        "symbolic_settlement_locations": {"color", "phase", "near"},
        "symbolic_longest_lengths": set(), "symbolic_longest_leaders": set(),
        "symbolic_longest_award": set(),
    }
    _keys(query, schemas[task], "query")
    if "color" in query:
        _require(query["color"] in data["colors"], "nonparticipant color")
    if "node" in query:
        _token(query["node"], "N")
    if task in ("symbolic_reachable", "symbolic_shortest_route"):
        route = owned_route(state, **query)
        return ("bool", route["nodes"] is not None) if task == "symbolic_reachable" else ("route", route)
    if task in ("symbolic_near", "symbolic_near_nodes"):
        nodes = _near(query["near"], data)
        return ("bool", query["node"] in nodes) if task == "symbolic_near" else ("set", nodes)
    if task == "symbolic_scene_tiles":
        _require(query["resource"] in RESOURCES_LOWER, "invalid resource")
        return "set", {t for t, (r, _) in data["tiles"].items() if r == query["resource"]}
    if task in TRANSFER_TASKS:
        facts = _transfer_facts(state["board"], tuple(state["colors"]))
        if task == "symbolic_settlement_locations":
            _require(query["phase"] in ("setup", "normal"), "invalid placement phase")
            nodes = set(facts["settlements"][query["color"], query["phase"]])
            return "set", nodes if query["near"] is None else nodes & _near(query["near"], data)
        lengths = dict(facts["lengths"])
        maximum = max(lengths.values())
        leaders = {c for c, length in lengths.items() if length == maximum}
        if task == "symbolic_longest_lengths":
            return "lengths", lengths
        if task == "symbolic_longest_leaders":
            return "set", leaders
        _require(maximum < 5 or len(leaders) == 1, "award ambiguous without known incumbent")
        return "set", {facts["award"]} if facts["award"] else set()
    board = detached_board(state)
    if task == "symbolic_owned_nodes":
        _require(query["piece"] in ("building", "settlement", "city"), "invalid piece selector")
        return "set", {f"<N{n:02d}>" for n, (c, p) in board.buildings.items()
                       if c.value == query["color"] and (query["piece"] == "building" or p.lower() == query["piece"])}
    if task == "symbolic_owned_roads":
        return "set", {e for e, ends in _atlas()["edges"].items()
                       if board.get_edge_color(tuple(int(n[2:-1]) for n in ends)) == Color(query["color"])}
    if task == "symbolic_piece_owner":
        token = _token(query["token"], "NE")
        color = (board.get_node_color(int(token[2:-1])) if token[1] == "N" else
                 board.get_edge_color(tuple(int(n[2:-1]) for n in _atlas()["edges"][token])))
        return "set", {color.value} if color else set()
    if task == "symbolic_owned_incident_roads":
        return "set", {e for e in _atlas()["node_edges"][query["node"]]
                       if board.is_friendly_road(tuple(int(n[2:-1]) for n in _atlas()["edges"][e]),
                                                 Color(query["color"]))}
    if task == "symbolic_local_constraint":
        node = query["node"]
        predicate = query["predicate"]
        _require(predicate in ("empty", "no_adjacent_building", "has_owned_incident_road"),
                 "only atomic local constraints are training tasks")
        return "bool", {
            "empty": node not in data["buildings"],
            "no_adjacent_building": not (_atlas()["graph"][node] & data["buildings"].keys()),
            "has_owned_incident_road": any(data["roads"].get(e) == query["color"]
                                          for e in _atlas()["node_edges"][node]),
        }[predicate]
    raise ValueError(f"unsupported task: {task}")


def symbolic_answer(task: str, target: dict) -> str:
    """Compute a canonical demonstration; tied shortest routes have multiple valid answers."""
    _require(task in SYMBOLIC_TASKS, "unknown symbolic task")
    kind, result = _solve(task, target)
    if kind == "bool":
        return "yes" if result else "no"
    if kind == "set":
        return " ".join(sorted(result)) or "NONE"
    if kind == "choice":
        return result
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def _route_valid(answer: object, gold: dict, state: dict, query: dict) -> bool:
    _keys(answer, {"nodes", "edges"}, "route")
    if gold["nodes"] is None:
        return answer == gold
    nodes, edges = answer["nodes"], answer["edges"]
    _require(isinstance(nodes, list) and isinstance(edges, list), "route arrays required")
    _require(all(isinstance(v, str) for v in nodes + edges), "route values must be strings")
    if not nodes or len(nodes) != len(edges) + 1 or len(edges) != len(gold["edges"]):
        return False
    if nodes[0] != query["start"] or nodes[-1] != query["end"] or len(set(edges)) != len(edges):
        return False
    data = decode_state(state)
    return all(_atlas()["edges"].get(edge) in ((a, b), (b, a))
               and data["roads"].get(edge) == query["color"]
               for a, b, edge in zip(nodes, nodes[1:], edges)) and all(
        node not in data["buildings"] or data["buildings"][node][0] == query["color"]
        for node in nodes[1:-1])


def score_symbolic_task(expected: str, response: str, metadata: dict) -> dict | None:
    """Evaluator-compatible metrics; recompute gold from metadata, never expected.

    None means an unrecognized task only. Known malformed metadata raises ValueError
    (including ambiguous awards); malformed model answers receive correct=False.
    Whitespace tolerance does not permit prose, duplicate sets, or JSON key extras.
    """
    _require(isinstance(metadata, dict), "metadata must be a dictionary")
    task = metadata.get("task_type")
    _require(task is None or isinstance(task, str), "task_type must be a string")
    if task not in SYMBOLIC_TASKS:
        return None
    try:
        if "training_family" in metadata:
            _require(metadata["training_family"] == task, "task/family declaration mismatch")
        if "task_role" in metadata or "split" in metadata:
            _require(metadata.get("task_role") == symbolic_task_role(task, metadata.get("split")),
                     "task role declaration mismatch")
        kind, gold = _solve(task, metadata.get("target"))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed metadata for {task}: {exc}") from exc
    normalized = response.strip() if isinstance(response, str) else response
    try:
        _require(isinstance(response, str), "response must be text")
        if kind in ("bool", "choice"):
            correct = normalized == (("yes" if gold else "no") if kind == "bool" else gold)
        elif kind == "set":
            values = [] if normalized == "NONE" else normalized.split()
            _require(normalized != "" and len(values) == len(set(values)), "invalid set")
            correct = set(values) == gold
            normalized = " ".join(sorted(values)) or "NONE"
        else:
            answer = strict_json(response)
            if kind == "lengths":
                _keys(answer, set(gold), "lengths")
                _require(all(type(v) is int and v >= 0 for v in answer.values()), "invalid lengths")
                correct = answer == gold
            else:
                correct = _route_valid(answer, gold, metadata["target"]["state"], metadata["target"]["query"])
            normalized = json.dumps(answer, sort_keys=True, separators=(",", ":"))
    except (ValueError, TypeError):
        correct = False
    gold_text = (("yes" if gold else "no") if kind == "bool" else
                 (" ".join(sorted(gold)) or "NONE") if kind == "set" else
                 gold if kind == "choice" else json.dumps(gold, sort_keys=True, separators=(",", ":")))
    return {"correct": bool(correct), "scoring": task, "expected_normalized": gold_text,
            "response_normalized": normalized}


def symbolic_prompt(task: str, target: dict) -> str:
    """Render only canonical board facts and the query, with no oracle expansions."""
    _solve(task, target)  # Fail closed before rendering malformed query/state metadata.
    q = target["query"]
    prefix = "Use the fixed learned Catan atlas. "
    if target["state"] is not None:
        state = target["state"]
        prefix += "Participants: " + " ".join(state["colors"]) + ".\nBoard: " + state["board"] + "\n"
    if task == "symbolic_direction":
        question = f"Is {q['a']} strictly {q['direction']} {q['b']}? Compare that axis independently; equality means no. Answer only yes or no."
    elif task == "symbolic_direction_choice":
        question = f"Which is farther {q['direction']}: {' or '.join(q['choices'])}? Compare that axis independently. Output only one offered token."
    elif task == "symbolic_neighbors":
        question = f"List all {'edge-connected node' if q['token'][1] == 'N' else 'side-sharing land tile'} neighbors of {q['token']}. " + SET_FORMAT
    elif task == "symbolic_incidence":
        question = f"List all {'node' if q['family'] == 'N' else 'edge' if q['family'] == 'E' else 'land tile' if q['family'] == 'T' else 'port'} tokens incident to (touching) {q['token']}. " + SET_FORMAT
    elif task == "symbolic_oriented_step":
        question = f"From {q['token']}, take exactly one {'node-edge' if q['token'][1] == 'N' else 'tile-side'} step {q['direction'].lower()}. Give the neighbor if it exists. " + SET_FORMAT
    elif task == "symbolic_owned_nodes":
        question = f"List nodes with a {q['color']} {q['piece']}. " + SET_FORMAT
    elif task == "symbolic_owned_roads":
        question = f"List all existing {q['color']} road edge tokens. " + SET_FORMAT
    elif task == "symbolic_piece_owner":
        question = f"Which participant owns the piece at {q['token']}? Give the color set. " + SET_FORMAT
    elif task == "symbolic_owned_incident_roads":
        question = f"List existing {q['color']} road edges incident to {q['node']}. " + SET_FORMAT
    elif task in ("symbolic_reachable", "symbolic_shortest_route"):
        question = ROUTE_RULES + f"For {q['color']}, "
        if task == "symbolic_reachable":
            question += f"is {q['end']} reachable from {q['start']} (zero edges allowed)? Answer only yes or no."
        else:
            question += f"give a shortest route from {q['start']} to {q['end']}. "
            question += ('Output only JSON with exactly "nodes" and "edges", ordered arrays of atlas tokens; '
                         'include both endpoints. Any tied shortest route is accepted. No route: '
                         '{"nodes":null,"edges":null}. Zero-edge route: the one start node and an empty edges array.')
    elif task in ("symbolic_near", "symbolic_near_nodes", "symbolic_settlement_locations"):
        near = q.get("near")
        description = (f"touching {near['kind']} {near['value']}" if near else "anywhere on the board")
        if task == "symbolic_near":
            question = f"Is node {q['node']} near {near['value']}, meaning {description}? Answer only yes or no."
        elif task == "symbolic_near_nodes":
            question = f"List nodes near {near['value']}, meaning {description}. " + SET_FORMAT
        else:
            question = SETTLEMENT_RULES + f"For {q['color']} in {q['phase']} phase, list every board-legal settlement node {description}. Near means touching. " + SET_FORMAT
    elif task == "symbolic_local_constraint":
        descriptions = {"empty": "has no building", "no_adjacent_building": "has no building at any edge-adjacent node",
                        "has_owned_incident_road": f"has at least one existing {q['color']} incident road"}
        question = f"Does {q['node']} satisfy this single condition: {descriptions[q['predicate']]}? Answer only yes or no."
    elif task == "symbolic_scene_tiles":
        question = f"List all land tiles labeled {q['resource']} in the supplied board. " + SET_FORMAT
    elif task == "symbolic_longest_lengths":
        question = TRAIL_RULES + 'Give each participant\'s maximum trail length, including zeros. Output only a JSON object keyed by all participant colors with integer lengths.'
    elif task == "symbolic_longest_leaders":
        question = TRAIL_RULES + "Give all colors tied for the maximum trail length, including all participants if all lengths are zero. This is not award ownership. " + SET_FORMAT
    else:
        question = TRAIL_RULES + "The Longest Road award requires at least five edges. A unique maximum qualifies; a tied incumbent keeps the award. This question has an unambiguous answer without incumbent history. Give the award-holder color set. " + SET_FORMAT
    return prefix + question
