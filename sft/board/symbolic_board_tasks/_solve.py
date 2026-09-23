"""Task solvers over static geometry and decoded dynamic states."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from functools import lru_cache
from typing import cast

from cle.game_engine.models.player import Color
from sft.board.symbolic_board_tasks._constants import (
    DIRECTIONS,
    OFFSETS,
    RESOURCES_LOWER,
    STATIC_TASKS,
    TRAIN_TASKS,
    TRANSFER_TASKS,
    _keys,
    _require,
)
from sft.board.symbolic_board_tasks._contracts import decode_state, detached_board
from sft.board.symbolic_board_tasks._geometry import _atlas, _token
from sft.board.symbolic_board_tasks._types import (
    DecodedState,
    Route,
    TransferFacts,
)
from sft.json_types import JsonDict


def _edge_pair(ends: tuple[str, ...]) -> tuple[int, int]:
    first, second = (int(n[2:-1]) for n in ends)
    return first, second


def _near(selector: object, data: DecodedState) -> set[str]:
    _keys(selector, {"kind", "value"}, "near selector")
    chosen = cast("JsonDict", selector)
    kind, value = chosen["kind"], chosen["value"]
    if kind in ("tile", "port"):
        token = _token(value, "T" if kind == "tile" else "P")
        return {n for n in _atlas()["touching"][token] if n[1] == "N"}
    _require(kind == "resource" and isinstance(value, str) and value in RESOURCES_LOWER,
             "invalid near selector")
    return {n for n, tiles in _atlas()["node_tiles"].items()
            if any(data["tiles"][t][0] == value for t in tiles)}


def _relation(a: object, b: object, direction: object) -> bool:
    first = _token(a, "NT")
    second = _token(b, "NT")
    _require(first != second and first[1] == second[1] and direction in DIRECTIONS,
             "invalid direction query")
    x, y = _atlas()["positions"][first]
    u, v = _atlas()["positions"][second]
    return {"left": x < u, "right": x > u, "above": y < v,
            "below": y > v}[cast("str", direction)]


def owned_route(state: Mapping[str, object], color: str, start: str, end: str) -> Route:
    """BFS on supplied ownership, symmetric even when either endpoint is blocked."""
    data = decode_state(state)
    _require(color in data["colors"], "nonparticipant color")
    _token(start, "N")
    _token(end, "N")
    graph: dict[str, list[tuple[str, str]]] = {n: [] for n in _atlas()["graph"]}
    for edge, owner in data["roads"].items():
        if owner == color:
            a, b = _atlas()["edges"][edge]
            graph[a].append((b, edge))
            graph[b].append((a, edge))
    parents: dict[str, tuple[str, str] | None] = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            nodes, edges = [node], []
            parent = parents[node]
            while parent is not None:
                node, edge = parent
                nodes.append(node)
                edges.append(edge)
                parent = parents[node]
            return Route(nodes=nodes[::-1], edges=edges[::-1])
        if node != start and node in data["buildings"] and data["buildings"][node][0] != color:
            continue
        for neighbor, edge in sorted(graph[node]):
            if neighbor not in parents:
                parents[neighbor] = (node, edge)
                queue.append(neighbor)
    return Route(nodes=None, edges=None)


def symbolic_task_role(task: str, split: object) -> str:
    """Canonical dataset eligibility: train, component_eval, or transfer."""
    if task in TRAIN_TASKS and split in ("train", "validation", "test"):
        return "train" if split == "train" else "component_eval"
    if task in TRANSFER_TASKS and split in ("transfer_validation", "transfer_test"):
        return "transfer"
    raise ValueError(f"task/split role violation: {task}/{split}")


@lru_cache(maxsize=256)
def _transfer_facts(board_text: str, colors: tuple[str, ...]) -> TransferFacts:
    # Cache immutable-source computations, never an engine Board exposed to callers.
    board = detached_board({"board": board_text, "colors": list(colors)})
    return TransferFacts(
        lengths={c: board.road_lengths[Color(c)] for c in colors},
        award=board.road_color.value if board.road_color else None,
        settlements={
            (color, phase): frozenset(f"<N{n:02d}>" for n in board.buildable_node_ids(
                Color(color), initial_build_phase=phase == "setup"))
            for color in colors for phase in ("setup", "normal")
        },
    )


def _solve(task: str, target: JsonDict) -> tuple[str, object]:
    _keys(target, {"state", "query"}, "target")
    query, state = cast("JsonDict", target["query"]), target["state"]
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
            atlas = _atlas()
            adjacency = atlas["graph"] if token[1] == "N" else atlas["tile_neighbors"]
            return "set", set(adjacency[token])
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
        atlas = _atlas()
        x, y = atlas["positions"][token]
        neighbors = atlas["graph"] if token[1] == "N" else atlas["tile_neighbors"]
        matches: set[str] = set()
        for n in neighbors[token]:
            u, v = _atlas()["positions"][n]
            direction = (("NORTH" if v < y else "SOUTH") if u == x else
                         ("EAST" if u > x else "WEST") if v == y else
                         ("NORTH" if v < y else "SOUTH") + ("EAST" if u > x else "WEST"))
            if direction == query["direction"]:
                matches.add(n)
        _require(len(matches) <= 1, "ambiguous oriented step")
        return "set", matches
    state_map = cast("JsonDict", state)
    data = decode_state(state_map)
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
        route = owned_route(state_map, cast("str", query["color"]),
                            cast("str", query["start"]), cast("str", query["end"]))
        return ("bool", route["nodes"] is not None) if task == "symbolic_reachable" else ("route", route)
    if task in ("symbolic_near", "symbolic_near_nodes"):
        nodes = _near(query["near"], data)
        return ("bool", query["node"] in nodes) if task == "symbolic_near" else ("set", nodes)
    if task == "symbolic_scene_tiles":
        _require(query["resource"] in RESOURCES_LOWER, "invalid resource")
        return "set", {t for t, (r, _) in data["tiles"].items() if r == query["resource"]}
    if task in TRANSFER_TASKS:
        facts = _transfer_facts(cast("str", state_map["board"]),
                                tuple(cast("list[str]", state_map["colors"])))
        if task == "symbolic_settlement_locations":
            _require(query["phase"] in ("setup", "normal"), "invalid placement phase")
            nodes = set(facts["settlements"][cast("str", query["color"]),
                                             cast("str", query["phase"])])
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
    board = detached_board(state_map)
    if task == "symbolic_owned_nodes":
        _require(query["piece"] in ("building", "settlement", "city"), "invalid piece selector")
        return "set", {f"<N{n:02d}>" for n, (c, p) in board.buildings.items()
                       if c.value == query["color"] and (query["piece"] == "building" or p.lower() == query["piece"])}
    if task == "symbolic_owned_roads":
        return "set", {e for e, ends in _atlas()["edges"].items()
                       if board.get_edge_color(_edge_pair(ends)) == Color(cast("str", query["color"]))}
    if task == "symbolic_piece_owner":
        token = _token(query["token"], "NE")
        color = (board.get_node_color(int(token[2:-1])) if token[1] == "N" else
                 board.get_edge_color(_edge_pair(_atlas()["edges"][token])))
        return "set", {color.value} if color else set()
    if task == "symbolic_owned_incident_roads":
        return "set", {e for e in _atlas()["node_edges"][cast("str", query["node"])]
                       if board.is_friendly_road(_edge_pair(_atlas()["edges"][e]),
                                                 Color(cast("str", query["color"])))}
    if task == "symbolic_local_constraint":
        node = cast("str", query["node"])
        predicate = cast("str", query["predicate"])
        _require(predicate in ("empty", "no_adjacent_building", "has_owned_incident_road"),
                 "only atomic local constraints are training tasks")
        return "bool", {
            "empty": node not in data["buildings"],
            "no_adjacent_building": not (_atlas()["graph"][node] & data["buildings"].keys()),
            "has_owned_incident_road": any(data["roads"].get(e) == query["color"]
                                          for e in _atlas()["node_edges"][node]),
        }[predicate]
    raise ValueError(f"unsupported task: {task}")
