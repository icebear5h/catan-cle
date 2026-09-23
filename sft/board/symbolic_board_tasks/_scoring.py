"""Canonical answers, recomputing strict scoring, and prompt rendering."""

from __future__ import annotations

import json
from typing import cast

from sft.board.symbolic_board_tasks._constants import (
    ROUTE_RULES,
    SET_FORMAT,
    SETTLEMENT_RULES,
    SYMBOLIC_TASKS,
    TRAIL_RULES,
    _keys,
    _require,
    strict_json,
)
from sft.board.symbolic_board_tasks._contracts import decode_state
from sft.board.symbolic_board_tasks._geometry import _atlas
from sft.board.symbolic_board_tasks._solve import _solve, symbolic_task_role
from sft.board.symbolic_board_tasks._types import Route
from sft.json_types import JsonDict


def symbolic_answer(task: str, target: JsonDict) -> str:
    """Compute a canonical demonstration; tied shortest routes have multiple valid answers."""
    _require(task in SYMBOLIC_TASKS, "unknown symbolic task")
    kind, result = _solve(task, target)
    if kind == "bool":
        return "yes" if result else "no"
    if kind == "set":
        return " ".join(sorted(cast("set[str]", result))) or "NONE"
    if kind == "choice":
        return cast("str", result)
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def _route_valid(answer: object, gold: Route, state: JsonDict, query: JsonDict) -> bool:
    _keys(answer, {"nodes", "edges"}, "route")
    if gold["nodes"] is None:
        return answer == gold
    gold_edges = gold["edges"]
    if gold_edges is None:
        raise TypeError("route gold carries nodes without edges")
    claimed = cast("JsonDict", answer)
    nodes = cast("list[str]", claimed["nodes"])
    edges = cast("list[str]", claimed["edges"])
    _require(isinstance(nodes, list) and isinstance(edges, list), "route arrays required")
    _require(all(isinstance(v, str) for v in nodes + edges), "route values must be strings")
    if not nodes or len(nodes) != len(edges) + 1 or len(edges) != len(gold_edges):
        return False
    if nodes[0] != query["start"] or nodes[-1] != query["end"] or len(set(edges)) != len(edges):
        return False
    data = decode_state(state)
    return all(_atlas()["edges"].get(edge) in ((a, b), (b, a))
               and data["roads"].get(edge) == query["color"]
               for a, b, edge in zip(nodes, nodes[1:], edges)) and all(
        node not in data["buildings"] or data["buildings"][node][0] == query["color"]
        for node in nodes[1:-1])


def score_symbolic_task(expected: str, response: str, metadata: JsonDict) -> JsonDict | None:
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
            _require(metadata.get("task_role") == symbolic_task_role(
                task, cast("str", metadata.get("split"))),
                "task role declaration mismatch")
        kind, gold = _solve(task, cast("JsonDict", metadata.get("target")))
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
                _keys(answer, set(cast("dict[str, int]", gold)), "lengths")
                _require(all(type(v) is int and v >= 0
                             for v in cast("JsonDict", answer).values()), "invalid lengths")
                correct = answer == gold
            else:
                target = cast("JsonDict", metadata["target"])
                correct = _route_valid(answer, cast("Route", gold),
                                       cast("JsonDict", target["state"]),
                                       cast("JsonDict", target["query"]))
            normalized = json.dumps(answer, sort_keys=True, separators=(",", ":"))
    except (ValueError, TypeError):
        correct = False
    gold_text = (("yes" if gold else "no") if kind == "bool" else
                 (" ".join(sorted(cast("set[str]", gold))) or "NONE") if kind == "set" else
                 cast("str", gold) if kind == "choice" else
                 json.dumps(gold, sort_keys=True, separators=(",", ":")))
    return {"correct": bool(correct), "scoring": task, "expected_normalized": gold_text,
            "response_normalized": normalized}


def symbolic_prompt(task: str, target: JsonDict) -> str:
    """Render only canonical board facts and the query, with no oracle expansions."""
    _solve(task, target)  # Fail closed before rendering malformed query/state metadata.
    q = cast("JsonDict", target["query"])
    prefix = "Use the fixed learned Catan atlas. "
    if target["state"] is not None:
        state = cast("JsonDict", target["state"])
        prefix += ("Participants: " + " ".join(cast("list[str]", state["colors"]))
                   + ".\nBoard: " + cast("str", state["board"]) + "\n")
    if task == "symbolic_direction":
        question = f"Is {q['a']} strictly {q['direction']} {q['b']}? Compare that axis independently; equality means no. Answer only yes or no."
    elif task == "symbolic_direction_choice":
        question = f"Which is farther {q['direction']}: {' or '.join(cast('list[str]', q['choices']))}? Compare that axis independently. Output only one offered token."
    elif task == "symbolic_neighbors":
        question = f"List all {'edge-connected node' if cast('str', q['token'])[1] == 'N' else 'side-sharing land tile'} neighbors of {q['token']}. " + SET_FORMAT
    elif task == "symbolic_incidence":
        question = f"List all {'node' if q['family'] == 'N' else 'edge' if q['family'] == 'E' else 'land tile' if q['family'] == 'T' else 'port'} tokens incident to (touching) {q['token']}. " + SET_FORMAT
    elif task == "symbolic_oriented_step":
        question = f"From {q['token']}, take exactly one {'node-edge' if cast('str', q['token'])[1] == 'N' else 'tile-side'} step {cast('str', q['direction']).lower()}. Give the neighbor if it exists. " + SET_FORMAT
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
        near = cast("JsonDict | None", q.get("near"))
        description = (f"touching {near['kind']} {near['value']}" if near else "anywhere on the board")
        if task == "symbolic_near":
            question = f"Is node {q['node']} near {cast('JsonDict', near)['value']}, meaning {description}? Answer only yes or no."
        elif task == "symbolic_near_nodes":
            question = f"List nodes near {cast('JsonDict', near)['value']}, meaning {description}. " + SET_FORMAT
        else:
            question = SETTLEMENT_RULES + f"For {q['color']} in {q['phase']} phase, list every board-legal settlement node {description}. Near means touching. " + SET_FORMAT
    elif task == "symbolic_local_constraint":
        descriptions = {"empty": "has no building", "no_adjacent_building": "has no building at any edge-adjacent node",
                        "has_owned_incident_road": f"has at least one existing {q['color']} incident road"}
        question = f"Does {q['node']} satisfy this single condition: {descriptions[cast('str', q['predicate'])]}? Answer only yes or no."
    elif task == "symbolic_scene_tiles":
        question = f"List all land tiles labeled {q['resource']} in the supplied board. " + SET_FORMAT
    elif task == "symbolic_longest_lengths":
        question = TRAIL_RULES + 'Give each participant\'s maximum trail length, including zeros. Output only a JSON object keyed by all participant colors with integer lengths.'
    elif task == "symbolic_longest_leaders":
        question = TRAIL_RULES + "Give all colors tied for the maximum trail length, including all participants if all lengths are zero. This is not award ownership. " + SET_FORMAT
    else:
        question = TRAIL_RULES + "The Longest Road award requires at least five edges. A unique maximum qualifies; a tied incumbent keeps the award. This question has an unambiguous answer without incumbent history. Give the award-holder color set. " + SET_FORMAT
    return prefix + question
