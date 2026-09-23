from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from itertools import combinations
from pathlib import Path
from typing import cast

from evals.catan_board_bench.tokens import atlas_tokens
from sft.board.symbolic_board_tasks import (
    DIRECTIONS,
    OFFSETS,
    RESOURCES_LOWER,
    atlas_geometry,
    decode_state,
    owned_route,
    symbolic_answer,
)
from sft.board.symbolic_board_tasks._types import Atlas, DecodedState, Route, StatePayload
from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_str

from ._sources import KNOWN_DIRECTION_TRAINING, SPLITS, _check, _hash, read_jsonl
from ._types import BucketKey, QueryDict, SelectorDict, SlotDict


def known_direction_exposure(
    paths: tuple[Path, ...] = KNOWN_DIRECTION_TRAINING,
) -> JsonLikeDict:
    """Union explicitly identified historical training corpora; never infer training from v1."""
    pair_sources: defaultdict[str, list[str]] = defaultdict(list)
    files: list[dict[str, str]] = []
    vocabulary = set(atlas_tokens())
    for path in paths:
        files.append(_hash(path))
        for row in read_jsonl(path):
            metadata = as_dict(row.get("metadata", {}))
            task = cast("str", row.get("task_type", metadata.get("task_type", "")))
            if "direction" not in task or task == "port_direction":
                continue
            target = as_dict(metadata.get("target", {}))
            raw = target.get("tokens", metadata.get("tokens"))
            if raw is None and "a" in target and "b" in target:
                raw = [target["a"], target["b"]]
            tokens = cast("list[str]", raw)
            _check(isinstance(tokens, list) and len(tokens) == 2 and len(set(tokens)) == 2
                   and set(tokens) <= vocabulary and tokens[0][1] == tokens[1][1]
                   and tokens[0][1] in "NT", "unparsed historical direction training query")
            key = " ".join(sorted(tokens))
            if str(path) not in pair_sources[key]:
                pair_sources[key].append(str(path))
    return {"files": files, "pair_sources": dict(sorted(pair_sources.items())),
            "known_pair_count": len(pair_sources),
            "status": "known_sources_audited_history_incomplete",
            "claim": "New-corpus holdout only. Known historical training pairs moved to train/recall; unobserved checkpoint exposure is not certified."}


def directional_pair_manifest(
    seed: int, exposed_pairs: Sequence[Sequence[str]] = (),
) -> dict[str, list[list[str]]]:
    """One partition for unordered pairs, irrespective of axis/inverse/choice order."""
    atlas = atlas_geometry()
    strata: defaultdict[tuple[str, bool, bool], list[list[str]]] = defaultdict(list)
    for family in "NT":
        tokens = sorted(t for t in atlas["positions"] if t[1] == family)
        for a, b in combinations(tokens, 2):
            x, y = atlas["positions"][a]
            u, v = atlas["positions"][b]
            strata[(family, x == u, y == v)].append([a, b])
    rng = random.Random(seed)
    splits: dict[str, list[list[str]]] = {s: [] for s in SPLITS[:3]}
    for _, pairs in sorted(strata.items()):
        rng.shuffle(pairs)
        n = max(1, len(pairs) // 5)
        splits["validation"].extend(pairs[:n])
        splits["test"].extend(pairs[n:2 * n])
        splits["train"].extend(pairs[2 * n:])
    exposed = {tuple(p) for p in exposed_pairs}
    for split in ("validation", "test"):
        splits["train"].extend(p for p in splits[split] if tuple(p) in exposed)
        splits[split] = [p for p in splits[split] if tuple(p) not in exposed]
    return {s: sorted(v) for s, v in splits.items()}


def _balanced(pool: list[QueryDict], count: int, key: Callable[[QueryDict], BucketKey],
              rng: random.Random) -> list[QueryDict]:
    buckets: defaultdict[BucketKey, list[QueryDict]] = defaultdict(list)
    for item in pool:
        buckets[key(item)].append(item)
    keys = sorted(buckets)
    _check(bool(keys), "empty query pool")
    for bucket in buckets.values():
        rng.shuffle(bucket)
    return [buckets[k][(i // len(keys)) % len(buckets[k])]
            for i in range(count) for k in [keys[i % len(keys)]]]


def static_queries(task: str, pairs: list[list[str]], count: int,
                   rng: random.Random) -> list[QueryDict]:
    atlas = atlas_geometry()
    pool: list[QueryDict]
    if task in ("symbolic_direction", "symbolic_direction_choice"):
        pool = []
        for pair in pairs:
            # In a choice prompt the offered order already expresses operand order.
            # Reversing hidden a/b would duplicate an identical model presentation.
            for a, b in ((pair,) if task.endswith("choice") else (pair, pair[::-1])):
                for direction in DIRECTIONS:
                    query: QueryDict = dict(a=a, b=b, direction=direction)
                    if task.endswith("choice"):
                        for choices in (pair, pair[::-1]):
                            candidate: QueryDict = dict(query, choices=choices)
                            try:
                                symbolic_answer(
                                    task, cast("JsonDict", {"state": None, "query": candidate}))
                            except ValueError:  # equal on this axis: no two-choice question
                                continue
                            pool.append(candidate)
                    else:
                        pool.append(query)

        def group(q: QueryDict) -> BucketKey:
            answer = symbolic_answer(task, cast("JsonDict", {"state": None, "query": q}))
            return ((as_str(q["a"])[1], cast("list[str]", q["choices"]).index(answer),
                     as_str(q["direction"])) if task.endswith("choice")
                    else (as_str(q["a"])[1], answer))

        return _balanced(pool, count, group, rng)
    if task == "symbolic_neighbors":
        pool = [{"token": t} for t in sorted(atlas["positions"])]
        return _balanced(pool, count, lambda q: (as_str(q["token"])[1],), rng)
    if task == "symbolic_incidence":
        pool = [{"token": t, "family": f} for t in sorted(atlas["tokens"])
                for f in ({"N": "TEP", "T": "NE", "E": "NT", "P": "N"}[t[1]])]
        # Every edge and port is an active query at least once in training.
        rng.shuffle(pool)
        return [pool[i % len(pool)] for i in range(count)]
    pool = [{"token": t, "direction": d} for t in sorted(atlas["positions"])
            for d in (tuple(OFFSETS) if t[1] == "N" else
                      ("EAST", "WEST", "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST"))]
    return _balanced(pool, count, lambda q: (
        as_str(q["token"])[1],
        symbolic_answer(task, cast("JsonDict", {"state": None, "query": q})) == "NONE"), rng)


def component_slots(task: str, count: int, rng: random.Random) -> list[SlotDict]:
    """Factorial roster-position x mode x polarity slots, shuffled independently.

    Every binary mode/position cell has exactly as many positive as negative rows.
    Mode and position marginal sizes may differ by one pair when quotas do not
    divide the full factorial. Selection never chooses the queried color by polarity.
    """
    modes = {
        "symbolic_local_constraint": ("empty", "no_adjacent_building", "has_owned_incident_road"),
        "symbolic_owned_nodes": ("building", "settlement", "city"),
        "symbolic_piece_owner": ("N", "E"),
        "symbolic_near": ("tile", "resource", "port"),
        "symbolic_near_nodes": ("tile", "resource", "port"),
        "symbolic_scene_tiles": tuple(sorted(RESOURCES_LOWER)),
    }.get(task, ("all",))
    color_tasks = {"symbolic_local_constraint", "symbolic_owned_nodes", "symbolic_owned_roads",
                   "symbolic_owned_incident_roads", "symbolic_reachable", "symbolic_shortest_route"}
    positions: Sequence[int | None] = range(4) if task in color_tasks else (None,)
    slots: list[SlotDict] = []
    if task in ("symbolic_reachable", "symbolic_shortest_route"):
        _check(count % 8 == 0, "route quota must be divisible by eight")
        for mode, n in (("zero", count // 8), ("no_route", count * 3 // 8), ("route", count // 2)):
            for i in range(n):
                slots.append(dict(mode=mode, position=i % 4, polarity=mode))
    elif task in ("symbolic_near_nodes", "symbolic_scene_tiles"):
        slots = [dict(mode=modes[i % len(modes)], position=None, polarity="nonempty") for i in range(count)]
    else:
        _check(count % 2 == 0, "binary component quotas must be even")
        cells = [(mode, position) for position in positions for mode in modes]
        for i in range(count // 2):
            mode, position = cells[i % len(cells)]
            for polarity in ("positive", "negative"):
                slots.append(dict(mode=mode, position=position, polarity=polarity))
    rng.shuffle(slots)
    return slots


def selector_nodes(selector: SelectorDict | None, data: DecodedState,
                   atlas: Atlas) -> set[str]:
    """Oracle-side selector projection; this node expansion never enters a prompt."""
    if selector is None:
        return set(atlas["graph"])
    if selector["kind"] in ("tile", "port"):
        return {n for n in atlas["touching"][selector["value"]] if n[1] == "N"}
    return {n for n, tiles in atlas["node_tiles"].items()
            if any(data["tiles"][tile][0] == selector["value"] for tile in tiles)}


def _selectors(kind: str, atlas: Atlas) -> list[SelectorDict]:
    values = (sorted(RESOURCES_LOWER) if kind == "resource" else
              sorted(t for t in atlas["tokens"] if t[1] == ("T" if kind == "tile" else "P")))
    return [dict(kind=kind, value=v) for v in values]


def _component_options(task: str, data: DecodedState, slot: SlotDict,
                       atlas: Atlas) -> list[QueryDict]:
    """Fast real-fact sampling predicates, independently checked by the final scorer."""
    mode, positive = cast("str", slot["mode"]), slot["polarity"] != "negative"
    position = cast("int | None", slot["position"])
    color = data["colors"][position] if position is not None else None
    buildings, roads = data["buildings"], data["roads"]
    if task == "symbolic_near_nodes":
        return [{"near": s} for s in _selectors(mode, atlas)]
    if task == "symbolic_scene_tiles":
        return [{"resource": mode}]
    if task == "symbolic_owned_nodes":
        present = any(c == color and (mode == "building" or p == mode) for c, p in buildings.values())
        return [dict(color=color, piece=mode)] if present == positive else []
    if task == "symbolic_owned_roads":
        return [dict(color=color)] if (color in roads.values()) == positive else []
    if task == "symbolic_piece_owner":
        occupied = buildings if mode == "N" else roads
        return [{"token": t} for t in atlas["tokens"] if t[1] == mode and (t in occupied) == positive]
    if task == "symbolic_near":
        # All candidates are source supported. Selection below balances selector kind.
        near_result: list[QueryDict] = []
        for selector in _selectors(mode, atlas):
            touching = selector_nodes(selector, data, atlas)
            near_result.extend(cast("QueryDict", dict(node=n, near=selector))
                               for n in atlas["graph"] if (n in touching) == positive)
        return near_result
    result: list[QueryDict] = []
    for node, neighbors in atlas["graph"].items():
        incident = any(roads.get(e) == color for e in atlas["node_edges"][node])
        if task == "symbolic_owned_incident_roads":
            if incident == positive:
                result.append(dict(color=color, node=node))
        else:
            condition = {"empty": node not in buildings,
                         "no_adjacent_building": not neighbors & buildings.keys(),
                         "has_owned_incident_road": incident}[mode]
            if condition == positive:
                result.append(dict(color=color, node=node, predicate=mode))
    return result


def _route_query(state: StatePayload, slot: SlotDict, rng: random.Random,
                 atlas: Atlas) -> QueryDict:
    data = decode_state(state)
    color = state["colors"][cast("int", slot["position"])]
    mode = cast("str", slot["mode"])
    nodes = sorted(atlas["graph"])
    if mode == "zero":
        node = rng.choice(nodes)
        return dict(color=color, start=node, end=node)
    road_nodes = sorted({n for e, c in data["roads"].items() if c == color for n in atlas["edges"][e]})
    pool = list(combinations(road_nodes, 2))
    rng.shuffle(pool)
    if mode == "no_route":
        # Prefer disconnected endpoints within an owned graph when supported.
        rest = list(combinations(nodes, 2))
        rng.shuffle(rest)
        pool += rest
    candidates: list[tuple[str, str]] = []
    for start, end in pool:
        route: Route = owned_route(state, color, start, end)
        if mode == "no_route" and route["nodes"] is None:
            candidates = [(start, end)]
            break
        if mode == "route" and route["nodes"] is not None:
            candidates.append((start, end))
            if len(cast("list[str]", route["edges"])) >= 2:
                candidates = [(start, end)]
                break
    _check(bool(candidates), "admitted route support did not yield a query")
    start, end = rng.choice(candidates)
    if rng.randrange(2):
        start, end = end, start
    return dict(color=color, start=start, end=end)
