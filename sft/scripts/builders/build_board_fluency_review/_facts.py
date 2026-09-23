from __future__ import annotations

from collections import Counter, deque
from typing import cast

from sft.board.symbolic_board_tasks._types import Atlas, DecodedState
from sft.json_types import JsonLikeDict

from ._sources import RESOURCES, Donor, QueryDict, compact, require


def richness(donors: list[Donor], atlas: Atlas) -> JsonLikeDict:
    """Observable source facts, without pretending independent rows are new states."""
    unique = {d.state_hash: d for d in donors}
    counts: Counter[str] = Counter()
    roads, buildings, cities = [], [], []
    for donor in unique.values():
        data = donor.data
        roads.append(len(data["roads"]))
        buildings.append(len(data["buildings"]))
        cities.append(sum(piece == "city" for _, piece in data["buildings"].values()))
        flags = set()
        if data["roads"] or data["buildings"]:
            flags.add("nonempty")
        if cities[-1]:
            flags.add("has_city")
        if any(piece == "settlement" for _, piece in data["buildings"].values()):
            flags.add("has_settlement")
        if any(atlas["node_ports"].get(n) for n in data["buildings"]):
            flags.add("has_port_building")
        if data["tiles"][data["robber"]][0] != "desert":
            flags.add("robber_on_resource")
        for color in data["colors"]:
            degree = Counter(n for e, c in data["roads"].items() if c == color
                             for n in atlas["edges"][e])
            if any(v >= 3 for v in degree.values()):
                flags.add("owned_road_branch")
            if any(degree[n] >= 2 and owner != color
                   for n, (owner, _) in data["buildings"].items()):
                flags.add("effective_enemy_junction")
        counts.update(flags)
    return {
        "unique_states": len(unique),
        "state_counts": dict(sorted(counts.items())),
        "roads_range": [min(roads), max(roads)],
        "buildings_range": [min(buildings), max(buildings)],
        "cities_range": [min(cities), max(cities)],
    }


def pips(number: int | None) -> int:
    return 0 if number is None else 6 - abs(7 - number)


def text_answer(value: object) -> str:
    if isinstance(value, set):
        return " ".join(sorted(value)) or "NONE"
    if type(value) is int:
        return str(value)
    return compact(value)


def production_change(before: dict[str, int], after: dict[str, int]) -> JsonLikeDict:
    return {"before": before, "after": after,
            "delta": {r: after[r] - before[r] for r in RESOURCES}}


class Facts:
    """Detached computations; local hypothetical parameters never mutate a donor."""

    def __init__(self, data: DecodedState, atlas: Atlas) -> None:
        self.data, self.atlas = data, atlas
        self._components: dict[str, list[set[str]]] = {}
        self._distances: dict[tuple[str, str, str | None, bool], dict[str, int]] = {}
        self.owned_nodes = {
            c: {n for n, (owner, _) in data["buildings"].items() if owner == c}
            for c in data["colors"]
        }
        self.owned_roads = {
            c: {e for e, owner in data["roads"].items() if owner == c}
            for c in data["colors"]
        }
        self.coverage = {
            c: {data["tiles"][t][0] for n in nodes for t in atlas["node_tiles"][n]
                if data["tiles"][t][0] != "desert"}
            for c, nodes in self.owned_nodes.items()
        }
        self.resource_pips = {
            r: sum(pips(number) for resource, number in data["tiles"].values() if resource == r)
            for r in RESOURCES
        }

    def enemy(self, color: str, node: str) -> bool:
        piece = self.data["buildings"].get(node)
        return piece is not None and piece[0] != color

    def effective_blockers(self, color: str) -> set[str]:
        return {n for n in self.data["buildings"] if self.enemy(color, n)
                and len(self.owned_roads[color] & self.atlas["node_edges"][n]) >= 2}

    def components(self, color: str) -> list[set[str]]:
        if color not in self._components:
            remaining = set(self.owned_roads[color])
            parts = []
            while remaining:
                start = min(remaining)
                group, queue = {start}, deque([start])
                remaining.remove(start)
                while queue:
                    edge = queue.popleft()
                    for node in self.atlas["edges"][edge]:
                        if self.enemy(color, node):
                            continue
                        joined = remaining & self.atlas["node_edges"][node]
                        for neighbor in sorted(joined):
                            remaining.remove(neighbor)
                            group.add(neighbor)
                            queue.append(neighbor)
                parts.append(group)
            self._components[color] = parts
        return self._components[color]

    def distances(self, color: str, start: str, removed: str | None = None,
                  ignore_blockers: bool = False) -> dict[str, int]:
        key = color, start, removed, ignore_blockers
        if key not in self._distances:
            roads = self.owned_roads[color] - ({removed} if removed else set())
            distance, queue = {start: 0}, deque([start])
            while queue:
                node = queue.popleft()
                if node != start and not ignore_blockers and self.enemy(color, node):
                    continue
                for edge in sorted(roads & self.atlas["node_edges"][node]):
                    a, b = self.atlas["edges"][edge]
                    neighbor = b if a == node else a
                    if neighbor not in distance:
                        distance[neighbor] = distance[node] + 1
                        queue.append(neighbor)
            self._distances[key] = distance
        return self._distances[key]

    def production(self, color: str, roll: int, *, upgrade: str | None = None,
                   robber: str | None = None) -> dict[str, int]:
        result = dict.fromkeys(RESOURCES, 0)
        blocked = self.data["robber"] if robber is None else robber
        for node in self.owned_nodes[color]:
            amount = 2 if node == upgrade or self.data["buildings"][node][1] == "city" else 1
            for tile in self.atlas["node_tiles"][node]:
                resource, number = self.data["tiles"][tile]
                if tile != blocked and resource != "desert" and number == roll:
                    result[resource] += amount
        return result

    def solve(self, op: str, q: QueryDict) -> object:
        data, atlas = self.data, self.atlas
        color = cast("str", q.get("color"))
        if color is not None:
            require(color in data["colors"], "query has nonparticipant color")
        if "node" in q:
            require(q["node"] in atlas["graph"], "invalid query node")
        if op == "owned_buildings_touching_resource":
            return {n for n in self.owned_nodes[color]
                    if any(data["tiles"][t][0] == q["resource"] for t in atlas["node_tiles"][n])}
        if op == "owned_incident_roads":
            return self.owned_roads[color] & atlas["node_edges"][cast("str", q["node"])]
        if op == "local_node_tiles":
            return {t: {"resource": data["tiles"][t][0], "number": data["tiles"][t][1]}
                    for t in sorted(atlas["node_tiles"][cast("str", q["node"])])}
        if op == "port_access":
            return {p for n in self.owned_nodes[color] for p in atlas["node_ports"].get(n, set())}
        if op.startswith("coverage_"):
            a = self.coverage[cast("str", q.get("a", color))]
            if op == "coverage_missing":
                return set(RESOURCES) - a
            require(q["a"] != q["b"], "coverage comparison needs different players")
            b = self.coverage[cast("str", q["b"])]
            if op == "coverage_union":
                return a | b
            if op == "coverage_intersection":
                return a & b
            if op == "coverage_difference":
                return a - b
        if op == "resource_pip_totals":
            return self.resource_pips
        if op == "resource_pip_argmax":
            maximum = max(self.resource_pips.values())
            return {r for r, total in self.resource_pips.items() if total == maximum}
        if op == "node_pip_sum":
            return sum(pips(data["tiles"][t][1]) for t in atlas["node_tiles"][cast("str", q["node"])])
        if op == "roll_production":
            return self.production(color, cast("int", q["roll"]))
        if op == "component_roads":
            require(q["edge"] in self.owned_roads[color], "component seed is not an owned road")
            return next(part for part in self.components(color) if q["edge"] in part)
        if op == "component_count":
            return len(self.components(color))
        if op == "shortest_distance":
            require(q["start"] != q["end"], "zero-hop route is not a review candidate")
            distance = self.distances(color, cast("str", q["start"])).get(cast("str", q["end"]))
            return "UNREACHABLE" if distance is None else distance
        if op == "reachable_nodes":
            return set(self.distances(color, cast("str", q["start"]))) - {q["start"]}
        if op == "distance_rule_witnesses":
            return atlas["graph"][cast("str", q["node"])] & data["buildings"].keys()
        if op == "road_removal_connectivity":
            require(q["remove_edge"] in self.owned_roads[color], "removed road is not present/owned")
            return {
                "before": self.distances(color, cast("str", q["start"])).get(cast("str", q["end"])),
                "after": self.distances(color, cast("str", q["start"]),
                                        cast("str", q["remove_edge"])).get(cast("str", q["end"])),
            }
        if op == "settlement_upgrade_production":
            require(data["buildings"].get(cast("str", q["node"])) == (color, "settlement"),
                    "upgrade must refer to an actually present owned settlement")
            return production_change(self.production(color, cast("int", q["roll"])),
                                     self.production(color, cast("int", q["roll"]),
                                                     upgrade=cast("str", q["node"])))
        if op == "robber_move_production":
            require(q["from_tile"] == data["robber"] and q["to_tile"] in data["tiles"]
                    and q["to_tile"] != q["from_tile"], "invalid hypothetical robber move")
            return production_change(self.production(color, cast("int", q["roll"])),
                                     self.production(color, cast("int", q["roll"]),
                                                     robber=cast("str", q["to_tile"])))
        raise ValueError(f"unsupported review operation: {op}")
