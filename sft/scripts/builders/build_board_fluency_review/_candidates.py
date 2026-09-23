from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sized
from dataclasses import dataclass
from itertools import combinations
from typing import Generic, Protocol, TypeVar, cast

from sft.board.symbolic_board_tasks._types import Atlas, DecodedState
from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_str

from ._facts import Facts
from ._sources import (
    CELLS,
    OPERATION_FAMILY,
    RESOURCES,
    Donor,
    QueryDict,
    compact,
    require,
)


class DonorView(Protocol):
    """What candidates and oracles read from a donor; the admitted SFT corpus
    reuses them with its own source-backed donor, which has no review lineage."""

    @property
    def state(self) -> Mapping[str, object]: ...
    @property
    def data(self) -> DecodedState: ...
    @property
    def provenance(self) -> JsonDict: ...
    @property
    def state_hash(self) -> str: ...


DonorT = TypeVar("DonorT", bound=DonorView)


@dataclass
class Candidate(Generic[DonorT]):
    donor: DonorT
    operation: str
    query: QueryDict
    stratum: str
    quality: int
    tie_rank: float


def candidates_for(donor: DonorT, atlas: Atlas, rng: random.Random) -> list[Candidate[DonorT]]:
    """Enumerate real-state queries, then retain a small varied bank per semantic cell."""
    data, facts = donor.data, Facts(donor.data, atlas)
    if not data["buildings"] or not data["roads"]:
        return []  # No empty-board negatives or map-only donors for this review.
    buckets: defaultdict[tuple[str, str], list[Candidate[DonorT]]] = defaultdict(list)

    def add(op: str, q: QueryDict, stratum: str, quality: int = 0) -> None:
        buckets[op, stratum].append(Candidate(donor, op, q, stratum, quality, rng.random()))

    def add_set(op: str, q: QueryDict, quality: int = 0) -> None:
        result = facts.solve(op, q)
        add(op, q, "nonempty" if result else "empty",
            quality + min(len(cast("Sized", result)), 3))

    for node in sorted(atlas["graph"]):
        tiles = atlas["node_tiles"][node]
        if len(tiles) >= 2:
            desert = any(data["tiles"][t][0] == "desert" for t in tiles)
            add("local_node_tiles", {"node": node}, "includes_desert" if desert else "resource_only",
                len(tiles))
            add("node_pip_sum", {"node": node}, "sum", len(tiles))
        # The query is atomic even when selection chooses currently empty nodes.
        if node not in data["buildings"]:
            add_set("distance_rule_witnesses", {"node": node})
    add("resource_pip_totals", {}, "totals")
    leaders = facts.solve("resource_pip_argmax", {})
    add("resource_pip_argmax", {}, "tied" if len(cast("Sized", leaders)) > 1 else "unique")

    active = sorted(c for c in data["colors"] if facts.owned_nodes[c])
    for a, b in combinations(active, 2):
        ca, cb = facts.coverage[a], facts.coverage[b]
        if ca - cb and cb - ca:
            add("coverage_union", {"a": a, "b": b}, "complementary", len(ca & cb))
        add_set("coverage_intersection", {"a": a, "b": b})
        add_set("coverage_difference", {"a": a, "b": b})
        add_set("coverage_difference", {"a": b, "b": a})

    for color in active:
        for resource in RESOURCES:
            add_set("owned_buildings_touching_resource", {"color": color, "resource": resource})
        add_set("port_access", {"color": color})
        add_set("coverage_missing", {"color": color})
        rolls = sorted({number for _, number in data["tiles"].values() if number is not None})
        for roll in rolls:
            production = facts.production(color, roll)
            amount = sum(production.values())
            blocked_loss = (data["tiles"][data["robber"]][1] == roll
                            and any(data["robber"] in atlas["node_tiles"][n]
                                    for n in facts.owned_nodes[color]))
            add("roll_production", {"color": color, "roll": roll},
                "positive" if amount else "zero", min(amount, 4) + 2 * blocked_loss)
        for node in sorted(facts.owned_nodes[color]):
            if data["buildings"][node][1] != "settlement":
                continue
            local_rolls = sorted({number for number in
                                  (data["tiles"][t][1] for t in atlas["node_tiles"][node])
                                  if number is not None})
            for roll in local_rolls:
                q: QueryDict = {"color": color, "node": node, "roll": roll}
                change = cast("dict[str, dict[str, int]]",
                              facts.solve("settlement_upgrade_production", q))
                gain = sum(change["delta"].values())
                add("settlement_upgrade_production", q, "gain" if gain else "blocked_zero",
                    gain + min(sum(change["before"].values()), 3))
        old_tile = data["robber"]
        for tile in sorted(data["tiles"]):
            if tile == old_tile:
                continue
            affected_rolls = {number for number in
                              (data["tiles"][t][1] for t in (old_tile, tile))
                              if number is not None}
            for roll in sorted(affected_rolls):
                q = {"color": color, "from_tile": old_tile, "to_tile": tile, "roll": roll}
                moved = cast("dict[str, dict[str, int]]",
                             facts.solve("robber_move_production", q))
                values = list(moved["delta"].values())
                positive, negative = any(v > 0 for v in values), any(v < 0 for v in values)
                cell = ("mixed" if positive and negative else "gain" if positive else
                        "loss" if negative else "unchanged")
                # A zero consequence should still have actual baseline production.
                if cell != "unchanged" or sum(moved["before"].values()) > 0:
                    add("robber_move_production", q, cell, sum(abs(v) for v in values))

    for color in sorted(data["colors"]):
        roads = facts.owned_roads[color]
        if len(roads) < 2:
            continue
        blockers = facts.effective_blockers(color)
        endpoints = sorted({n for e in roads for n in atlas["edges"][e]})
        for node in sorted(atlas["graph"]):
            if atlas["node_edges"][node] & data["roads"].keys():
                add_set("owned_incident_roads", {"color": color, "node": node})
        parts = facts.components(color)
        if len(roads) >= 3:
            add("component_count", {"color": color}, "multiple" if len(parts) > 1 else "single",
                int(bool(blockers)) + min(len(roads), 5))
        for part in parts:
            boundary = {n for e in part for n in atlas["edges"][e]} & blockers
            for edge in sorted(part):
                add("component_roads", {"color": color, "edge": edge},
                    "blocked_boundary" if boundary else "ordinary", min(len(part), 6))
        for start in endpoints:
            distances = facts.distances(color, start)
            if len(distances) >= 3:
                add("reachable_nodes", {"color": color, "start": start},
                    "enemy_start" if facts.enemy(color, start) else "ordinary_start",
                    min(max(distances.values()), 6) + int(start in blockers))
            for end in endpoints:
                if start >= end:
                    continue
                distance = distances.get(end)
                if distance is not None and distance < 2:
                    continue
                if distance is not None:
                    cell, quality = "reachable", min(distance, 6)
                else:
                    raw_connected = end in facts.distances(color, start, ignore_blockers=True)
                    cell, quality = "blocked_unreachable" if raw_connected else "disconnected", 0
                add("shortest_distance", {"color": color, "start": start, "end": end}, cell, quality)
        # Removing an existing edge tests bridges versus real alternative routes.
        # Endpoints are explicit; the removed edge is never an irrelevant road.
        for edge in sorted(roads):
            start, end = atlas["edges"][edge]
            after = facts.distances(color, start, removed=edge).get(end)
            q = {"color": color, "remove_edge": edge, "start": start, "end": end}
            add("road_removal_connectivity", q, "disconnects" if after is None else "alternate_route",
                0 if after is None else min(after, 8))

    # Keep high-value examples while leaving alternatives for color/token diversity.
    retained: list[Candidate[DonorT]] = []
    for key in sorted(buckets):
        ranked = sorted(buckets[key], key=lambda c: (-c.quality, c.tie_rank))
        retained.extend(ranked[:16])
    return retained


def select(donors: list[Donor], atlas: Atlas,
           seed: int) -> tuple[list[Candidate[Donor]], JsonLikeDict]:
    rng = random.Random(seed)
    bank = [c for d in donors for c in candidates_for(d, atlas, rng)]
    by_cell: defaultdict[tuple[str, str], defaultdict[str, list[Candidate[Donor]]]] = (
        defaultdict(lambda: defaultdict(list)))
    for candidate in bank:
        by_cell[candidate.operation, candidate.stratum][candidate.donor.state_hash].append(candidate)
    slots = [(op, cell, i) for op in OPERATION_FAMILY for cell, count in CELLS[op].items()
             for i in range(count)]
    require(len(slots) == 200, "review slot count must equal 200")
    for op, cell, _ in slots:
        require(bool(by_cell[op, cell]), f"unsupported required cell in bounded prefix: {op}/{cell}")
    order = sorted(range(len(slots)), key=lambda i: (len(by_cell[slots[i][:2]]), i))
    state_donors = {d.state_hash: d for d in donors}
    shuffled = sorted(state_donors)
    rng.shuffle(shuffled)
    ranks = {state: i for i, state in enumerate(shuffled)}
    assigned: dict[int, str] = {}
    owner: dict[str, int] = {}
    map_uses: Counter[str] = Counter()
    kind_uses: Counter[str] = Counter()
    density_uses: Counter[str] = Counter()

    def preference(state: str) -> tuple[int, int, int, int]:
        p = state_donors[state].provenance
        return (map_uses[as_str(p["board_map_sha256"])],
                kind_uses[as_str(as_dict(p["source"])["kind"])],
                density_uses[as_str(p["density_bin"])], ranks[state])

    def augment(slot: int, visited: set[str]) -> bool:
        for state in sorted(by_cell[slots[slot][:2]], key=preference):
            if state in visited:
                continue
            visited.add(state)
            previous = owner.get(state)
            if previous is None or augment(previous, visited):
                assigned[slot] = state
                owner[state] = slot
                return True
        return False

    unmatched = []
    for slot in order:
        options = [state for state in by_cell[slots[slot][:2]] if state not in owner]
        if options:
            state = min(options, key=preference)
            assigned[slot], owner[state] = state, slot
        elif not augment(slot, set()):
            unmatched.append(slot)
        map_uses.clear()
        kind_uses.clear()
        density_uses.clear()
        for state in assigned.values():
            p = state_donors[state].provenance
            map_uses[as_str(p["board_map_sha256"])] += 1
            kind_uses[as_str(as_dict(p["source"])["kind"])] += 1
            density_uses[as_str(p["density_bin"])] += 1

    # Maximum matching certifies maximum possible distinct state use for these cells.
    maximum_distinct = len(assigned)
    state_uses = Counter(assigned.values())
    for slot in unmatched:
        state = min(by_cell[slots[slot][:2]], key=lambda s: (state_uses[s], preference(s)))
        assigned[slot] = state
        state_uses[state] += 1
    selected: list[Candidate[Donor]] = []
    used_queries: set[tuple[str, str, str]] = set()
    query_uses: Counter[tuple[str, str, str]] = Counter()
    for slot, (op, cell, _) in enumerate(slots):
        pool = [c for c in by_cell[op, cell][assigned[slot]]
                if (c.donor.state_hash, op, compact(c.query)) not in used_queries]
        require(bool(pool), "insufficient unique state/query candidates; no duplicate padding allowed")
        chosen = min(pool, key=lambda c: (
            sum(query_uses[op, key, str(value)] for key, value in c.query.items()),
            -c.quality, c.tie_rank,
        ))
        selected.append(chosen)
        used_queries.add((chosen.donor.state_hash, op, compact(chosen.query)))
        query_uses.update((op, key, str(value)) for key, value in chosen.query.items())
    support: list[JsonLikeDict] = [
        {"operation": op, "stratum": cell, "eligible_unique_states": len(states),
         "retained_candidate_queries": sum(len(cs) for cs in states.values()),
         "requested": CELLS.get(op, {}).get(cell, 0),
         "selected": sum(c.operation == op and c.stratum == cell for c in selected)}
        for (op, cell), states in sorted(by_cell.items())
    ]
    return selected, {
        "seed": seed, "retained_candidate_count": len(bank),
        "maximum_distinct_states_for_required_cells": maximum_distinct,
        "support_by_cell": support,
        "policy": "Seeded candidate ties; scarce cells first, maximum state-slot matching; "
                  "prefer unused maps, underrepresented source kinds/densities and query parameters. "
                  "No fabricated state and no duplicate state/query padding.",
    }
