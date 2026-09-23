from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import TypeAlias

import networkx

from sft.board.symbolic_board_tasks import atlas_geometry
from sft.json_types import JsonLikeDict
from sft.scripts.builders import build_board_fluency_review as build_board_fluency_review

from ._selection import apportion, quotas, roster_positions
from ._sources import OPERATIONS, PIP_OPERATIONS, Donor, require

nx = networkx
review = build_board_fluency_review

Candidate: TypeAlias = review.Candidate[Donor]
# (operation, stratum, queried roster positions)
Group: TypeAlias = tuple[str, str, tuple[int, ...]]
# "source"/"sink", ("state"|"reuse", hash), ("terrain", operation, terrain hash), or a group.
Node: TypeAlias = str | tuple[str, str] | tuple[str, str, str] | Group


def select(donors: list[Donor], split: str, seed: int) -> tuple[list[Candidate], JsonLikeDict]:
    """Min-cost integral flow: exact quotas, unique questions, maximum distinct states.

    Dynamic queries partition by operation/stratum/queried roster positions.
    Pip queries pass through a capacity-one terrain gate per operation. A free
    first use of every state and costly later uses maximize distinct state use.
    """
    rng, atlas = random.Random(seed), atlas_geometry()
    bank: defaultdict[Group, defaultdict[str, list[Candidate]]] = defaultdict(lambda: defaultdict(list))
    for donor in donors:
        for c in review.candidates_for(donor, atlas, rng):
            bank[c.operation, c.stratum, roster_positions(c)][donor.state_hash].append(c)
    by_state = {d.state_hash: d for d in donors}
    capacities: dict[Group, int] = {}
    for group, states in bank.items():
        capacities[group] = (len({by_state[h].terrain_hash for h in states}) if group[0] in PIP_OPERATIONS
                             else sum(len(cs) for cs in states.values()))
    requested: dict[Group, int] = {}
    for op, count in quotas(split).items():
        cells = sorted({g[1] for g in bank if g[0] == op})
        cell_counts = apportion({cell: sum(n for g, n in capacities.items() if g[:2] == (op, cell))
                                 for cell in cells}, count)
        for cell, n in cell_counts.items():
            requested.update(apportion({g: capacity for g, capacity in capacities.items()
                                         if g[:2] == (op, cell)}, n))
    total = sum(requested.values())
    graph: nx.DiGraph[Node, dict[str, int], dict[str, int]] = nx.DiGraph()
    graph.add_node("source", demand=-total)
    graph.add_node("sink", demand=total)
    state_order = sorted(by_state)
    rng.shuffle(state_order)
    # Losing one distinct state costs more than every possible rank tie combined.
    reuse_cost = (len(state_order) + 1) * total
    for rank, h in enumerate(state_order):
        graph.add_edge(("state", h), "sink", capacity=1, weight=rank)
        graph.add_edge(("state", h), ("reuse", h), capacity=3, weight=reuse_cost + rank)
        graph.add_edge(("reuse", h), "sink", capacity=total, weight=0)
    for group, n in sorted(requested.items()):
        if not n:
            continue
        graph.add_edge("source", group, capacity=n, weight=0)
        for h, candidates in sorted(bank[group].items()):
            if group[0] in PIP_OPERATIONS:
                terrain = ("terrain", group[0], by_state[h].terrain_hash)
                graph.add_edge(group, terrain, capacity=1, weight=0)
                graph.add_edge(terrain, ("state", h), capacity=1, weight=0)
            else:
                graph.add_edge(group, ("state", h), capacity=len(candidates), weight=0)
    cap, failed_caps = 4, list[int]()
    flow: dict[Node, dict[Node, int]]
    while True:
        try:
            _, flow = nx.network_simplex(graph)
            break
        except nx.NetworkXUnfeasible:
            failed_caps.append(cap)
            # Training stays within the requested cap. Heldout has only existing
            # states: raise its cap only after proving the previous cap infeasible.
            require(split != "train" and cap < total,
                    f"{split}: quotas unsupported at {cap} queries/state; no corpus written")
            cap += 1
            for h in state_order:
                graph[("state", h)][("reuse", h)]["capacity"] = cap - 1
    selected: list[Candidate] = []
    query_uses: Counter[tuple[str, str, str]] = Counter()
    for group, n in sorted(requested.items()):
        if not n:
            continue
        assignments: Counter[str] = Counter()
        for node, amount in flow[group].items():
            if not amount:
                continue
            if node[0] == "terrain":
                assignments.update({dest[1]: used for dest, used in flow[node].items() if used})
            else:
                assignments[node[1]] += amount
        for h, amount in sorted(assignments.items()):
            options = list(bank[group][h])
            for _ in range(amount):
                chosen = min(options, key=lambda c: (
                    sum(query_uses[c.operation, k, str(v)] for k, v in c.query.items()),
                    -c.quality, c.tie_rank))
                options.remove(chosen)
                selected.append(chosen)
                query_uses.update((chosen.operation, k, str(v)) for k, v in chosen.query.items())
    require(len(selected) == total, "flow extraction did not fill quotas")
    used = Counter(c.donor.state_hash for c in selected)
    selected_counts = Counter((c.operation, c.stratum, roster_positions(c)) for c in selected)
    support: list[JsonLikeDict] = [{"operation": g[0], "stratum": g[1], "roster_positions": list(g[2]),
                "eligible_unique_states": len(bank[g]), "unique_query_capacity": capacities[g],
                "selected": selected_counts[g]} for g in sorted(bank)]
    missing = [{"operation": op, "stratum": cell} for op in OPERATIONS for cell in review.CELLS[op]
               if not any(g[:2] == (op, cell) for g in bank)]
    return interleave(selected, seed + 100), {
        "initial_state_cap": 4, "state_cap": cap, "proven_infeasible_caps": failed_caps,
        "maximum_distinct_states_at_cap": len(used), "max_presentations_per_state": max(used.values()),
        "support_by_cell_and_roster": support, "absent_review_strata": missing,
        "retained_candidate_queries": sum(len(cs) for states in bank.values() for cs in states.values()),
        "policy": "Equal supported strata, then equal supported queried roster positions, saturating finite "
                  "capacity; integral min-cost flow maximizes distinct states under the cap. "
                  "One question per terrain per pip operation. Reviewed candidate enumeration only.",
    }


def interleave(candidates: list[Candidate], seed: int) -> list[Candidate]:
    rng = random.Random(seed)
    groups: dict[str, list[Candidate]] = {op: [] for op in OPERATIONS}
    for c in candidates:
        groups[c.operation].append(c)
    for values in groups.values():
        rng.shuffle(values)
    return [groups[op][i] for i in range(max(map(len, groups.values())))
            for op in OPERATIONS if i < len(groups[op])]
