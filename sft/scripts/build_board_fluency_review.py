"""Build the bounded, text-only 200-example board-fluency human review batch.

This is a review schema, not an admitted trainer task schema. Only the first
MAX_SOURCE_ROWS physical source lines are read; the full source hash is merely
reported from its manifest. Source states and original provenance stay intact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import combinations, islice
from pathlib import Path

from data_pipeline.board_recognition.replay_dataset import static_board_facts
from data_pipeline.board_recognition.sources import (
    canonical_sha256,
    file_sha256,
    repository_relative,
    visible_board_facts,
)
from evals.catan_board_bench.annotations import contract_to_render_state
from sft.spatial_tasks import dice_production, local_node_tiles
from sft.symbolic_board_tasks import (
    ROUTE_RULES,
    TRAIN_TASKS,
    atlas_geometry,
    decode_state,
    owned_route,
    symbolic_answer,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/generated/sft/symbolic_board_v2/train.jsonl"
OUTPUT = ROOT / "artifacts/generated/sft/symbolic_board_fluency_review_v1"
SCHEMA = "catan_board_fluency_review/v1"
VERSION = "symbolic_board_fluency_review_v1"
MAX_SOURCE_ROWS = 400
SEED = 20260914
RESOURCES = ("brick", "ore", "sheep", "wheat", "wood")
FAMILIES = {
    "relations_joins": (
        "owned_buildings_touching_resource", "owned_incident_roads",
        "local_node_tiles", "port_access",
    ),
    "sets_coverage": (
        "coverage_union", "coverage_intersection", "coverage_difference", "coverage_missing",
    ),
    "aggregation_comparison": (
        "resource_pip_totals", "resource_pip_argmax", "node_pip_sum", "roll_production",
    ),
    "connectivity_structure": (
        "component_roads", "component_count", "shortest_distance", "reachable_nodes",
    ),
    "constraints_consequences": (
        "distance_rule_witnesses", "road_removal_connectivity",
        "settlement_upgrade_production", "robber_move_production",
    ),
}
OPERATION_FAMILY = {op: family for family, ops in FAMILIES.items() for op in ops}
# Every operation gets ten rows. These are semantic cells, not answer padding.
CELLS = {
    "owned_buildings_touching_resource": {"nonempty": 8, "empty": 2},
    "owned_incident_roads": {"nonempty": 8, "empty": 2},
    "local_node_tiles": {"resource_only": 8, "includes_desert": 2},
    "port_access": {"nonempty": 8, "empty": 2},
    "coverage_union": {"complementary": 10},
    "coverage_intersection": {"nonempty": 8, "empty": 2},
    "coverage_difference": {"nonempty": 8, "empty": 2},
    "coverage_missing": {"nonempty": 8, "empty": 2},
    "resource_pip_totals": {"totals": 10},
    "resource_pip_argmax": {"tied": 5, "unique": 5},
    "node_pip_sum": {"sum": 10},
    "roll_production": {"positive": 8, "zero": 2},
    "component_roads": {"blocked_boundary": 2, "ordinary": 8},
    "component_count": {"multiple": 5, "single": 5},
    "shortest_distance": {"reachable": 6, "blocked_unreachable": 2, "disconnected": 2},
    "reachable_nodes": {"enemy_start": 2, "ordinary_start": 8},
    "distance_rule_witnesses": {"nonempty": 8, "empty": 2},
    "road_removal_connectivity": {"disconnects": 5, "alternate_route": 5},
    "settlement_upgrade_production": {"gain": 8, "blocked_zero": 2},
    "robber_move_production": {"gain": 4, "loss": 4, "unchanged": 2},
}
SET_FORMAT = "Output only the sorted, space-separated set; NONE if empty. No explanation."
VECTOR_FORMAT = (
    'Output only compact JSON with all five lowercase resource keys '
    '"brick", "ore", "sheep", "wheat", "wood" and integer values.'
)
PIP_RULES = (
    "Pips are the number of two-dice outcomes: 2/12=1, 3/11=2, 4/10=3, "
    "5/9=4, 6/8=5; desert=0. Ignore the robber and all buildings; count each "
    "tile once. "
)
PRODUCTION_RULES = (
    "Count production from all of that player's existing buildings: a settlement "
    "receives 1 card and a city 2 per touching tile with the rolled number; "
    "the robber blocks its tile. Ignore bank supply. "
)
COVERAGE_RULES = (
    "A player's resource coverage is the set of resource types on tiles touching "
    "at least one of their settlements or cities. Exclude desert; ignore the "
    "robber and roads; include each resource type only once. "
)
COMPONENT_RULES = (
    "Partition the player's existing road edges into traversable components: "
    "two edges can join through a shared node unless an opponent building "
    "occupies that junction. Each edge belongs to exactly one component; "
    "blocked boundary nodes may occur in more than one component. "
)
CHANGE_FORMAT = (
    'Output only compact JSON with "before", "after", and "delta", each a '
    'five-resource integer object (brick, ore, sheep, wheat, wood). '
    'Delta means after minus before.'
)
OWN_FILES = frozenset({"review.jsonl", "metadata.json", "preview.json"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass
class Donor:
    line: int
    row: dict
    state: dict
    data: dict
    provenance: dict
    state_hash: str
    line_sha256: str


def load_prefix() -> tuple[list[Donor], dict]:
    """Read at most 400 physical lines, including static rows and blank lines."""
    manifest_path = SOURCE.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    declaration = manifest["files"]["train"]
    require(Path(declaration["path"]).resolve() == SOURCE.resolve(), "manifest source path mismatch")
    require(declaration["rows"] == manifest["counts"]["train"] == 3200,
            "unexpected declared training corpus size")
    digest, byte_count, physical_lines = hashlib.sha256(), 0, 0
    donors, skipped, seen = [], Counter(), set()
    with SOURCE.open("rb") as handle:
        for line, raw in enumerate(islice(handle, MAX_SOURCE_ROWS), 1):
            physical_lines = line
            digest.update(raw)
            byte_count += len(raw)
            if not raw.strip():
                skipped["blank_physical_lines"] += 1
                continue
            row = json.loads(raw)
            m = row["metadata"]
            task = row["task_type"]
            require(row["schema"] == "catan_symbolic_board_row/v2", f"source schema: {line}")
            require(task in TRAIN_TASKS, f"non-training task: {line}")
            for key, value in (("split", "train"), ("task_role", "train"),
                               ("task_type", task), ("training_family", task)):
                require(row.get(key) == m.get(key) == value, f"source declaration {key}: {line}")
            require(row["id"] == row["row_id"] and row["row_id"] not in seen,
                    f"duplicate/mismatched donor ID: {line}")
            seen.add(row["row_id"])
            require(m["row_position"] == line - 1, f"source physical position: {line}")
            state = m["target"]["state"]
            if state is None:
                skipped["static_rows_without_state"] += 1
                continue
            p = m["provenance"]
            require(p["split"] == "train", f"non-training donor provenance: {line}")
            require(p["source"].get("split", "train") == "train",
                    f"non-training underlying source: {line}")
            require(isinstance(p["source"]["kind"], str) and p["source"]["kind"],
                    f"missing original source kind: {line}")
            for key in ("state_id", "density_bin", "board_map_sha256", "board_fact_sha256"):
                require(isinstance(p[key], str) and p[key], f"missing provenance {key}: {line}")
            require(p["paths"]["contract"] and len(p["sha256"]["contract"]) == 64,
                    f"missing original contract/hash: {line}")
            donors.append(Donor(line, row, state, decode_state(state), p,
                                canonical_sha256(state), hashlib.sha256(raw).hexdigest()))
    require(physical_lines == MAX_SOURCE_ROWS, "source prefix shorter than required 400 lines")
    audit = {
        "path": repository_relative(SOURCE),
        "max_physical_lines": MAX_SOURCE_ROWS,
        "physical_lines_read": physical_lines,
        "prefix_bytes": byte_count,
        "verified_prefix_sha256": digest.hexdigest(),
        "prefix_hash_scope": "Exact raw bytes of physical lines 1 through 400, including newlines",
        "manifest_path": repository_relative(manifest_path),
        "verified_manifest_file_sha256": file_sha256(manifest_path),
        "manifest_declared_full_rows": declaration["rows"],
        "manifest_declared_full_sha256": declaration["sha256"],
        "full_corpus_hash_verified": False,
        "skipped": dict(skipped),
        "state_bearing_donor_rows": len(donors),
        "distinct_donor_state_ids": len({d.provenance["state_id"] for d in donors}),
        "distinct_canonical_states": len({d.state_hash for d in donors}),
        "distinct_maps": len({d.provenance["board_map_sha256"] for d in donors}),
        "donor_rows_by_density": dict(Counter(d.provenance["density_bin"] for d in donors)),
        "donor_rows_by_source_kind": dict(Counter(d.provenance["source"]["kind"] for d in donors)),
    }
    return donors, audit


def richness(donors: list[Donor], atlas: dict) -> dict:
    """Observable source facts, without pretending independent rows are new states."""
    unique = {d.state_hash: d for d in donors}
    counts = Counter()
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


def production_change(before: dict, after: dict) -> dict:
    return {"before": before, "after": after,
            "delta": {r: after[r] - before[r] for r in RESOURCES}}


class Facts:
    """Detached computations; local hypothetical parameters never mutate a donor."""

    def __init__(self, data: dict, atlas: dict):
        self.data, self.atlas = data, atlas
        self._components, self._distances = {}, {}
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

    def solve(self, op: str, q: dict) -> object:
        data, atlas = self.data, self.atlas
        color = q.get("color")
        if color is not None:
            require(color in data["colors"], "query has nonparticipant color")
        if "node" in q:
            require(q["node"] in atlas["graph"], "invalid query node")
        if op == "owned_buildings_touching_resource":
            return {n for n in self.owned_nodes[color]
                    if any(data["tiles"][t][0] == q["resource"] for t in atlas["node_tiles"][n])}
        if op == "owned_incident_roads":
            return self.owned_roads[color] & atlas["node_edges"][q["node"]]
        if op == "local_node_tiles":
            return {t: {"resource": data["tiles"][t][0], "number": data["tiles"][t][1]}
                    for t in sorted(atlas["node_tiles"][q["node"]])}
        if op == "port_access":
            return {p for n in self.owned_nodes[color] for p in atlas["node_ports"].get(n, set())}
        if op.startswith("coverage_"):
            a = self.coverage[q.get("a", color)]
            if op == "coverage_missing":
                return set(RESOURCES) - a
            require(q["a"] != q["b"], "coverage comparison needs different players")
            b = self.coverage[q["b"]]
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
            return sum(pips(data["tiles"][t][1]) for t in atlas["node_tiles"][q["node"]])
        if op == "roll_production":
            return self.production(color, q["roll"])
        if op == "component_roads":
            require(q["edge"] in self.owned_roads[color], "component seed is not an owned road")
            return next(part for part in self.components(color) if q["edge"] in part)
        if op == "component_count":
            return len(self.components(color))
        if op == "shortest_distance":
            require(q["start"] != q["end"], "zero-hop route is not a review candidate")
            distance = self.distances(color, q["start"]).get(q["end"])
            return "UNREACHABLE" if distance is None else distance
        if op == "reachable_nodes":
            return set(self.distances(color, q["start"])) - {q["start"]}
        if op == "distance_rule_witnesses":
            return atlas["graph"][q["node"]] & data["buildings"].keys()
        if op == "road_removal_connectivity":
            require(q["remove_edge"] in self.owned_roads[color], "removed road is not present/owned")
            return {
                "before": self.distances(color, q["start"]).get(q["end"]),
                "after": self.distances(color, q["start"], q["remove_edge"]).get(q["end"]),
            }
        if op == "settlement_upgrade_production":
            require(data["buildings"].get(q["node"]) == (color, "settlement"),
                    "upgrade must refer to an actually present owned settlement")
            return production_change(self.production(color, q["roll"]),
                                     self.production(color, q["roll"], upgrade=q["node"]))
        if op == "robber_move_production":
            require(q["from_tile"] == data["robber"] and q["to_tile"] in data["tiles"]
                    and q["to_tile"] != q["from_tile"], "invalid hypothetical robber move")
            return production_change(self.production(color, q["roll"]),
                                     self.production(color, q["roll"], robber=q["to_tile"]))
        raise ValueError(f"unsupported review operation: {op}")


def answer(facts: Facts, op: str, q: dict) -> str:
    result = facts.solve(op, q)
    return result if isinstance(result, str) else text_answer(result)


def question(op: str, q: dict) -> str:
    if op == "owned_buildings_touching_resource":
        return (f"Which nodes hold a {q['color']} settlement or city touching at least one "
                f"{q['resource']} tile? Ignore the robber. " + SET_FORMAT)
    if op == "owned_incident_roads":
        return f"List the existing {q['color']} road edges incident to {q['node']}. " + SET_FORMAT
    if op == "local_node_tiles":
        return (f"For every land tile touching {q['node']}, give its resource and number. "
                'Output only compact JSON keyed by tile token, each value with "resource" '
                'and "number"; use lowercase resources and {"resource":"desert","number":null} '
                "for desert. Include all touching tiles exactly once.")
    if op == "port_access":
        return (f"Which port tokens can {q['color']} access through an existing settlement "
                "or city on an attached node? Roads alone do not grant port access. " + SET_FORMAT)
    if op.startswith("coverage_"):
        if op == "coverage_missing":
            text = (f"Which of brick, ore, sheep, wheat, wood are missing from {q['color']}'s "
                    "coverage?")
        else:
            descriptions = {
                "coverage_union": f"covered by {q['a']} or {q['b']} (their union)",
                "coverage_intersection": f"covered by both {q['a']} and {q['b']} (their intersection)",
                "coverage_difference": f"covered by {q['a']} but not {q['b']} (the ordered difference)",
            }
            text = f"Which resource types are {descriptions[op]}?"
        return COVERAGE_RULES + text + " " + SET_FORMAT
    if op == "resource_pip_totals":
        return PIP_RULES + "Sum tile pips across the entire board separately for each resource. " + VECTOR_FORMAT
    if op == "resource_pip_argmax":
        return (PIP_RULES + "Sum tile pips across the entire board separately for each resource. "
                "Which resource types tie for the largest total? Return every tied maximum. " + SET_FORMAT)
    if op == "node_pip_sum":
        return PIP_RULES + f"What is the sum for land tiles touching {q['node']}? Output only an integer."
    if op == "roll_production":
        return PRODUCTION_RULES + f"What does {q['color']} receive on a roll of {q['roll']}? " + VECTOR_FORMAT
    if op == "component_roads":
        return (COMPONENT_RULES + f"For {q['color']}, list all edges in the component "
                f"containing the existing road {q['edge']}. " + SET_FORMAT)
    if op == "component_count":
        return (COMPONENT_RULES + f"How many nonempty road-edge components does {q['color']} "
                "have? Count isolated owned edges as components. Output only an integer.")
    if op == "shortest_distance":
        return (ROUTE_RULES + f"For {q['color']}, what is the minimum number of road edges "
                f"from {q['start']} to {q['end']}? Output only an integer, or UNREACHABLE.")
    if op == "reachable_nodes":
        return (ROUTE_RULES + f"For {q['color']}, list all nodes reachable from {q['start']}, "
                "excluding the start itself. " + SET_FORMAT)
    if op == "distance_rule_witnesses":
        return (f"List the edge-adjacent nodes of {q['node']} that currently have a settlement "
                "or city of any color. These are witnesses violating only the no-adjacent-building "
                "distance condition; do not test occupancy at the queried node, road connection, "
                "or full placement legality. " + SET_FORMAT)
    if op == "road_removal_connectivity":
        return (ROUTE_RULES + f"For {q['color']}, hypothetically remove only their existing road "
                f"{q['remove_edge']}; leave all other pieces unchanged. What are the shortest "
                f"road distances from {q['start']} to {q['end']} before and after this removal? "
                'Output only compact JSON with integer-or-null "before" and "after"; '
                "null means disconnected. This is a local hypothetical, not a recorded successor.")
    if op == "settlement_upgrade_production":
        return (PRODUCTION_RULES + f"Hypothetically replace {q['color']}'s existing settlement "
                f"at {q['node']} with their city, leaving everything else unchanged. Ignore upgrade "
                f"cost and piece supply. For a roll of {q['roll']}, give {q['color']}'s total "
                "production before and after, and its change. This is a local hypothetical, "
                "not a recorded successor. " + CHANGE_FORMAT)
    if op == "robber_move_production":
        return (PRODUCTION_RULES + f"Hypothetically move the robber from its current tile "
                f"{q['from_tile']} to {q['to_tile']}, leaving all buildings unchanged and ignoring "
                f"stealing. For a subsequent roll of {q['roll']}, give {q['color']}'s total "
                "production before and after, and its change. This is a local hypothetical, "
                "not a recorded successor. " + CHANGE_FORMAT)
    raise ValueError(f"missing question for {op}")


def prompt(state: dict, text: str) -> str:
    return ("Use the fixed learned Catan atlas. Participants: " + " ".join(state["colors"])
            + ".\nBoard: " + state["board"] + "\n" + text)


@dataclass
class Candidate:
    donor: Donor
    operation: str
    query: dict
    stratum: str
    quality: int
    tie_rank: float


def candidates_for(donor: Donor, atlas: dict, rng: random.Random) -> list[Candidate]:
    """Enumerate real-state queries, then retain a small varied bank per semantic cell."""
    data, facts = donor.data, Facts(donor.data, atlas)
    if not data["buildings"] or not data["roads"]:
        return []  # No empty-board negatives or map-only donors for this review.
    buckets = defaultdict(list)

    def add(op: str, q: dict, stratum: str, quality: int = 0) -> None:
        buckets[op, stratum].append(Candidate(donor, op, q, stratum, quality, rng.random()))

    def add_set(op: str, q: dict, quality: int = 0) -> None:
        result = facts.solve(op, q)
        add(op, q, "nonempty" if result else "empty", quality + min(len(result), 3))

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
    add("resource_pip_argmax", {}, "tied" if len(leaders) > 1 else "unique")

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
            local_rolls = sorted({data["tiles"][t][1] for t in atlas["node_tiles"][node]
                                  if data["tiles"][t][1] is not None})
            for roll in local_rolls:
                q = {"color": color, "node": node, "roll": roll}
                result = facts.solve("settlement_upgrade_production", q)
                gain = sum(result["delta"].values())
                add("settlement_upgrade_production", q, "gain" if gain else "blocked_zero",
                    gain + min(sum(result["before"].values()), 3))
        old_tile = data["robber"]
        for tile in sorted(data["tiles"]):
            if tile == old_tile:
                continue
            affected_rolls = {data["tiles"][t][1] for t in (old_tile, tile)} - {None}
            for roll in sorted(affected_rolls):
                q = {"color": color, "from_tile": old_tile, "to_tile": tile, "roll": roll}
                result = facts.solve("robber_move_production", q)
                values = list(result["delta"].values())
                positive, negative = any(v > 0 for v in values), any(v < 0 for v in values)
                cell = ("mixed" if positive and negative else "gain" if positive else
                        "loss" if negative else "unchanged")
                # A zero consequence should still have actual baseline production.
                if cell != "unchanged" or sum(result["before"].values()) > 0:
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
    result = []
    for key in sorted(buckets):
        values = sorted(buckets[key], key=lambda c: (-c.quality, c.tie_rank))
        result.extend(values[:16])
    return result


def select(donors: list[Donor], atlas: dict, seed: int) -> tuple[list[Candidate], dict]:
    rng = random.Random(seed)
    bank = [c for d in donors for c in candidates_for(d, atlas, rng)]
    by_cell = defaultdict(lambda: defaultdict(list))
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
    assigned, owner = {}, {}
    map_uses, kind_uses, density_uses = Counter(), Counter(), Counter()

    def preference(state: str) -> tuple:
        p = state_donors[state].provenance
        return (map_uses[p["board_map_sha256"]], kind_uses[p["source"]["kind"]],
                density_uses[p["density_bin"]], ranks[state])

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
            map_uses[p["board_map_sha256"]] += 1
            kind_uses[p["source"]["kind"]] += 1
            density_uses[p["density_bin"]] += 1

    # Maximum matching certifies maximum possible distinct state use for these cells.
    maximum_distinct = len(assigned)
    state_uses = Counter(assigned.values())
    for slot in unmatched:
        state = min(by_cell[slots[slot][:2]], key=lambda s: (state_uses[s], preference(s)))
        assigned[slot] = state
        state_uses[state] += 1
    selected, used_queries, query_uses = [], set(), Counter()
    for slot, (op, cell, _) in enumerate(slots):
        options = by_cell[op, cell][assigned[slot]]
        options = [c for c in options
                   if (c.donor.state_hash, op, compact(c.query)) not in used_queries]
        require(bool(options), "insufficient unique state/query candidates; no duplicate padding allowed")
        chosen = min(options, key=lambda c: (
            sum(query_uses[op, key, str(value)] for key, value in c.query.items()),
            -c.quality, c.tie_rank,
        ))
        selected.append(chosen)
        used_queries.add((chosen.donor.state_hash, op, compact(chosen.query)))
        query_uses.update((op, key, str(value)) for key, value in chosen.query.items())
    support = [
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


def audit_contract(donor: Donor) -> dict:
    p = donor.provenance
    path = Path(p["paths"]["contract"]).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), "original contract path is unavailable")
    require(file_sha256(path) == p["sha256"]["contract"], f"original contract SHA mismatch: {path}")
    contract = json.loads(path.read_text())
    require(contract["sample"]["id"] == p["state_id"], "contract sample identity mismatch")
    require(contract["source"] == p["source"], "original source provenance mismatch")
    require(validate_contract(contract) == donor.state, "contract does not encode exact donor state")
    require(canonical_sha256(visible_board_facts(contract)) == p["board_fact_sha256"],
            "original visible board hash mismatch")
    require(canonical_sha256(static_board_facts(contract)) == p["board_map_sha256"],
            "original board map hash mismatch")
    m = donor.row["metadata"]
    require(m["query_sha256"] == canonical_sha256(m["target"]["query"]), "original query hash mismatch")
    require(symbolic_answer(m["task_type"], m["target"]) == donor.row["messages"][1]["content"],
            "original training donor answer failed recomputation")
    return contract


def contract_components(contract: dict, color: str) -> list[set[str]]:
    """Independent union-find on vertices split per edge at opponent buildings."""
    nodes = {n["id"]: n for n in contract["nodes"]}
    roads = [e for e in contract["edges"] if e["road_color"] == color]
    parent = {}
    edge_vertices = {}

    def root(vertex: tuple) -> tuple:
        parent.setdefault(vertex, vertex)
        while vertex != parent[vertex]:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for edge in roads:
        vertices = []
        for node in edge["id"]:
            enemy = nodes[node]["color"] not in (None, color)
            vertices.append((node, edge["token"] if enemy else "shared"))
        a, b = vertices
        parent[root(a)] = root(b)
        edge_vertices[edge["token"]] = a
    parts = defaultdict(set)
    for edge, vertex in edge_vertices.items():
        parts[root(vertex)].add(edge)
    return list(parts.values())


def reference_answer(candidate: Candidate, contract: dict, checks: Counter) -> str:
    """Check every gold using raw contract joins or existing independent oracles."""
    op, q, state = candidate.operation, candidate.query, candidate.donor.state
    color = q.get("color")
    tiles = {t["token"]: t for t in contract["tiles"]}
    nodes = {n["token"]: n for n in contract["nodes"]}
    result = None
    if op == "owned_buildings_touching_resource":
        result = {n["token"] for n in contract["nodes"] if n["color"] == color
                  and any(tiles[t]["resource"] == q["resource"].upper()
                          for t in n["adjacent_tile_tokens"])}
    elif op == "owned_incident_roads":
        result = symbolic_answer("symbolic_owned_incident_roads", {"state": state, "query": q})
        checks["existing_atomic_oracle_crosschecks"] += 1
    elif op == "local_node_tiles":
        result = local_node_tiles(contract, q["node"])
        checks["existing_local_join_crosschecks"] += 1
    elif op == "port_access":
        result = {p["token"] for p in contract["ports"]
                  if any(nodes[n]["color"] == color for n in p["attached_node_tokens"])}
    elif op.startswith("coverage_"):
        coverage = {}
        for player in state["colors"]:
            touching = {t for n in contract["nodes"] if n["color"] == player
                        for t in n["adjacent_tile_tokens"]}
            coverage[player] = {tiles[t]["resource"].lower() for t in touching
                                if tiles[t]["resource"] is not None}
        if op == "coverage_missing":
            result = {r for r in RESOURCES if r not in coverage[color]}
        else:
            a, b = coverage[q["a"]], coverage[q["b"]]
            if op == "coverage_union":
                result = {r for r in RESOURCES if r in a or r in b}
            elif op == "coverage_intersection":
                result = {r for r in RESOURCES if r in a and r in b}
            else:
                result = {r for r in RESOURCES if r in a and r not in b}
    elif op in ("resource_pip_totals", "resource_pip_argmax", "node_pip_sum"):
        # Count the 36 dice outcomes, independently of the primary arithmetic formula.
        weights = Counter(a + b for a in range(1, 7) for b in range(1, 7))
        if op == "node_pip_sum":
            result = sum(weights[tiles[t]["number"]] for t in nodes[q["node"]]["adjacent_tile_tokens"])
        else:
            totals = dict.fromkeys(RESOURCES, 0)
            for tile in contract["tiles"]:
                if tile["resource"] is not None:
                    totals[tile["resource"].lower()] += weights[tile["number"]]
            result = (totals if op == "resource_pip_totals" else
                      {r for r in RESOURCES if all(totals[r] >= v for v in totals.values())})
    elif op in ("component_roads", "component_count"):
        parts = contract_components(contract, color)
        result = (len(parts) if op == "component_count" else
                  next(part for part in parts if q["edge"] in part))
        checks["split_vertex_component_crosschecks"] += 1
    elif op in ("shortest_distance", "reachable_nodes", "road_removal_connectivity"):
        checks["owned_route_crosschecked_rows"] += 1
        if op == "reachable_nodes":
            result = {n for n in nodes if n != q["start"]
                      and owned_route(state, color, q["start"], n)["edges"] is not None}
            checks["owned_route_calls"] += len(nodes) - 1
        else:
            route = owned_route(state, color, q["start"], q["end"])["edges"]
            checks["owned_route_calls"] += 1
            distance = None if route is None else len(route)
            if op == "shortest_distance":
                result = "UNREACHABLE" if distance is None else distance
            else:
                # This scratch state is exclusively an oracle query condition.
                # It is never rendered, persisted as source, or admitted as a donor.
                edge = q["remove_edge"]
                entries = [f"{edge} empty" if entry.strip().startswith(edge + " ") else entry.strip()
                           for entry in state["board"].split(";")]
                scratch = {"board": "; ".join(entries), "colors": list(state["colors"])}
                changed = decode_state(scratch)
                original = candidate.donor.data
                require(changed["roads"] == {e: c for e, c in original["roads"].items() if e != edge},
                        "hypothetical removed more than one actual road")
                for key in ("tiles", "ports", "buildings", "colors", "robber"):
                    require(changed[key] == original[key], "road-removal hypothetical changed other facts")
                after_route = owned_route(scratch, color, q["start"], q["end"])["edges"]
                checks["owned_route_calls"] += 1
                result = {"before": distance, "after": None if after_route is None else len(after_route)}
    elif op == "distance_rule_witnesses":
        adjacent = {n for e in contract["edges"] if q["node"] in e["node_tokens"]
                    for n in e["node_tokens"] if n != q["node"]}
        result = {n for n in adjacent if nodes[n]["building"] is not None}
        no_neighbor = symbolic_answer("symbolic_local_constraint", {
            "state": state, "query": {"node": q["node"], "color": state["colors"][0],
                                       "predicate": "no_adjacent_building"},
        })
        require((no_neighbor == "yes") == (not result), "distance witness/atomic predicate mismatch")
        checks["existing_atomic_oracle_crosschecks"] += 1
    elif op in ("roll_production", "settlement_upgrade_production", "robber_move_production"):
        before = dice_production(contract, color, q["roll"])
        checks["existing_production_oracle_calls"] += 1
        if op == "roll_production":
            result = before
        else:
            scratch = copy.deepcopy(contract)
            if op == "settlement_upgrade_production":
                node = next(n for n in scratch["nodes"] if n["token"] == q["node"])
                require(node["color"] == color and node["building"] == "SETTLEMENT",
                        "reference upgrade has no actual owned settlement")
                node["building"], node["building_token"] = "CITY", "<CITY>"
            else:
                for tile in scratch["tiles"]:
                    tile["has_robber"] = tile["token"] == q["to_tile"]
                destination = tiles[q["to_tile"]]
                scratch["robber"].update(tile_id=destination["id"], tile_token=destination["token"],
                                         coord=destination["coord"])
            after = dice_production(scratch, color, q["roll"])
            checks["existing_production_oracle_calls"] += 1
            result = {"before": before, "after": after,
                      "delta": {r: after[r] - before[r] for r in RESOURCES}}
    else:
        raise ValueError(f"no independent answer check for {op}")
    checks["independent_answer_crosschecks"] += 1
    return result if isinstance(result, str) else text_answer(result)


def validate_render(contract: dict, render: dict, state: dict) -> None:
    """Human board facts must be exactly the contract that encoded the model state."""
    require(validate_contract(contract) == state, "render source/state mismatch")
    require(render["colors"] == state["colors"], "render participant mismatch")
    require(render["robber_coordinate"] == contract["robber"]["coord"], "render robber mismatch")
    land = {placed["tile"]["id"]: placed for placed in render["tiles"]
            if placed["tile"]["type"] != "PORT"}
    ports = {placed["tile"]["id"]: placed for placed in render["tiles"]
             if placed["tile"]["type"] == "PORT"}
    require(len(land) == 19 and len(ports) == 9, "render terrain completeness")
    for tile in contract["tiles"]:
        placed = land[tile["id"]]
        require(placed["coordinate"] == tile["coord"], "render terrain coordinate")
        require(placed["tile"].get("resource") == tile["resource"]
                and placed["tile"].get("number") == tile["number"], "render terrain facts")
    for port in contract["ports"]:
        placed = ports[port["id"]]
        require(placed["coordinate"] == port["coord"]
                and placed["tile"]["resource"] == port["resource"]
                and placed["tile"]["direction"] == port["direction"]
                and placed["tile"]["port_nodes"] == port["attached_nodes"], "render port facts")
    require(len(render["nodes"]) == 54 and len(render["edges"]) == 72, "render piece completeness")
    for node in contract["nodes"]:
        actual = render["nodes"][str(node["id"])]
        require((actual["id"], actual["color"], actual["building"])
                == (node["id"], node["color"], node["building"]), "render building mismatch")
    actual_roads = {tuple(e["id"]): e["color"] for e in render["edges"]}
    require(actual_roads == {tuple(e["id"]): e["road_color"] for e in contract["edges"]},
            "render road mismatch")


def build(donors: list[Donor], source_audit: dict, seed: int) -> dict[str, bytes]:
    atlas = atlas_geometry()
    selected, selection = select(donors, atlas, seed)
    rows, preview_rows, boards, contracts = [], [], {}, {}
    checks, op_indices = Counter(), Counter()
    for candidate in selected:
        donor, op, q = candidate.donor, candidate.operation, candidate.query
        family = OPERATION_FAMILY[op]
        if donor.state_hash not in contracts:
            contracts[donor.state_hash] = audit_contract(donor)
            checks["original_contract_sha_state_source_map_verified"] += 1
            checks["original_training_donor_answer_recomputed"] += 1
        contract = contracts[donor.state_hash]
        facts = Facts(decode_state(donor.state), atlas)
        gold, text = answer(facts, op, q), question(op, q)
        require(gold == reference_answer(candidate, contract, checks),
                f"answer crosscheck failed: {op} / source line {donor.line} / {q}")
        require(canonical_sha256(donor.state) == donor.state_hash, "donor state was mutated")
        state_id = donor.provenance["state_id"]
        if state_id not in boards:
            boards[state_id] = contract_to_render_state(contract)
            validate_render(contract, boards[state_id], donor.state)
            checks["exact_render_state_crosschecks"] += 1
        row_id = f"{VERSION}/{family}/{op}/{op_indices[op]:03d}"
        op_indices[op] += 1
        source = {
            "row_id": donor.row["row_id"], "line": donor.line,
            "density": donor.provenance["density_bin"],
            "map_id": donor.provenance["board_map_sha256"],
            "source_kind": donor.provenance["source"]["kind"],
        }
        model_prompt = prompt(donor.state, text)
        query = {"operation": op, **q}
        metadata = {
            "review_only": True, "admitted_for_training": False,
            "class": "board_fluency", "family": family, "operation": op,
            "state_id": state_id, "question": text, "answer": gold,
            "target": {"state": donor.state, "query": query},
            "source": source, "provenance": donor.provenance,
            "state_sha256": donor.state_hash, "query_sha256": canonical_sha256(query),
            "donor_physical_line_sha256": donor.line_sha256,
            "donor_declarations": {key: donor.row[key] for key in
                                   ("schema", "split", "task_role", "task_type", "training_family")},
            "selection_stratum": candidate.stratum,
            "hypothetical_query_only": op in (
                "road_removal_connectivity", "settlement_upgrade_production", "robber_move_production"),
        }
        rows.append({
            "schema": SCHEMA, "id": row_id, "row_id": row_id,
            "messages": [{"role": "user", "content": model_prompt},
                         {"role": "assistant", "content": gold}],
            "metadata": metadata,
        })
        preview_rows.append({
            "row_id": row_id, "state_id": state_id, "family": family, "operation": op,
            "question": text, "answer": gold, "prompt": model_prompt, "source": source,
        })

    counts_family = dict(Counter(r["family"] for r in preview_rows))
    counts_operation = dict(Counter(r["operation"] for r in preview_rows))
    require(len(rows) == 200 and counts_family == dict.fromkeys(FAMILIES, 40),
            "hard family count assertion failed: expected exactly 40 in each of five families")
    require(counts_operation == dict.fromkeys(OPERATION_FAMILY, 10), "operation counts must all equal 10")
    require(len({r["row_id"] for r in rows}) == 200, "duplicate stable row IDs")
    require(len({(r["metadata"]["state_sha256"], compact(r["metadata"]["target"]["query"]))
                 for r in rows}) == 200, "duplicate canonical state/query padding")
    require(len({r["messages"][0]["content"] for r in rows}) == 200, "duplicate model prompts")
    for row, view in zip(rows, preview_rows, strict=True):
        m = row["metadata"]
        require([msg["role"] for msg in row["messages"]] == ["user", "assistant"], "message roles")
        require(all(set(msg) == {"role", "content"} and isinstance(msg["content"], str)
                    for msg in row["messages"]), "model input must be text-only")
        require(view["answer"] == m["answer"] == row["messages"][1]["content"], "preview gold mismatch")
        require(view["prompt"] == row["messages"][0]["content"]
                == prompt(m["target"]["state"], question(m["operation"], {
                    key: value for key, value in m["target"]["query"].items() if key != "operation"
                })), "prompt/target/preview mismatch")
        require(set(view) == {"row_id", "state_id", "family", "operation", "question",
                              "answer", "prompt", "source"}, "unexpected shared UI row schema")
        require(view["state_id"] in boards and 1 <= view["source"]["line"] <= MAX_SOURCE_ROWS,
                "preview/source join failed")
    checks.update({"validated_review_rows": len(rows), "validated_preview_rows": len(preview_rows),
                   "duplicate_state_query_pairs": 0, "duplicate_prompts": 0,
                   "family_count_assertions": 5, "operation_count_assertions": 20})
    states = Counter(c.donor.state_hash for c in selected)
    donor_uses = Counter(c.donor.row["row_id"] for c in selected)
    summary = {
        "row_count": len(rows), "counts_by_family": counts_family,
        "counts_by_operation": counts_operation, "unique_state_count": len(states),
        "unique_donor_row_count": len(donor_uses), "reused_donor_presentations": len(rows) - len(donor_uses),
        "max_presentations_per_state": max(states.values()),
        "unique_map_count": len({c.donor.provenance["board_map_sha256"] for c in selected}),
        "unique_trajectory_count": len({c.donor.provenance["source"]["trajectory_id"] for c in selected}),
        "counts_by_density": dict(Counter(r["source"]["density"] for r in preview_rows)),
        "counts_by_source_kind": dict(Counter(r["source"]["source_kind"] for r in preview_rows)),
        "seed": seed, "review_only": True, "admitted_for_training": False,
        "source_prefix_physical_lines": source_audit["physical_lines_read"],
    }
    require(len(states) == selection["maximum_distinct_states_for_required_cells"],
            "selection failed to maximize distinct source states")
    observations = {
        "prefix": richness(donors, atlas),
        "selected": richness([c.donor for c in selected], atlas),
        "selected_cells": {op: dict(Counter(c.stratum for c in selected if c.operation == op))
                           for op in OPERATION_FAMILY},
        "unsupported_required_cells": [],
        "observed_but_unselected_cells": [s for s in selection["support_by_cell"] if not s["selected"]],
        "scope_limits": [
            "Only the first 400 physical source rows were examined; richness is not a full-corpus claim.",
            "Source contracts certify material state, not an independently reconstructed game history.",
            "Road-removal examples query the removed edge's endpoints, contrasting bridges and actual alternate routes.",
            "Production ignores bank shortages; upgrades explicitly ignore cost and piece supply.",
            "Full settlement-rule conjunction/legal placement and Longest Road optimization remain transfer-only.",
            "Strategic policy is outside this deterministic board-fluency review and belongs to RL.",
            "Source states/maps/trajectories are correlated; row count is not an independent-sample claim.",
        ],
    }
    preview = {
        "schema": SCHEMA, "title": "Symbolic board fluency — review v1",
        "metadata": summary, "rows": preview_rows, "boards": boards,
    }
    payloads = {
        "review.jsonl": ("".join(compact(row) + "\n" for row in rows)).encode(),
        "preview.json": (json.dumps(preview, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    }
    dependencies = (
        Path(__file__), ROOT / "sft/symbolic_board_tasks.py", ROOT / "sft/spatial_tasks.py",
        ROOT / "evals/catan_board_bench/annotations.py", ROOT / "evals/catan_board_bench/tokens.py",
        ROOT / "data_pipeline/board_recognition/sources.py",
        ROOT / "data_pipeline/board_recognition/replay_dataset.py",
        ROOT / "data_pipeline/board_recognition/full_board_readout.py",
    )
    metadata = {
        "schema": SCHEMA, "title": preview["title"], **summary,
        "source_audit": source_audit, "selection": selection, "observed_richness": observations,
        "validation": {"status": "passed", **dict(checks)},
        "build_code": {repository_relative(path): file_sha256(path) for path in dependencies},
        "artifacts": {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                      for name, payload in payloads.items()},
        "source_verification_scope": {
            "all_prefix_row_training_declarations": True,
            "selected_original_contract_sha256": True,
            "selected_contract_equals_original_target_state": True,
            "selected_original_provenance_preserved": True,
            "selected_map_and_visible_fact_hashes": True,
            "full_source_file_sha256": False,
            "original_label_files_rehashed": False,
            "full_source_loader_invoked": False,
            "model_input_contains_images_or_render_state": False,
            "successor_states_persisted_as_source": False,
        },
    }
    payloads["metadata.json"] = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    return payloads


def publish(payloads: dict[str, bytes], *, rewrite: bool, verify_only: bool) -> None:
    require(set(payloads) == OWN_FILES, "unexpected output files")
    require(OUTPUT.parent.is_dir() and not OUTPUT.is_symlink(), "invalid output parent/directory")
    if verify_only:
        require(OUTPUT.is_dir(), "no existing batch to verify")
    elif OUTPUT.exists():
        require(rewrite, "output directory already exists; use --verify-only or explicit --rewrite-own-artifacts")
        require(OUTPUT.is_dir(), "output path is not a directory")
        require({p.name for p in OUTPUT.iterdir()} <= OWN_FILES, "refusing to overwrite unowned artifacts")
    else:
        OUTPUT.mkdir()  # Fail on races; no silent replacement of another build.
    for name, payload in payloads.items():
        path = OUTPUT / name
        require(not path.is_symlink(), f"refusing artifact symlink: {path}")
        expected_hash = hashlib.sha256(payload).hexdigest()
        if not verify_only:
            # Explicit rewrite touches only these three owned generated files.
            with path.open("wb" if rewrite else "xb") as handle:
                handle.write(payload)
        require(file_sha256(path) == expected_hash, f"artifact rebuild/hash mismatch: {name}")
    metadata = json.loads((OUTPUT / "metadata.json").read_text())
    preview = json.loads((OUTPUT / "preview.json").read_text())
    with (OUTPUT / "review.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    require(len(rows) == len(preview["rows"]) == metadata["row_count"] == 200,
            "written artifacts failed row-count round trip")
    require(len(preview["boards"]) == metadata["unique_state_count"], "written preview board count")
    print(json.dumps({
        "status": "verified_byte_identical_rebuild" if verify_only else "built_and_validated",
        "artifacts": {name: repository_relative(OUTPUT / name) for name in sorted(OWN_FILES)},
        "row_count": metadata["row_count"], "counts_by_family": metadata["counts_by_family"],
        "counts_by_operation": metadata["counts_by_operation"],
        "unique_state_count": metadata["unique_state_count"],
        "unique_donor_row_count": metadata["unique_donor_row_count"],
        "unique_map_count": metadata["unique_map_count"],
        "unique_trajectory_count": metadata["unique_trajectory_count"],
        "reused_donor_presentations": metadata["reused_donor_presentations"],
        "counts_by_density": metadata["counts_by_density"],
        "counts_by_source_kind": metadata["counts_by_source_kind"],
        "source_prefix_physical_lines": metadata["source_prefix_physical_lines"],
        "full_corpus_hash_verified": metadata["source_audit"]["full_corpus_hash_verified"],
        "validation": metadata["validation"],
    }, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-prefix", action="store_true",
                        help="Print bounded source richness without writing artifacts")
    parser.add_argument("--seed", type=int, default=SEED)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true",
                      help="Rebuild in memory and compare all three artifact hashes without writes")
    mode.add_argument("--rewrite-own-artifacts", action="store_true",
                      help="Explicitly rewrite only this builder's three artifacts (also repairs partial writes)")
    args = parser.parse_args()
    if not args.inspect_prefix and OUTPUT.exists() and not (args.verify_only or args.rewrite_own_artifacts):
        parser.error("output directory already exists; use --verify-only or explicit --rewrite-own-artifacts")
    donors, audit = load_prefix()
    if args.inspect_prefix:
        print(json.dumps({"source": audit, "richness": richness(donors, atlas_geometry()),
                          "example_original_provenance": donors[0].provenance},
                         indent=2, sort_keys=True))
        return 0
    payloads = build(donors, audit, args.seed)
    publish(payloads, rewrite=args.rewrite_own_artifacts, verify_only=args.verify_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
