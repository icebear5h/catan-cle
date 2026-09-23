from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import TypeAlias, TypedDict, cast

import networkx as networkx

from data_pipeline.board_recognition.sources import canonical_sha256
from sft.board.symbolic_board_tasks import (
    atlas_geometry,
    decode_state,
    strict_json,
    symbolic_answer,
)
from sft.board.symbolic_board_tasks._types import Atlas, DecodedState, StatePayload
from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_float, as_str

from ._components import _selectors, selector_nodes
from ._sampling import _ordered_sources, _row
from ._sources import _check
from ._types import QueryDict, SelectorDict, SourceRecord

nx = networkx

TokenGraph: TypeAlias = "networkx.Graph[str, dict[str, object], dict[str, object]]"


def _endpoints(ends: tuple[str, ...]) -> tuple[str, str]:
    first, second = ends
    return first, second


class Candidate(TypedDict):
    """One task-relevant unique projection with the sources that collapse onto it."""

    task: str
    query: QueryDict
    record: SourceRecord
    mode: str
    polarity: str
    projection: str
    equivalent_sources: list[str]


def _lengths(state: StatePayload) -> dict[str, float]:
    """Engine trail length per color, decoded from the oracle's JSON answer."""
    answer = symbolic_answer("symbolic_longest_lengths",
                             cast("JsonDict", {"state": state, "query": {}}))
    return {key: as_float(value) for key, value in as_dict(strict_json(answer)).items()}


def transfer_projection(task: str, state: StatePayload, query: QueryDict, *,
                        atlas: Atlas | None = None,
                        data: DecodedState | None = None) -> str:
    """Hash relevant inputs, never the answer. Ignore robber, irrelevant terrain and pieces.

    Setup ignores queried color, ownership and roads. Normal keeps queried incident
    roads. Near selector kinds remain distinct reasoning modes; within each mode,
    equivalent candidate-node filters and local occupancy deduplicate. Road tasks
    retain participant identities, owned edges and effective transit blockers only.
    """
    atlas = atlas if atlas is not None else atlas_geometry()
    data = data if data is not None else decode_state(state)
    projection: JsonLikeDict
    if task == "symbolic_settlement_locations":
        near = cast("SelectorDict | None", query["near"])
        candidates = selector_nodes(near, data, atlas)
        relevant_nodes = candidates | {n for v in candidates for n in atlas["graph"][v]}
        projection = {
            "task": task, "phase": query["phase"],
            "near_kind": near["kind"] if near else "all",
            "candidates": sorted(candidates),
            "occupied": sorted(relevant_nodes & data["buildings"].keys()),
        }
        if query["phase"] == "normal":
            projection["owned_incident_edges"] = sorted(
                e for e, c in data["roads"].items() if c == query["color"]
                and candidates.intersection(atlas["edges"][e]))
    else:
        blockers: dict[str, list[str]] = {}
        for color in state["colors"]:
            degree = Counter(n for e, c in data["roads"].items() if c == color for n in atlas["edges"][e])
            blockers[color] = sorted(n for n, (owner, _) in data["buildings"].items()
                                     if owner != color and degree[n] >= 2)
        projection = {"task": task, "colors": sorted(state["colors"]),
                      "roads": data["roads"], "blockers": blockers}
    return canonical_sha256(projection)


def graph_case_coverage(records: list[SourceRecord]) -> JsonLikeDict:
    """Measured graph cases on real, unmodified sources, including absent hard cases."""
    atlas = atlas_geometry()
    cases: list[dict[str, str | bool]] = []
    for record in records:
        state, p = record["state"], record["provenance"]
        data = decode_state(state)
        lengths = _lengths(state)
        maximum = max(lengths.values())
        flags = dict(cycle=False, effective_blocker=False, branch=False,
                     tied_max_ge5=maximum >= 5 and sum(v == maximum for v in lengths.values()) > 1,
                     nonzero=maximum > 0, award_eligible=maximum >= 5)
        for color in state["colors"]:
            graph: TokenGraph = nx.Graph()
            graph.add_edges_from(_endpoints(atlas["edges"][e])
                                 for e, c in data["roads"].items() if c == color)
            flags["cycle"] |= bool(nx.cycle_basis(graph))
            flags["branch"] |= any(d >= 3 for _, d in graph.degree)
            flags["effective_blocker"] |= any(
                owner != color and n in graph and graph.degree[n] >= 2
                for n, (owner, _) in data["buildings"].items())
        cases.append({"state_id": as_str(p["state_id"]),
                      "board_fact_sha256": as_str(p["board_fact_sha256"]), **flags})
    counts = {key: sum(bool(c[key]) for c in cases) for key in
              ("cycle", "effective_blocker", "branch", "tied_max_ge5", "nonzero", "award_eligible")}
    missing = [key for key in ("cycle", "effective_blocker", "tied_max_ge5") if not counts[key]]
    return {"states": len(records), "counts": counts, "cases": cases,
            "missing_coverage": missing,
            "status": "transfer_pilot_limited" if missing else "observed_cases_only_not_exhaustive",
            "definitions": {"cycle": "cycle in one player's owned undirected graph",
                            "effective_blocker": "opponent building with at least two incident roads of a player; blocks interior transit",
                            "tied_max_ge5": "at least two colors tie for maximum engine trail length >=5"}}


def _diverse_candidates(pool: list[Candidate], count: int,
                        rng: random.Random) -> list[Candidate]:
    candidates = list(pool)
    rng.shuffle(candidates)
    uses: Counter[str] = Counter()
    density: Counter[str] = Counter()
    selected: list[Candidate] = []
    while candidates and len(selected) < count:
        index = min(range(len(candidates)), key=lambda i: (
            uses[as_str(candidates[i]["record"]["provenance"]["state_id"])],
            density[as_str(candidates[i]["record"]["provenance"]["density_bin"])], i))
        candidate = candidates.pop(index)
        selected.append(candidate)
        p = candidate["record"]["provenance"]
        uses[as_str(p["state_id"])] += 1
        density[as_str(p["density_bin"])] += 1
    return selected


def transfer_weights(rows: list[JsonDict]) -> list[dict[str, float]]:
    """Per-family mode/polarity macro weights with equal state weight within each cell."""
    states: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    labels: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    modes: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        m = as_dict(row["metadata"])
        family, mode, label = (as_str(row["task_type"]), as_str(m["evaluation_mode"]),
                               as_str(m["polarity"]))
        states[family, mode, label][as_str(as_dict(m["provenance"])["state_id"])] += 1
        labels[family, mode].add(label)
        modes[family].add(mode)
    result: list[dict[str, float]] = []
    for row in rows:
        m = as_dict(row["metadata"])
        family, mode, label = (as_str(row["task_type"]), as_str(m["evaluation_mode"]),
                               as_str(m["polarity"]))
        counts = states[family, mode, label]
        state_weight = 1 / (len(counts)
                            * counts[as_str(as_dict(m["provenance"])["state_id"])])
        result.append({"state_weight": state_weight,
                       "macro_weight": state_weight / (len(labels[family, mode]) * len(modes[family]))})
    return result


def transfer_rows(records: list[SourceRecord], split: str,
                  seed: int) -> tuple[list[JsonDict], JsonLikeDict]:
    """Enumerate real candidates; retain all unique normal positives and positive road cases.

    Setup is a bounded matched control (<=16 positives per selector kind), rather
    than an enormous color/terrain-expanded panel. Negative subsets never exceed
    positive support; a missing positive cell stays explicitly empty, never repaired.
    """
    rng, atlas = random.Random(seed), atlas_geometry()
    unique: dict[str, Candidate] = {}
    population: Counter[tuple[str, str, str]] = Counter()
    skipped: list[dict[str, str]] = []

    def admit(task: str, query: QueryDict, record: SourceRecord, data: DecodedState,
              mode: str, positive: bool) -> None:
        label = "positive" if positive else "negative"
        population[task, mode, label] += 1
        projection = transfer_projection(task, record["state"], query, atlas=atlas, data=data)
        if projection not in unique:
            unique[projection] = Candidate(task=task, query=query, record=record, mode=mode,
                                           polarity=label, projection=projection,
                                           equivalent_sources=[])
        candidate = unique[projection]
        _check(candidate["polarity"] == label, "task-relevant projection collapsed different labels")
        sid = as_str(record["provenance"]["state_id"])
        if sid not in candidate["equivalent_sources"]:
            candidate["equivalent_sources"].append(sid)

    selectors: list[SelectorDict | None] = [None]
    selectors += [s for kind in ("tile", "resource", "port") for s in _selectors(kind, atlas)]
    for record in _ordered_sources(records, rng):
        state = record["state"]
        data = decode_state(state)
        for color in state["colors"]:
            for phase in ("setup", "normal"):
                base = symbolic_answer("symbolic_settlement_locations", cast("JsonDict", {
                    "state": state, "query": dict(color=color, phase=phase, near=None)}))
                legal = set(base.split()) - {"NONE"}
                for near in selectors:
                    mode = phase + "/" + (near["kind"] if near else "all")
                    query: QueryDict = dict(color=color, phase=phase, near=near)
                    admit("symbolic_settlement_locations", query,
                          record, data, mode, bool(legal & selector_nodes(near, data, atlas)))
        lengths = _lengths(state)
        maximum = max(lengths.values())
        for task in ("symbolic_longest_lengths", "symbolic_longest_leaders", "symbolic_longest_award"):
            if task == "symbolic_longest_award" and maximum >= 5 and sum(v == maximum for v in lengths.values()) > 1:
                skipped.append({"state_id": as_str(record["provenance"]["state_id"]),
                                "task_type": task,
                                "reason": "unknown_incumbent_tied_maximum"})
                continue
            admit(task, {}, record, data, "all", maximum >= (5 if task == "symbolic_longest_award" else 1))
    buckets: defaultdict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    for candidate in unique.values():
        buckets[candidate["task"], candidate["mode"], candidate["polarity"]].append(candidate)
    selected: list[Candidate] = []
    group_report: list[JsonLikeDict] = []
    groups = sorted({key[:2] for key in population})
    for task, mode in groups:
        pos, neg = buckets[task, mode, "positive"], buckets[task, mode, "negative"]
        budget = len(pos)
        if mode.startswith("setup/"):
            normal_support = len(buckets[task, mode.replace("setup/", "normal/"), "positive"])
            budget = min(budget, normal_support, 16)
        chosen_pos = _diverse_candidates(pos, budget, rng)
        chosen_neg = _diverse_candidates(neg, len(chosen_pos), rng)
        selected.extend(chosen_pos + chosen_neg)
        group_report.append({"task_type": task, "mode": mode,
                             "positive": {"population": population[task, mode, "positive"],
                                          "unique": len(pos), "selected": len(chosen_pos)},
                             "negative": {"population": population[task, mode, "negative"],
                                          "unique": len(neg), "selected": len(chosen_neg)}})
    rng.shuffle(selected)
    result: list[JsonDict] = []
    for candidate in selected:
        row = _row(candidate["task"], candidate["query"], candidate["record"], split, len(result))
        as_dict(row["metadata"]).update(cast("JsonDict", dict(
            evaluation_mode=candidate["mode"], polarity=candidate["polarity"],
            task_projection_sha256=candidate["projection"],
            equivalent_source_state_ids=sorted(candidate["equivalent_sources"]))))
        result.append(row)
    for row, weights in zip(result, transfer_weights(result), strict=True):
        as_dict(row["metadata"]).update(weights)
    selected_records = {
        as_str(as_dict(as_dict(r["metadata"])["provenance"])["state_id"]): SourceRecord(
            state=cast("StatePayload", as_dict(as_dict(r["metadata"])["target"])["state"]),
            provenance=as_dict(as_dict(r["metadata"])["provenance"]))
        for r in result}
    report: JsonLikeDict = {
        "status": "transfer_pilot_limited", "population_rows": sum(population.values()),
        "unique_relevant_queries": len(unique), "selected_rows": len(result), "groups": group_report,
        "empty_groups": [
            {"task_type": g["task_type"], "mode": g["mode"], "polarity": label,
             "reason": ("no_population_support"
                        if not as_dict(g[label])["unique"]
                        else "matched_selection_has_no_positive_support")}
            for g in group_report for label in ("positive", "negative")
            if not as_dict(g[label])["selected"]],
        "skipped_ambiguous_awards": skipped,
        "population_graph_cases": graph_case_coverage(records),
        "selected_graph_cases": graph_case_coverage(list(selected_records.values())),
        "selection_policy": "All task-relevant unique normal-placement positives and positive road cases retained; equal-or-smaller negative subsets. Setup controls matched to normal positive support with <=16 positives per near kind. No source split changes or fabricated states.",
        "aggregation": {"pooled_micro_headline": False, "report_families_and_modes_separately": True,
                        "weights": "macro_weight sums to one within each task family, balancing supported modes/polarities and source states; state_weight sums to one within each mode/polarity cell",
                        "inference_unit": "whole source state/game; rows are correlated query projections",
                        "missing_groups": "report as unsupported, not zero accuracy or full coverage"},
    }
    return result, report
