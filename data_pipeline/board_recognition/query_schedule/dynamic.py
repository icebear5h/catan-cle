"""Capacity-constrained scheduling of occupied and empty board slots."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence, TypeAlias

import networkx as nx

from data_pipeline.board_recognition import query_schedule as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_str

FlowNode: TypeAlias = "tuple[str, str]"
FlowGraph: TypeAlias = "nx.DiGraph[FlowNode, dict[str, object], dict[str, object]]"

__all__ = ["_dynamic_class_quotas", "_schedule_dynamic_head", "nx"]


def _dynamic_class_quotas(
    capacities: Counter[str],
    *,
    total: int,
    split: str,
    head: str,
    seed: int,
) -> dict[str, int]:
    classes = sorted(capacities)
    if not classes or total < len(classes):
        raise api.QueryScheduleError(f"{split} {head} cannot cover its positive classes")
    if split == "train":
        expected = [name for name in api.RECOGNITION_CLASS_VOCABULARIES[head] if name != "EMPTY"]
        if set(classes) != set(expected):
            missing = sorted(set(expected) - set(classes))
            raise api.QueryScheduleError(f"train {head} lacks source classes: {missing}")
        base, remainder = divmod(total, len(classes))
        ranked = sorted(classes, key=lambda name: api.stable_rank(seed, split, head, name))
        quotas = {name: base + (ranked.index(name) < remainder) for name in classes}
        under_capacity = {
            name: (quotas[name], capacities[name])
            for name in classes
            if quotas[name] > capacities[name]
        }
        if under_capacity:
            raise api.QueryScheduleError(f"train {head} cannot meet balanced quotas: {under_capacity}")
        return quotas

    return {name: 1 for name in classes}


def _schedule_dynamic_head(
    ordered_states: Sequence[JsonDict],
    *,
    labels_by_state: dict[str, JsonDict],
    entity_type: str,
    head: str,
    queries_per_state: int,
    split: str,
    seed: int,
) -> dict[str, list[JsonDict]]:
    positive_by_state: dict[str, list[JsonDict]] = {}
    empty_by_state: dict[str, list[JsonDict]] = {}
    capacities: Counter[str] = Counter()
    for state in ordered_states:
        state_id = as_str(state["sample_id"])
        candidates = api.state_query_candidates(labels_by_state[state_id], entity_type, head)
        positive = [row for row in candidates if api._is_positive(row)]
        empty = [row for row in candidates if not api._is_positive(row)]
        positive_by_state[state_id] = positive
        empty_by_state[state_id] = empty
        state_class_counts = Counter(as_str(row["class_name"]) for row in positive)
        for class_name, count in state_class_counts.items():
            capacities[class_name] += min(queries_per_state, count)
    total_queries = len(ordered_states) * queries_per_state
    positive_target = total_queries // 2
    quotas = api._dynamic_class_quotas(
        capacities, total=positive_target, split=split, head=head, seed=seed,
    )

    graph: FlowGraph = nx.DiGraph()
    source = ("source", head)
    sink = ("sink", head)
    graph.add_node(source, demand=-positive_target)
    if split == "train":
        graph.add_node(sink, demand=positive_target)
    else:
        graph.add_node(sink, demand=positive_target - len(quotas))
    for state in ordered_states:
        state_id = as_str(state["sample_id"])
        state_node = ("state", state_id)
        state_capacity = min(queries_per_state, len(positive_by_state[state_id]))
        graph.add_node(state_node, demand=0)
        graph.add_edge(source, state_node, capacity=state_capacity, weight=0)
        by_class = Counter(as_str(row["class_name"]) for row in positive_by_state[state_id])
        for class_name, capacity in sorted(by_class.items()):
            class_node = ("class", class_name)
            graph.add_node(class_node, demand=0 if split == "train" else 1)
            graph.add_edge(
                state_node,
                class_node,
                capacity=min(queries_per_state, capacity),
                weight=api.stable_rank(seed, split, head, state_id, class_name) % 10_000,
            )
    for class_name, quota in sorted(quotas.items()):
        capacity = quota if split == "train" else capacities[class_name] - 1
        graph.add_edge(("class", class_name), sink, capacity=capacity, weight=0)
    try:
        flow = nx.min_cost_flow(graph)
    except (nx.NetworkXUnfeasible, nx.NetworkXError) as exc:
        raise api.QueryScheduleError(
            f"{split} {head} cannot satisfy deterministic class quotas"
        ) from exc

    scheduled: dict[str, list[JsonDict]] = defaultdict(list)
    slot_counts: Counter[str] = Counter()
    for state in ordered_states:
        state_id = as_str(state["sample_id"])
        state_node = ("state", state_id)
        used_slots: set[str] = set()
        for class_name in sorted(quotas):
            amount = flow[state_node].get(("class", class_name), 0)
            if not amount:
                continue
            candidates = sorted(
                (
                    row
                    for row in positive_by_state[state_id]
                    if row["class_name"] == class_name and row["slot"] not in used_slots
                ),
                key=lambda row: (
                    slot_counts[as_str(row["slot"])],
                    api.stable_rank(seed, split, head, state_id, class_name, as_str(row["slot"])),
                ),
            )
            if len(candidates) < amount:
                raise api.QueryScheduleError(f"{state_id} lacks assigned {class_name} slots")
            for row in candidates[:amount]:
                scheduled[state_id].append(row)
                used_slots.add(as_str(row["slot"]))
                slot_counts[as_str(row["slot"])] += 1
        empty_needed = queries_per_state - len(scheduled[state_id])
        empties = sorted(
            (row for row in empty_by_state[state_id] if row["slot"] not in used_slots),
            key=lambda row: (
                slot_counts[as_str(row["slot"])],
                api.stable_rank(seed, split, head, state_id, "empty", as_str(row["slot"])),
            ),
        )
        if len(empties) < empty_needed:
            raise api.QueryScheduleError(f"{state_id} lacks {empty_needed} empty {head} slots")
        for row in empties[:empty_needed]:
            scheduled[state_id].append(row)
            slot_counts[as_str(row["slot"])] += 1
    return scheduled
