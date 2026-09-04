"""Deterministic split-level query scheduling for replay_v1 recognition."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import networkx as nx

from data_pipeline.board_recognition.replay_dataset import read_jsonl
from evals.catan_board_bench.tokens import (
    RECOGNITION_CLASS_VOCABULARIES,
    recognition_answer_token,
    recognition_query_token,
)


JsonDict = dict[str, Any]
ENTITY_TYPES = ("tile", "node", "edge", "port")
COLLECTION_BY_ENTITY = {
    "tile": "tiles",
    "node": "nodes",
    "edge": "edges",
    "port": "ports",
}
HEADS_BY_ENTITY = {
    "tile": ("tile.resource", "tile.number", "tile.robber"),
    "node": ("node.occupancy",),
    "edge": ("edge.owner",),
    "port": ("port.port_type",),
}
ATTRIBUTE_BY_HEAD = {
    "tile.resource": "resource",
    "tile.number": "number",
    "tile.robber": "robber",
    "node.occupancy": "occupancy",
    "edge.owner": "owner",
    "port.port_type": "port_type",
}
ENTITY_TYPE_IDS = {name: index for index, name in enumerate(ENTITY_TYPES)}
ATTRIBUTE_IDS = {head: index for index, head in enumerate(ATTRIBUTE_BY_HEAD)}


class QueryScheduleError(RuntimeError):
    """Raised when a split cannot satisfy deterministic query gates."""


def stable_rank(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def normalize_class_name(head: str, value: Any) -> str:
    if head == "tile.number":
        return "NONE" if value is None else str(value)
    if head == "tile.robber":
        return "PRESENT" if value else "ABSENT"
    return str(value)


def state_query_candidates(labels: JsonDict, entity_type: str, head: str) -> list[JsonDict]:
    collection = COLLECTION_BY_ENTITY[entity_type]
    attribute = ATTRIBUTE_BY_HEAD[head]
    vocabulary = RECOGNITION_CLASS_VOCABULARIES[head]
    candidates = []
    for slot_index, row in enumerate(labels["entities"][collection]):
        class_name = normalize_class_name(head, row[attribute])
        if class_name not in vocabulary:
            raise QueryScheduleError(
                f"unknown class {class_name!r} for {head} in {labels['sample_id']}"
            )
        candidates.append(
            {
                "entity_type": entity_type,
                "entity_type_id": ENTITY_TYPE_IDS[entity_type],
                "attribute": attribute,
                "attribute_id": ATTRIBUTE_IDS[head],
                "head": head,
                "slot": row["id"],
                "slot_index": slot_index,
                "class_name": class_name,
                "class_index": vocabulary.index(class_name),
                "class_vocabulary": list(vocabulary),
                "query_token": recognition_query_token(head),
                "answer_token": recognition_answer_token(head, class_name),
            }
        )
    return candidates


def _is_positive(candidate: JsonDict) -> bool:
    return candidate["class_name"] != "EMPTY"


def _choose_tile_head(
    state_id: str,
    round_index: int,
    head_counts: Counter[str],
    selected_keys: set[tuple[str, str]],
    labels: JsonDict,
) -> str:
    available = []
    for head in HEADS_BY_ENTITY["tile"]:
        candidates = state_query_candidates(labels, "tile", head)
        if any((head, row["slot"]) not in selected_keys for row in candidates):
            available.append(head)
    if not available:
        raise QueryScheduleError(f"state {state_id} exhausted tile query candidates")
    return min(
        available,
        key=lambda head: (
            head_counts[head],
            stable_rank(state_id, round_index, head),
        ),
    )


def _select_candidate(
    candidates: Sequence[JsonDict],
    *,
    state_id: str,
    round_index: int,
    selected_keys: set[tuple[str, str]],
    class_counts: Counter[tuple[str, str]],
    slot_counts: Counter[tuple[str, str]],
    polarity_counts: Counter[tuple[str, str]],
) -> JsonDict:
    remaining = [row for row in candidates if (row["head"], row["slot"]) not in selected_keys]
    if not remaining:
        raise QueryScheduleError(
            f"state {state_id} exhausted candidates for {candidates[0]['head']}"
        )
    head = remaining[0]["head"]
    if head in {"node.occupancy", "edge.owner"}:
        available_polarities = {"positive" if _is_positive(row) else "empty" for row in remaining}
        desired_polarity = min(
            available_polarities,
            key=lambda polarity: (
                polarity_counts[(head, polarity)],
                stable_rank(state_id, round_index, head, polarity),
            ),
        )
        preferred = [
            row
            for row in remaining
            if ("positive" if _is_positive(row) else "empty") == desired_polarity
        ]
    else:
        preferred = remaining
    return min(
        preferred,
        key=lambda row: (
            class_counts[(head, row["class_name"])],
            slot_counts[(head, row["slot"])],
            stable_rank(state_id, round_index, head, row["slot"], row["class_name"]),
        ),
    )


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
        raise QueryScheduleError(f"{split} {head} cannot cover its positive classes")
    if split == "train":
        expected = [name for name in RECOGNITION_CLASS_VOCABULARIES[head] if name != "EMPTY"]
        if set(classes) != set(expected):
            missing = sorted(set(expected) - set(classes))
            raise QueryScheduleError(f"train {head} lacks source classes: {missing}")
        base, remainder = divmod(total, len(classes))
        ranked = sorted(classes, key=lambda name: stable_rank(seed, split, head, name))
        quotas = {name: base + (ranked.index(name) < remainder) for name in classes}
        under_capacity = {
            name: (quotas[name], capacities[name])
            for name in classes
            if quotas[name] > capacities[name]
        }
        if under_capacity:
            raise QueryScheduleError(f"train {head} cannot meet balanced quotas: {under_capacity}")
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
        state_id = state["sample_id"]
        candidates = state_query_candidates(labels_by_state[state_id], entity_type, head)
        positive = [row for row in candidates if _is_positive(row)]
        empty = [row for row in candidates if not _is_positive(row)]
        positive_by_state[state_id] = positive
        empty_by_state[state_id] = empty
        state_class_counts = Counter(row["class_name"] for row in positive)
        for class_name, count in state_class_counts.items():
            capacities[class_name] += min(queries_per_state, count)
    total_queries = len(ordered_states) * queries_per_state
    positive_target = total_queries // 2
    quotas = _dynamic_class_quotas(
        capacities,
        total=positive_target,
        split=split,
        head=head,
        seed=seed,
    )

    graph = nx.DiGraph()
    source = ("source", head)
    sink = ("sink", head)
    graph.add_node(source, demand=-positive_target)
    if split == "train":
        graph.add_node(sink, demand=positive_target)
    else:
        graph.add_node(sink, demand=positive_target - len(quotas))
    for state in ordered_states:
        state_id = state["sample_id"]
        state_node = ("state", state_id)
        state_capacity = min(queries_per_state, len(positive_by_state[state_id]))
        graph.add_node(state_node, demand=0)
        graph.add_edge(source, state_node, capacity=state_capacity, weight=0)
        by_class = Counter(row["class_name"] for row in positive_by_state[state_id])
        for class_name, capacity in sorted(by_class.items()):
            class_node = ("class", class_name)
            graph.add_node(class_node, demand=0 if split == "train" else 1)
            graph.add_edge(
                state_node,
                class_node,
                capacity=min(queries_per_state, capacity),
                weight=stable_rank(seed, split, head, state_id, class_name) % 10_000,
            )
    for class_name, quota in sorted(quotas.items()):
        capacity = quota if split == "train" else capacities[class_name] - 1
        graph.add_edge(("class", class_name), sink, capacity=capacity, weight=0)
    try:
        flow = nx.min_cost_flow(graph)
    except (nx.NetworkXUnfeasible, nx.NetworkXError) as exc:
        raise QueryScheduleError(
            f"{split} {head} cannot satisfy deterministic class quotas"
        ) from exc

    scheduled: dict[str, list[JsonDict]] = defaultdict(list)
    slot_counts: Counter[str] = Counter()
    for state in ordered_states:
        state_id = state["sample_id"]
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
                    slot_counts[row["slot"]],
                    stable_rank(seed, split, head, state_id, class_name, row["slot"]),
                ),
            )
            if len(candidates) < amount:
                raise QueryScheduleError(f"{state_id} lacks assigned {class_name} slots")
            for row in candidates[:amount]:
                scheduled[state_id].append(row)
                used_slots.add(row["slot"])
                slot_counts[row["slot"]] += 1
        empty_needed = queries_per_state - len(scheduled[state_id])
        empties = sorted(
            (row for row in empty_by_state[state_id] if row["slot"] not in used_slots),
            key=lambda row: (
                slot_counts[row["slot"]],
                stable_rank(seed, split, head, state_id, "empty", row["slot"]),
            ),
        )
        if len(empties) < empty_needed:
            raise QueryScheduleError(f"{state_id} lacks {empty_needed} empty {head} slots")
        for row in empties[:empty_needed]:
            scheduled[state_id].append(row)
            slot_counts[row["slot"]] += 1
    return scheduled


def build_split_query_plan(
    state_rows: Sequence[JsonDict],
    *,
    dataset_dir: Path,
    split: str,
    seed: int,
    queries_per_state: int = 8,
    epoch_index: int = 0,
) -> list[JsonDict]:
    if queries_per_state <= 0 or queries_per_state % len(ENTITY_TYPES):
        raise ValueError("queries_per_state must be positive and divisible by four")
    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative")
    rounds_per_entity = queries_per_state // len(ENTITY_TYPES)
    schedule_seed = stable_rank(seed, "epoch", epoch_index)
    if not state_rows:
        raise QueryScheduleError(f"no states available for split {split}")
    ordered_states = sorted(
        state_rows,
        key=lambda row: (
            stable_rank(schedule_seed, split, row["sample_id"]),
            row["sample_id"],
        ),
    )
    labels_by_state = {
        row["sample_id"]: json.loads((dataset_dir / row["label_path"]).read_text())
        for row in ordered_states
    }
    plans = {
        row["sample_id"]: {
            "schema": "catan_board_recognition_query_plan/v1",
            "state_id": row["sample_id"],
            "split": split,
            "epoch": epoch_index,
            "queries": [],
        }
        for row in ordered_states
    }
    selected_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
    head_counts: Counter[str] = Counter()
    class_counts: Counter[tuple[str, str]] = Counter()
    slot_counts: Counter[tuple[str, str]] = Counter()
    polarity_counts: Counter[tuple[str, str]] = Counter()

    for entity_type, head in (
        ("node", "node.occupancy"),
        ("edge", "edge.owner"),
    ):
        scheduled = _schedule_dynamic_head(
            ordered_states,
            labels_by_state=labels_by_state,
            entity_type=entity_type,
            head=head,
            queries_per_state=rounds_per_entity,
            split=split,
            seed=schedule_seed,
        )
        for state in ordered_states:
            state_id = state["sample_id"]
            for selected in scheduled[state_id]:
                selected_keys[state_id].add((head, selected["slot"]))
                head_counts[head] += 1
                class_counts[(head, selected["class_name"])] += 1
                slot_counts[(head, selected["slot"])] += 1
                polarity = "positive" if _is_positive(selected) else "empty"
                polarity_counts[(head, polarity)] += 1
                plans[state_id]["queries"].append(selected)

    for round_index in range(rounds_per_entity):
        for entity_type in ("tile", "port"):
            for state in ordered_states:
                state_id = state["sample_id"]
                labels = labels_by_state[state_id]
                if entity_type == "tile":
                    head = _choose_tile_head(
                        state_id,
                        round_index,
                        head_counts,
                        selected_keys[state_id],
                        labels,
                    )
                else:
                    head = HEADS_BY_ENTITY[entity_type][0]
                candidates = state_query_candidates(labels, entity_type, head)
                selected = _select_candidate(
                    candidates,
                    state_id=state_id,
                    round_index=round_index,
                    selected_keys=selected_keys[state_id],
                    class_counts=class_counts,
                    slot_counts=slot_counts,
                    polarity_counts=polarity_counts,
                )
                selected_keys[state_id].add((head, selected["slot"]))
                head_counts[head] += 1
                class_counts[(head, selected["class_name"])] += 1
                slot_counts[(head, selected["slot"])] += 1
                if head in {"node.occupancy", "edge.owner"}:
                    polarity = "positive" if _is_positive(selected) else "empty"
                    polarity_counts[(head, polarity)] += 1
                plans[state_id]["queries"].append(selected)

    output = []
    rows_by_id = {row["sample_id"]: row for row in state_rows}
    for state_id in sorted(plans):
        plan = plans[state_id]
        plan["queries"].sort(
            key=lambda row: (
                ENTITY_TYPES.index(row["entity_type"]),
                row["attribute_id"],
                row["slot"],
            )
        )
        for query_index, query in enumerate(plan["queries"]):
            query["query_id"] = f"{state_id}_e{epoch_index:03d}_q{query_index:03d}"
        plan["source"] = {
            "image_path": rows_by_id[state_id]["image_path"],
            "label_path": rows_by_id[state_id]["label_path"],
            "source_kind": rows_by_id[state_id]["source"]["kind"],
            "game_id": rows_by_id[state_id]["source"].get("game_id"),
            "trajectory_id": rows_by_id[state_id]["source"]["trajectory_id"],
            "density_bin": rows_by_id[state_id]["density_bin"],
        }
        output.append(plan)
    validate_query_plan(
        output,
        expected_states=len(state_rows),
        split=split,
        queries_per_state=queries_per_state,
        epoch_index=epoch_index,
    )
    return output


def query_plan_metrics(plans: Sequence[JsonDict]) -> JsonDict:
    entity_counts: Counter[str] = Counter()
    head_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    slot_counts: dict[str, Counter[str]] = defaultdict(Counter)
    polarity_counts: Counter[str] = Counter()
    duplicates = 0
    for plan in plans:
        seen = set()
        for query in plan["queries"]:
            key = (query["head"], query["slot"])
            duplicates += key in seen
            seen.add(key)
            entity_counts[query["entity_type"]] += 1
            head_counts[query["head"]] += 1
            class_counts[f"{query['head']}.{query['class_name']}"] += 1
            slot_counts[query["head"]][query["slot"]] += 1
            if query["head"] in {"node.occupancy", "edge.owner"}:
                polarity = "positive" if query["class_name"] != "EMPTY" else "empty"
                polarity_counts[f"{query['head']}.{polarity}"] += 1
    return {
        "states": len(plans),
        "rows": sum(entity_counts.values()),
        "duplicates": duplicates,
        "entity_counts": dict(sorted(entity_counts.items())),
        "head_counts": dict(sorted(head_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "slot_counts": {
            head: dict(sorted(counts.items())) for head, counts in sorted(slot_counts.items())
        },
        "polarity_counts": dict(sorted(polarity_counts.items())),
    }


def validate_query_plan(
    plans: Sequence[JsonDict],
    *,
    expected_states: int,
    split: str,
    queries_per_state: int = 8,
    epoch_index: int = 0,
) -> JsonDict:
    if len(plans) != expected_states:
        raise QueryScheduleError(
            f"query plan has {len(plans)}/{expected_states} states for {split}"
        )
    state_ids = [plan["state_id"] for plan in plans]
    if len(state_ids) != len(set(state_ids)):
        raise QueryScheduleError(f"query plan duplicated state IDs in {split}")
    expected_per_entity = queries_per_state // len(ENTITY_TYPES)
    for plan in plans:
        if plan.get("epoch") != epoch_index:
            raise QueryScheduleError(f"state {plan['state_id']} has the wrong epoch")
        queries = plan["queries"]
        if len(queries) != queries_per_state:
            raise QueryScheduleError(f"state {plan['state_id']} has {len(queries)} queries")
        entity_counts = Counter(query["entity_type"] for query in queries)
        if entity_counts != Counter({entity: expected_per_entity for entity in ENTITY_TYPES}):
            raise QueryScheduleError(
                f"state {plan['state_id']} entity balance is {dict(entity_counts)}"
            )
        keys = [(query["head"], query["slot"]) for query in queries]
        if len(keys) != len(set(keys)):
            raise QueryScheduleError(f"state {plan['state_id']} repeats a head/slot query")
        for query in queries:
            if query["answer_token"] != recognition_answer_token(
                query["head"], query["class_name"]
            ):
                raise QueryScheduleError(f"state {plan['state_id']} answer token mismatch")
            if query["query_token"] != recognition_query_token(query["head"]):
                raise QueryScheduleError(f"state {plan['state_id']} query token mismatch")
    metrics = query_plan_metrics(plans)
    if metrics["duplicates"]:
        raise QueryScheduleError(f"query plan contains {metrics['duplicates']} duplicates")
    return metrics


def validate_split_balance(plans: Sequence[JsonDict], *, split: str) -> JsonDict:
    metrics = query_plan_metrics(plans)
    expected_slots = {
        "tile.resource": 19,
        "tile.number": 19,
        "tile.robber": 19,
        "node.occupancy": 54,
        "edge.owner": 72,
        "port.port_type": 9,
    }
    missing_slots = {
        head: expected - len(metrics["slot_counts"].get(head, {}))
        for head, expected in expected_slots.items()
        if len(metrics["slot_counts"].get(head, {})) != expected
    }
    if missing_slots:
        raise QueryScheduleError(f"{split} query plan lacks slot coverage: {missing_slots}")

    required_heads = set(RECOGNITION_CLASS_VOCABULARIES)
    if split in {"validation", "test"}:
        required_heads = {
            "tile.resource",
            "tile.number",
            "tile.robber",
            "port.port_type",
        }
    missing_classes = []
    for head in sorted(required_heads):
        for class_name in RECOGNITION_CLASS_VOCABULARIES[head]:
            key = f"{head}.{class_name}"
            if metrics["class_counts"].get(key, 0) == 0:
                missing_classes.append(key)
    if missing_classes:
        raise QueryScheduleError(f"{split} query plan lacks required classes: {missing_classes}")

    for head in ("node.occupancy", "edge.owner"):
        positive = metrics["polarity_counts"].get(f"{head}.positive", 0)
        empty = metrics["polarity_counts"].get(f"{head}.empty", 0)
        fraction = positive / (positive + empty)
        if not 0.45 <= fraction <= 0.55:
            raise QueryScheduleError(
                f"{split} {head} positive fraction {fraction:.4f} is outside [0.45, 0.55]"
            )
        if split == "train":
            counts = [
                metrics["class_counts"].get(f"{head}.{class_name}", 0)
                for class_name in RECOGNITION_CLASS_VOCABULARIES[head]
                if class_name != "EMPTY"
            ]
            ratio = max(counts) / min(counts)
            if ratio > 1.25:
                raise QueryScheduleError(
                    f"train {head} positive class ratio {ratio:.4f} exceeds 1.25"
                )
    return metrics


def load_query_plans(path: Path) -> list[JsonDict]:
    return read_jsonl(path)
