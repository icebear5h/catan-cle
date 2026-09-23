"""Coverage, balance, and identity checks for deterministic query plans."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from data_pipeline.board_recognition import query_schedule as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_dict, as_int, as_list, as_str


def _queries(plan: JsonDict) -> list[JsonDict]:
    """The query records of one state plan."""

    return [as_dict(query) for query in as_list(plan["queries"])]


def _counts(metrics: JsonDict, key: str) -> dict[str, int]:
    """One counter section of a plan-metrics payload."""

    return {name: as_int(value) for name, value in as_dict(metrics[key]).items()}


def query_plan_metrics(plans: Sequence[JsonDict]) -> JsonDict:
    entity_counts: Counter[str] = Counter()
    head_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    slot_counts: dict[str, Counter[str]] = defaultdict(Counter)
    polarity_counts: Counter[str] = Counter()
    duplicates = 0
    for plan in plans:
        seen: set[tuple[str, str]] = set()
        for query in _queries(plan):
            key = (as_str(query["head"]), as_str(query["slot"]))
            duplicates += key in seen
            seen.add(key)
            entity_counts[as_str(query["entity_type"])] += 1
            head_counts[as_str(query["head"])] += 1
            class_counts[f"{query['head']}.{query['class_name']}"] += 1
            slot_counts[as_str(query["head"])][as_str(query["slot"])] += 1
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
        raise api.QueryScheduleError(
            f"query plan has {len(plans)}/{expected_states} states for {split}"
        )
    state_ids = [as_str(plan["state_id"]) for plan in plans]
    if len(state_ids) != len(set(state_ids)):
        raise api.QueryScheduleError(f"query plan duplicated state IDs in {split}")
    expected_per_entity = queries_per_state // len(api.ENTITY_TYPES)
    for plan in plans:
        if plan.get("epoch") != epoch_index:
            raise api.QueryScheduleError(f"state {plan['state_id']} has the wrong epoch")
        queries = _queries(plan)
        if len(queries) != queries_per_state:
            raise api.QueryScheduleError(f"state {plan['state_id']} has {len(queries)} queries")
        entity_counts = Counter(as_str(query["entity_type"]) for query in queries)
        if entity_counts != Counter({entity: expected_per_entity for entity in api.ENTITY_TYPES}):
            raise api.QueryScheduleError(
                f"state {plan['state_id']} entity balance is {dict(entity_counts)}"
            )
        keys = [(as_str(query["head"]), as_str(query["slot"])) for query in queries]
        if len(keys) != len(set(keys)):
            raise api.QueryScheduleError(f"state {plan['state_id']} repeats a head/slot query")
        for query in queries:
            if query["answer_token"] != api.recognition_answer_token(
                as_str(query["head"]), as_str(query["class_name"])
            ):
                raise api.QueryScheduleError(f"state {plan['state_id']} answer token mismatch")
            if query["query_token"] != api.recognition_query_token(as_str(query["head"])):
                raise api.QueryScheduleError(f"state {plan['state_id']} query token mismatch")
    metrics = api.query_plan_metrics(plans)
    if metrics["duplicates"]:
        raise api.QueryScheduleError(f"query plan contains {metrics['duplicates']} duplicates")
    return metrics


def validate_split_balance(plans: Sequence[JsonDict], *, split: str) -> JsonDict:
    metrics = api.query_plan_metrics(plans)
    expected_slots = {
        "tile.resource": 19,
        "tile.number": 19,
        "tile.robber": 19,
        "node.occupancy": 54,
        "edge.owner": 72,
        "port.port_type": 9,
    }
    slot_counts = {
        head: as_dict(value) for head, value in as_dict(metrics["slot_counts"]).items()
    }
    missing_slots = {
        head: expected - len(slot_counts.get(head, {}))
        for head, expected in expected_slots.items()
        if len(slot_counts.get(head, {})) != expected
    }
    if missing_slots:
        raise api.QueryScheduleError(f"{split} query plan lacks slot coverage: {missing_slots}")

    required_heads = set(api.RECOGNITION_CLASS_VOCABULARIES)
    if split in {"validation", "test"}:
        required_heads = {
            "tile.resource", "tile.number", "tile.robber", "port.port_type",
        }
    class_counts = _counts(metrics, "class_counts")
    missing_classes = []
    for head in sorted(required_heads):
        for class_name in api.RECOGNITION_CLASS_VOCABULARIES[head]:
            key = f"{head}.{class_name}"
            if class_counts.get(key, 0) == 0:
                missing_classes.append(key)
    if missing_classes:
        raise api.QueryScheduleError(f"{split} query plan lacks required classes: {missing_classes}")

    polarity_counts = _counts(metrics, "polarity_counts")
    for head in ("node.occupancy", "edge.owner"):
        positive = polarity_counts.get(f"{head}.positive", 0)
        empty = polarity_counts.get(f"{head}.empty", 0)
        fraction = positive / (positive + empty)
        if not 0.45 <= fraction <= 0.55:
            raise api.QueryScheduleError(
                f"{split} {head} positive fraction {fraction:.4f} is outside [0.45, 0.55]"
            )
        if split == "train":
            counts = [
                class_counts.get(f"{head}.{class_name}", 0)
                for class_name in api.RECOGNITION_CLASS_VOCABULARIES[head]
                if class_name != "EMPTY"
            ]
            ratio = max(counts) / min(counts)
            if ratio > 1.25:
                raise api.QueryScheduleError(
                    f"train {head} positive class ratio {ratio:.4f} exceeds 1.25"
                )
    return metrics


def load_query_plans(path: Path) -> list[JsonDict]:
    return api.read_jsonl(path)
