"""Deterministic split-level query scheduling for replay_v1 recognition."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from data_pipeline.board_recognition.query_schedule.selection import (
    _choose_tile_head as _choose_tile_head,
)
from data_pipeline.board_recognition.query_schedule.selection import (
    _is_positive as _is_positive,
)
from data_pipeline.board_recognition.query_schedule.selection import (
    _select_candidate as _select_candidate,
)
from data_pipeline.board_recognition.replay_dataset import read_jsonl as read_jsonl
from data_pipeline.json_coerce import as_dict, as_int, as_list, as_str
from data_pipeline.json_types import JsonDict, JsonValue
from evals.catan_board_bench.tokens import (
    RECOGNITION_CLASS_VOCABULARIES as RECOGNITION_CLASS_VOCABULARIES,
)
from evals.catan_board_bench.tokens import (
    recognition_answer_token as recognition_answer_token,
)
from evals.catan_board_bench.tokens import (
    recognition_query_token as recognition_query_token,
)

from .dynamic import _dynamic_class_quotas as _dynamic_class_quotas
from .dynamic import _schedule_dynamic_head as _schedule_dynamic_head
from .dynamic import nx as nx
from .validation import load_query_plans as load_query_plans
from .validation import query_plan_metrics as query_plan_metrics
from .validation import validate_query_plan as validate_query_plan
from .validation import validate_split_balance as validate_split_balance

ENTITY_TYPES = ("tile", "node", "edge", "port")
COLLECTION_BY_ENTITY = {
    "tile": "tiles", "node": "nodes", "edge": "edges", "port": "ports",
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


def stable_rank(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def normalize_class_name(head: str, value: object) -> str:
    if head == "tile.number":
        return "NONE" if value is None else str(value)
    if head == "tile.robber":
        return "PRESENT" if value else "ABSENT"
    return str(value)


def state_query_candidates(labels: JsonDict, entity_type: str, head: str) -> list[JsonDict]:
    collection = COLLECTION_BY_ENTITY[entity_type]
    attribute = ATTRIBUTE_BY_HEAD[head]
    vocabulary = RECOGNITION_CLASS_VOCABULARIES[head]
    candidates: list[JsonDict] = []
    entities = as_dict(labels["entities"])
    for slot_index, raw_row in enumerate(as_list(entities[collection])):
        row = as_dict(raw_row)
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
        as_str(row["sample_id"]): as_dict(
            json.loads((dataset_dir / as_str(row["label_path"])).read_text())
        )
        for row in ordered_states
    }
    queries_by_state: dict[str, list[JsonValue]] = {
        as_str(row["sample_id"]): [] for row in ordered_states
    }
    plans: dict[str, JsonDict] = {
        as_str(row["sample_id"]): {
            "schema": "catan_board_recognition_query_plan/v1",
            "state_id": row["sample_id"],
            "split": split,
            "epoch": epoch_index,
            "queries": queries_by_state[as_str(row["sample_id"])],
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
            state_id = as_str(state["sample_id"])
            for selected in scheduled[state_id]:
                selected_keys[state_id].add((head, as_str(selected["slot"])))
                head_counts[head] += 1
                class_counts[(head, as_str(selected["class_name"]))] += 1
                slot_counts[(head, as_str(selected["slot"]))] += 1
                polarity = "positive" if _is_positive(selected) else "empty"
                polarity_counts[(head, polarity)] += 1
                queries_by_state[state_id].append(selected)

    for round_index in range(rounds_per_entity):
        for entity_type in ("tile", "port"):
            for state in ordered_states:
                state_id = as_str(state["sample_id"])
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
                selected_keys[state_id].add((head, as_str(selected["slot"])))
                head_counts[head] += 1
                class_counts[(head, as_str(selected["class_name"]))] += 1
                slot_counts[(head, as_str(selected["slot"]))] += 1
                if head in {"node.occupancy", "edge.owner"}:
                    polarity = "positive" if _is_positive(selected) else "empty"
                    polarity_counts[(head, polarity)] += 1
                queries_by_state[state_id].append(selected)

    output = []
    rows_by_id = {as_str(row["sample_id"]): row for row in state_rows}
    for state_id in sorted(plans):
        plan = plans[state_id]
        queries = queries_by_state[state_id]
        queries.sort(
            key=lambda row: (
                ENTITY_TYPES.index(as_str(as_dict(row)["entity_type"])),
                as_int(as_dict(row)["attribute_id"]),
                as_str(as_dict(row)["slot"]),
            )
        )
        for query_index, raw_query in enumerate(queries):
            as_dict(raw_query)["query_id"] = (
                f"{state_id}_e{epoch_index:03d}_q{query_index:03d}"
            )
        state_source = as_dict(rows_by_id[state_id]["source"])
        plan["source"] = {
            "image_path": rows_by_id[state_id]["image_path"],
            "label_path": rows_by_id[state_id]["label_path"],
            "source_kind": state_source["kind"],
            "game_id": state_source.get("game_id"),
            "trajectory_id": state_source["trajectory_id"],
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
