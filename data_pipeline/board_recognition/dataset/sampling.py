"""Entity-balanced, class-balanced query sampling for dense board states."""

from __future__ import annotations

from collections import Counter
from typing import Sequence, cast

from data_pipeline.board_recognition import dataset as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_dict, as_list, as_str
from data_pipeline.json_types import JsonValue

ENTITY_TYPES = ("tile", "node", "edge", "port")
ENTITY_TYPE_IDS = {name: index for index, name in enumerate(ENTITY_TYPES)}
COLLECTION_BY_ENTITY = {
    "tile": "tiles", "node": "nodes", "edge": "edges", "port": "ports",
}
CLASS_VOCAB_KEY = {
    ("tile", "resource"): "tile_resource",
    ("tile", "number"): "tile_number",
    ("tile", "robber"): "tile_robber",
    ("node", "occupancy"): "node_occupancy",
    ("edge", "owner"): "edge_owner",
    ("port", "port_type"): "port_type",
}


def normalize_class_name(attribute: str, value: object) -> str:
    if attribute == "number":
        return "NONE" if value is None else str(value)
    if attribute == "robber":
        return "PRESENT" if value else "ABSENT"
    return str(value)


def _attributes_by_entity(spec: JsonDict) -> JsonDict:
    """The per-entity attribute order declared by a curriculum spec."""

    return as_dict(as_dict(spec["sampling"])["attributes_by_entity_type"])


def _optional_str(value: JsonValue) -> str | None:
    """A string field that may be absent from a forced-query payload."""

    return None if value is None else as_str(value)


def class_vocabulary(spec: JsonDict, entity_type: str, attribute: str) -> tuple[str, ...]:
    key = api.CLASS_VOCAB_KEY[(entity_type, attribute)]
    return tuple(
        api.normalize_class_name(attribute, value)
        for value in as_list(as_dict(spec["class_vocabularies"])[key])
    )


def attribute_ids(spec: JsonDict) -> dict[tuple[str, str], int]:
    identifiers: dict[tuple[str, str], int] = {}
    for entity_type in api.ENTITY_TYPES:
        for attribute in as_list(_attributes_by_entity(spec)[entity_type]):
            identifiers[(entity_type, as_str(attribute))] = len(identifiers)
    return identifiers


def sample_state_queries(
    dense_labels: JsonDict,
    *,
    spec: JsonDict,
    queries_per_state: int,
    seed: int,
    epoch: int = 0,
    forced_query: JsonDict | None = None,
    sampling_key: str | None = None,
    sampling_dense_labels: JsonDict | None = None,
) -> list[JsonDict]:
    """Sample an exactly entity-balanced query set from one dense state."""

    if queries_per_state <= 0 or queries_per_state % len(api.ENTITY_TYPES):
        raise ValueError("queries_per_state must be positive and divisible by four")
    state_id = as_str(dense_labels["sample_id"])
    query_sampling_key = sampling_key or state_id
    entities = as_dict(dense_labels["entities"])
    sampling_entities = as_dict((sampling_dense_labels or dense_labels)["entities"])
    attr_ids = api.attribute_ids(spec)
    state_offset = int(api.hashlib.sha256(query_sampling_key.encode()).hexdigest()[:16], 16)
    occurrences: Counter[str] = api.Counter()
    forced_entity_type = (forced_query or {}).get("entity_type")

    queries: list[JsonDict] = []
    for query_index in range(queries_per_state):
        entity_type = api.ENTITY_TYPES[query_index % len(api.ENTITY_TYPES)]
        occurrence = occurrences[entity_type]
        occurrences[entity_type] += 1
        attributes = [
            as_str(name) for name in as_list(_attributes_by_entity(spec)[entity_type])
        ]
        rows = [as_dict(row) for row in as_list(entities[api.COLLECTION_BY_ENTITY[entity_type]])]
        sampling_rows = [
            as_dict(row)
            for row in as_list(sampling_entities[api.COLLECTION_BY_ENTITY[entity_type]])
        ]
        if entity_type == forced_entity_type and occurrence == 0:
            forced = cast(JsonDict, forced_query)
            attribute = as_str(forced["attribute"])
            slot_index = next(
                index for index, row in enumerate(rows) if row["id"] == forced["entity_id"]
            )
        else:
            attribute = attributes[(state_offset + epoch + occurrence) % len(attributes)]
            sampling_slot_index = api.class_balanced_slot_index(
                sampling_rows,
                attribute=attribute,
                vocabulary=api.class_vocabulary(spec, entity_type, attribute),
                seed=api.stable_seed(
                    seed, epoch, query_sampling_key, entity_type, attribute, occurrence,
                ),
                excluded_slot=(
                    _optional_str((forced_query or {}).get("entity_id"))
                    if entity_type == forced_entity_type
                    else None
                ),
            )
            sampled_slot = as_str(sampling_rows[sampling_slot_index]["id"])
            slot_index = next(index for index, row in enumerate(rows) if row["id"] == sampled_slot)
        row = rows[slot_index]
        class_name = api.normalize_class_name(attribute, row[attribute])
        vocabulary = api.class_vocabulary(spec, entity_type, attribute)
        if class_name not in vocabulary:
            raise ValueError(
                f"unknown class {class_name!r} for {entity_type}.{attribute} in {state_id}"
            )
        queries.append(
            {
                "query_id": f"{state_id}_e{epoch:04d}_q{query_index:03d}",
                "entity_type": entity_type,
                "entity_type_id": api.ENTITY_TYPE_IDS[entity_type],
                "attribute": attribute,
                "attribute_id": attr_ids[(entity_type, attribute)],
                "head": f"{entity_type}.{attribute}",
                "slot": row["id"],
                "slot_index": slot_index,
                "class_name": class_name,
                "class_index": vocabulary.index(class_name),
                "class_vocabulary": list(vocabulary),
            }
        )
    if api.Counter(as_str(query["entity_type"]) for query in queries) != {
        entity_type: queries_per_state // len(api.ENTITY_TYPES) for entity_type in api.ENTITY_TYPES
    }:
        raise AssertionError("query sampler lost entity-type balance")
    return queries


def class_balanced_slot_index(
    rows: Sequence[JsonDict],
    *,
    attribute: str,
    vocabulary: Sequence[str],
    seed: int,
    excluded_slot: str | None = None,
) -> int:
    indices_by_class: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row["id"] == excluded_slot:
            continue
        class_name = api.normalize_class_name(attribute, row[attribute])
        indices_by_class.setdefault(class_name, []).append(index)
    available_classes = [name for name in vocabulary if name in indices_by_class]
    if not available_classes:
        raise ValueError(f"no available classes for attribute {attribute}")
    rng = api.random.Random(seed)
    class_name = available_classes[rng.randrange(len(available_classes))]
    candidates = indices_by_class[class_name]
    return candidates[rng.randrange(len(candidates))]


def stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts)
    return int(api.hashlib.sha256(payload.encode()).hexdigest()[:16], 16)
