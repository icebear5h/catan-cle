"""Strict stored-gold scoring and admission for reviewed and admitted board fluency."""

from __future__ import annotations

import json
import re
from functools import lru_cache

from evals.catan_board_bench.tokens import atlas_tokens
from sft.symbolic_board_tasks import strict_json


SCHEMA = "catan_board_fluency_review/v1"
SFT_SCHEMA = "catan_board_fluency_sft/v1"
SCHEMAS = frozenset({SCHEMA, SFT_SCHEMA})
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
OPERATION_FAMILY = {op: family for family, operations in FAMILIES.items() for op in operations}
SET_OPERATIONS = {
    "owned_buildings_touching_resource": "N",
    "owned_incident_roads": "E",
    "port_access": "P",
    "coverage_union": None,
    "coverage_intersection": None,
    "coverage_difference": None,
    "coverage_missing": None,
    "resource_pip_argmax": None,
    "component_roads": "E",
    "reachable_nodes": "N",
    "distance_rule_witnesses": "N",
}
INTEGER_OPERATIONS = frozenset({"node_pip_sum", "component_count", "shortest_distance"})
RESOURCES = frozenset({"brick", "ore", "sheep", "wheat", "wood"})
# Transport suffixes only: never strip atlas tokens or truncate at an interior EOT.
TRANSPORT_TOKENS = (
    "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>",
    "<|finetune_right_pad_id|>", "<|pad|>", "</s>", "<pad>",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=4)
def _atlas_vocabulary(prefix: str) -> frozenset[str]:
    return frozenset(token for token in atlas_tokens() if token.startswith(f"<{prefix}"))


def _strip_transport(response: str) -> str:
    text = response.strip()
    while True:
        for token in TRANSPORT_TOKENS:
            if text.endswith(token):
                text = text[:-len(token)].rstrip()
                break
        else:
            return text


def _keys(value: object, keys: set[str] | frozenset[str]) -> None:
    _require(type(value) is dict and value.keys() == keys, "invalid JSON object keys")


def _resource_vector(value: object, *, signed: bool = False) -> None:
    _keys(value, RESOURCES)
    _require(all(type(n) is int and (signed or n >= 0) for n in value.values()),
             "resource counts must be integers (nonnegative outside delta)")


def _json_answer(text: str, operation: str) -> dict:
    value = strict_json(text)
    if operation == "local_node_tiles":
        _require(type(value) is dict and 1 <= len(value) <= 3
                 and value.keys() <= _atlas_vocabulary("T"), "invalid local tile keys")
        for tile in value.values():
            _keys(tile, {"resource", "number"})
            resource, number = tile["resource"], tile["number"]
            _require(type(resource) is str and resource in RESOURCES | {"desert"},
                     "invalid tile resource")
            _require(number is None if resource == "desert" else
                     type(number) is int and 2 <= number <= 12 and number != 7,
                     "invalid tile number")
    elif operation in {"resource_pip_totals", "roll_production"}:
        _resource_vector(value)
    elif operation == "road_removal_connectivity":
        _keys(value, {"before", "after"})
        _require(all(n is None or (type(n) is int and n >= 0) for n in value.values()),
                 "distances must be nonnegative integers or null")
    else:
        _keys(value, {"before", "after", "delta"})
        for key, vector in value.items():
            _resource_vector(vector, signed=key == "delta")
    return value


def _parse_answer(text: str, operation: str) -> object:
    _require(isinstance(text, str), "answer must be text")
    text = text.strip()
    if operation in SET_OPERATIONS:
        if text == "NONE":
            return frozenset()
        items = text.split()
        prefix = SET_OPERATIONS[operation]
        vocabulary = RESOURCES if prefix is None else _atlas_vocabulary(prefix)
        _require(bool(items) and len(items) == len(set(items)) and set(items) <= vocabulary,
                 "answer must be a unique space-separated item set, or NONE")
        return frozenset(items)
    if operation in INTEGER_OPERATIONS:
        if operation == "shortest_distance" and text == "UNREACHABLE":
            return text
        _require(re.fullmatch(r"0|[1-9][0-9]*", text) is not None,
                 "answer must be a nonnegative integer")
        return int(text)
    return _json_answer(text, operation)


def _typed_equal(expected: object, actual: object) -> bool:
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            _typed_equal(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(
            _typed_equal(a, b) for a, b in zip(expected, actual)
        )
    return expected == actual


def _normalized(value: object) -> str:
    if isinstance(value, frozenset):
        return " ".join(sorted(value)) or "NONE"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def validate_board_fluency_metadata(metadata: dict) -> None:
    """Keep review-only and split-aware SFT admission distinct and fail closed."""
    _require(isinstance(metadata, dict), "metadata must be a dictionary")
    schema = metadata.get("schema")
    _require(schema in SCHEMAS, "unsupported board-fluency schema")
    operation = metadata.get("operation")
    _require(isinstance(operation, str) and operation in OPERATION_FAMILY,
             "unsupported board-fluency operation")
    _require(metadata.get("family") == OPERATION_FAMILY[operation]
             and metadata.get("class") == "board_fluency", "invalid board-fluency family/class")
    if schema == SCHEMA:
        _require(metadata.get("review_only") is True
                 and metadata.get("admitted_for_training") is False, "invalid review declarations")
    else:
        split = metadata.get("split")
        _require(split in {"train", "validation", "test"}, "invalid board-fluency SFT split")
        training = split == "train"
        _require(metadata.get("review_only") is False
                 and metadata.get("admitted_for_training") is training
                 and metadata.get("task_role") == ("train" if training else "component_eval"),
                 "invalid board-fluency SFT admission/role")
        provenance = metadata.get("provenance")
        _require(isinstance(provenance, dict) and provenance.get("split") == split
                 and isinstance(provenance.get("source"), dict)
                 and provenance["source"].get("split") == split, "conflicting SFT source split")
    target = metadata.get("target")
    _require(isinstance(target, dict) and set(target) == {"state", "query"}
             and isinstance(target["state"], dict) and isinstance(target["query"], dict)
             and target["query"].get("operation") == operation, "invalid board-fluency target operation")


def score_board_fluency(expected: str, response: str, metadata: dict) -> dict | None:
    """Bad contracts/golds raise; malformed predictions score False, never generic.

    Gold recomputation belongs to the dataset validator. Only trailing transport
    tokens and surrounding whitespace are ignored when scoring predictions.
    """
    _require(isinstance(metadata, dict), "metadata must be a dictionary")
    schema = metadata.get("schema")
    if schema not in SCHEMAS:
        _require(metadata.get("class") != "board_fluency"
                 and not str(schema).startswith("catan_board_fluency"),
                 "missing or unsupported board-fluency schema")
        return None
    validate_board_fluency_metadata(metadata)
    operation = metadata["operation"]
    _require(isinstance(expected, str) and metadata.get("answer") == expected,
             "stored board-fluency answer differs from expected")
    gold = _parse_answer(expected, operation)
    normalized = response
    try:
        _require(isinstance(response, str), "response must be text")
        normalized = _strip_transport(response)
        answer = _parse_answer(normalized, operation)
        correct = _typed_equal(gold, answer)
        normalized = _normalized(answer)
    except (ValueError, TypeError, RecursionError):
        correct = False
    return {
        "correct": correct,
        "scoring": schema,
        "expected_normalized": _normalized(gold),
        "response_normalized": normalized,
    }
