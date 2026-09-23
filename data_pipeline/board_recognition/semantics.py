"""Controlled natural-language contract for semantic board recognition."""

from __future__ import annotations

from data_pipeline.json_types import JsonDict, JsonValue
from evals.catan_board_bench.tokens import RECOGNITION_CLASS_VOCABULARIES, atlas_tokens

SEMANTIC_CONTRACT_SCHEMA = "catan_board_recognition_semantic_contract/v1"
QUERY_TEXT_BY_HEAD = {
    "tile.resource": "resource?",
    "tile.number": "number?",
    "tile.robber": "robber here?",
    "node.occupancy": "building?",
    "edge.owner": "road?",
    "port.port_type": "port?",
}
SLOT_PREFIX_BY_HEAD = {
    "tile.resource": "<T",
    "tile.number": "<T",
    "tile.robber": "<T",
    "node.occupancy": "<N",
    "edge.owner": "<E",
    "port.port_type": "<P",
}
ATLAS_TOKENS = frozenset(atlas_tokens())


def _words(value: str) -> str:
    return value.lower().replace("_", " ")


def semantic_answer(head: str, class_name: str) -> str:
    """Map one engine-derived recognition class to its legal answer phrase."""

    vocabulary = RECOGNITION_CLASS_VOCABULARIES.get(head)
    if vocabulary is None:
        raise ValueError(f"unknown board-recognition head: {head}")
    if class_name not in vocabulary:
        raise ValueError(f"unknown class {class_name!r} for {head}")

    if head == "tile.resource":
        return _words(class_name)
    if head == "tile.number":
        return "none" if class_name == "NONE" else class_name
    if head == "tile.robber":
        return "yes" if class_name == "PRESENT" else "no"
    if head == "node.occupancy":
        if class_name == "EMPTY":
            return "empty"
        owner, building = class_name.rsplit("_", 1)
        return f"{_words(owner)} {_words(building)}"
    if head == "edge.owner":
        return "empty" if class_name == "EMPTY" else f"{_words(class_name)} road"
    if head == "port.port_type":
        if class_name == "THREE_TO_ONE":
            return "3:1 port"
        return f"{_words(class_name.removeprefix('TWO_TO_ONE_'))} port"
    raise AssertionError(f"unhandled board-recognition head: {head}")


def semantic_prompt(head: str, slot: str) -> str:
    """Render one terse query containing exactly one opaque location token."""

    query_text = QUERY_TEXT_BY_HEAD.get(head)
    if query_text is None:
        raise ValueError(f"unknown board-recognition head: {head}")
    if (
        not isinstance(slot, str)
        or slot not in ATLAS_TOKENS
        or not slot.startswith(SLOT_PREFIX_BY_HEAD[head])
    ):
        raise ValueError(f"slot {slot!r} is invalid for {head}")
    return f"{slot} {query_text}"


def semantic_candidates(head: str) -> tuple[str, ...]:
    """Return the closed legal phrase set for one classification head."""

    vocabulary = RECOGNITION_CLASS_VOCABULARIES.get(head)
    if vocabulary is None:
        raise ValueError(f"unknown board-recognition head: {head}")
    candidates = tuple(semantic_answer(head, class_name) for class_name in vocabulary)
    if len(candidates) != len(set(candidates)):
        raise RuntimeError(f"semantic answer phrases are not unique for {head}")
    return candidates


def semantic_contract() -> JsonDict:
    """Return the complete versioned phrase and engine-class mapping."""

    heads: dict[str, JsonValue] = {}
    for head, query_text in QUERY_TEXT_BY_HEAD.items():
        classes = RECOGNITION_CLASS_VOCABULARIES[head]
        heads[head] = {
            "query_text": query_text,
            "slot_prefix": SLOT_PREFIX_BY_HEAD[head],
            "classes": [
                {
                    "class_name": class_name,
                    "answer": semantic_answer(head, class_name),
                }
                for class_name in classes
            ],
            "candidate_answers": list(semantic_candidates(head)),
        }
    return {
        "schema": SEMANTIC_CONTRACT_SCHEMA,
        "prompt_format": "<location_token> <query_text>",
        "answer_style": "lowercase controlled natural language without punctuation",
        "heads": heads,
    }
