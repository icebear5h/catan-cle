"""Strict, task-specific spatial answer scoring."""

from __future__ import annotations

import json
from typing import cast

from sft.board.spatial_tasks._contracts import _check_color_roll, _check_resource_number
from sft.board.spatial_tasks._topology import (
    RESOURCE_KEYS,
    TASK_TYPES,
    _topology,
    node_tile_tokens,
    shortest_node_path,
)
from sft.json_types import JsonDict, JsonLikeDict, as_str


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_number(value: str) -> None:
    raise ValueError(f"non-integer JSON number: {value}")


def _json_answer(text: str, task: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(text, str):
        raise ValueError("answer must be JSON text")
    answer = json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_float=_reject_json_number,
        parse_constant=_reject_json_number,
    )
    if not isinstance(answer, dict) or set(answer) != keys:
        raise ValueError("answer must be an object with exactly the required keys")
    for value in answer.values():
        if task == "dice_production":
            if type(value) is not int or value < 0:
                raise ValueError("production counts must be nonnegative integers")
        else:
            if not isinstance(value, dict) or set(value) != {"resource", "number"}:
                raise ValueError("tile answer must contain exactly resource and number")
            _check_resource_number(value["resource"], value["number"])
    return answer


def _token_answer(text: str, vocabulary: set[str]) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("answer must be token text")
    tokens = text.split()
    if not tokens or len(tokens) != len(set(tokens)) or not set(tokens) <= vocabulary:
        raise ValueError("answer must contain distinct canonical tokens only")
    return tokens


def _valid_path(path: list[str], start: str, end: str, length: int) -> bool:
    graph = _topology()[0]
    return (
        len(path) == length
        and path[0] == start
        and path[-1] == end
        and all(b in graph[a] for a, b in zip(path, path[1:]))
    )


def score_spatial_task(expected: str, response: str,
                       metadata: JsonDict) -> JsonLikeDict | None:
    """Score the four named tasks; malformed gold/targets raise, other tasks return None.

    Direct callers get whitespace tolerance only. The evaluator applies its existing
    chat-wrapper normalization first. Dynamic gold comes from the contract solvers;
    this boundary validates its complete answer shape, not an inferred text prefix.
    """

    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    task = metadata.get("task_type")
    if task is not None and not isinstance(task, str):
        raise ValueError("task_type must be a string")
    if task not in TASK_TYPES:
        return None
    target = metadata.get("target")
    target_keys = (
        {"start", "end"}
        if task == "shortest_node_path"
        else {"color", "roll"}
        if task == "dice_production"
        else {"node"}
    )
    if not isinstance(target, dict) or set(target) != target_keys:
        raise ValueError(f"invalid target for {task}: {target!r}")
    gold: list[str] | dict[str, object]
    if task == "shortest_node_path":
        start = cast("str", target["start"])
        end = cast("str", target["end"])
        length = len(shortest_node_path(start, end))
        vocabulary = set(_topology()[0])
        gold = _token_answer(expected, vocabulary)
        if not _valid_path(gold, start, end, length):
            raise ValueError("gold is not a shortest path for its target")
    elif task == "node_tiles":
        keys = set(node_tile_tokens(cast("str", target["node"])))
        vocabulary = {as_str(tile["token"]) for tile in _topology()[2].values()}
        gold = _token_answer(expected, vocabulary)
        if set(gold) != keys:
            raise ValueError("gold tiles disagree with target node")
    else:
        if task == "dice_production":
            _check_color_roll(target["color"], target["roll"])
            keys = set(RESOURCE_KEYS)
        else:
            keys = set(node_tile_tokens(cast("str", target["node"])))
        gold = _json_answer(expected, task, keys)
        if task == "dice_production" and target["roll"] == 7 and any(gold.values()):
            raise ValueError("a roll of seven cannot produce resources")

    response_norm = response.strip() if isinstance(response, str) else response
    try:
        if task in ("node_tiles", "shortest_node_path"):
            tokens = _token_answer(response, vocabulary)
            correct = (
                set(tokens) == set(gold)
                if task == "node_tiles"
                else _valid_path(tokens, start, end, length)
            )
            response_norm = " ".join(sorted(tokens) if task == "node_tiles" else tokens)
        else:
            decoded = _json_answer(response, task, keys)
            correct = decoded == gold
            response_norm = json.dumps(decoded, sort_keys=True)
    except ValueError:
        correct = False
    expected_norm = (
        json.dumps(gold, sort_keys=True)
        if isinstance(gold, dict)
        else " ".join(sorted(gold) if task == "node_tiles" else gold)
    )
    return {
        "correct": correct,
        "scoring": task,
        "expected_normalized": expected_norm,
        "response_normalized": response_norm,
    }
