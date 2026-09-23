"""Frozen strict scorer; function source text is part of saved manifest identity."""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Sequence

from evals.catan_board_bench.ascii_variations.schema import JsonDict

STRICT_SCORER_VERSION = "strict_typed_json/v2"


def score_strict_json_answer(expected: JsonDict, response: str) -> JsonDict:
    stripped = response.strip()
    try:
        parsed = json.loads(
            stripped,
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "correct": False,
            "json_valid": False,
            "protocol_exact": False,
            "error": str(exc),
            "parsed": None,
        }
    if not isinstance(parsed, dict):
        return {
            "correct": False,
            "json_valid": True,
            "protocol_exact": False,
            "error": "response must be one JSON object",
            "parsed": parsed,
        }
    semantic_correct = _semantic_equal(expected, parsed)
    return {
        "correct": semantic_correct,
        "json_valid": True,
        "protocol_exact": stripped == canonical_answer_text(expected),
        "error": None if semantic_correct else "parsed JSON does not exactly match",
        "parsed": parsed,
    }


def canonical_answer_text(answer: JsonDict) -> str:
    return json.dumps(answer, separators=(",", ":"), sort_keys=True)


def strict_scorer_digest() -> str:
    payload = "\n".join(
        (
            STRICT_SCORER_VERSION,
            inspect.getsource(score_strict_json_answer),
            inspect.getsource(_reject_duplicate_object_pairs),
            inspect.getsource(_semantic_equal),
            inspect.getsource(_stable_value),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _reject_duplicate_object_pairs(
    pairs: Sequence[tuple[str, Any]],
) -> JsonDict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _semantic_equal(expected: Any, actual: Any) -> bool:
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return set(expected) == set(actual) and all(
            _semantic_equal(expected[key], actual[key]) for key in expected
        )
    if isinstance(expected, list):
        expected_keys = [_stable_value(item) for item in expected]
        actual_keys = [_stable_value(item) for item in actual]
        if len(actual_keys) != len(set(actual_keys)):
            return False
        return sorted(expected_keys) == sorted(actual_keys)
    return expected == actual


def _stable_value(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)
