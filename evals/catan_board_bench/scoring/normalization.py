"""Response normalization, token repair, and containment predicates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Iterable, Optional, Sequence

from evals.catan_board_bench.scoring.categories import (
    CATEGORY_ALIASES,
    COLOR_TOKEN_NAMES,
    JsonDict,
)
from evals.json_types import JsonValue


def find_by_token(items: Iterable[JsonDict], token_value: object) -> Optional[JsonDict]:
    for item in items:
        if item.get("token") == token_value:
            return item
    return None



def canonical_category(category: str) -> str:
    return CATEGORY_ALIASES.get(category, category)


def _token_text(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string token, got {type(value).__name__}")
    return value


def resource_token(target: Mapping[str, JsonValue]) -> str:
    if target.get("resource_token"):
        return _token_text(target["resource_token"], "resource_token")
    resource = target.get("resource")
    if resource:
        return f"<{resource}>"
    return "<DESERT>"


def color_token(target: Mapping[str, JsonValue], *, color_key: str = "color") -> Optional[str]:
    token_key = f"{color_key}_token"
    if target.get(token_key):
        return _token_text(target[token_key], token_key)
    if target.get("color_token") and color_key != "color":
        return _token_text(target["color_token"], "color_token")
    color = target.get(color_key, target.get("color"))
    if color:
        return f"<{color}>"
    return None


def building_token(target: Mapping[str, JsonValue]) -> Optional[str]:
    if target.get("building_token"):
        return _token_text(target["building_token"], "building_token")
    building = target.get("building")
    if building:
        return f"<{building}>"
    return None


def component_score(*checks: bool) -> JsonDict:
    total = len(checks)
    correct_components = sum(bool(check) for check in checks)
    return {
        "correct": correct_components == total,
        "component_correct": correct_components,
        "component_total": total,
        "component_accuracy": correct_components / total if total else 0.0,
    }


def normalize_text(value: object) -> str:
    text = str(value).upper().strip()
    replacements = {
        "NO NUMBER": "NO_NUMBER",
        "NO-NUMBER": "NO_NUMBER",
        "NO PLAYER": "NONE",
        "NO ONE": "NONE",
        "NONE.": "NONE",
        "EMPTY.": "EMPTY",
        "DESERT": "<DESERT>",
        "WOOD": "<WOOD>",
        "BRICK": "<BRICK>",
        "SHEEP": "<SHEEP>",
        "WHEAT": "<WHEAT>",
        "ORE": "<ORE>",
        "SETTLEMENT": "<SETTLEMENT>",
        "CITY": "<CITY>",
        "GEN": "GENERIC",
    }
    replacements.update(
        {
            color_name.replace("_", separator): f"<{color_name}>"
            for color_name in COLOR_TOKEN_NAMES
            if "_" in color_name
            for separator in (" ", "-")
        }
    )
    replacements.update({color_name: f"<{color_name}>" for color_name in COLOR_TOKEN_NAMES})
    for src, dst in replacements.items():
        text = re.sub(rf"(?<![A-Z0-9_<]){re.escape(src)}(?![A-Z0-9_>])", dst, text)
    text = repair_merged_tokens(text)
    text = repair_partial_tokens(text)
    return re.sub(r"\s+", " ", text)


def repair_merged_tokens(text: str) -> str:
    colors = "|".join(re.escape(color_name) for color_name in COLOR_TOKEN_NAMES)
    pieces = "SETTLEMENT|CITY|ROAD"
    return re.sub(rf"<({colors})_({pieces})>", r"<\1> <\2>", text)


def repair_partial_tokens(text: str) -> str:
    named_tokens = [
        "WOOD",
        "BRICK",
        "SHEEP",
        "WHEAT",
        "ORE",
        "DESERT",
        "RED",
        "BLUE",
        "WHITE",
        "BLACK",
        "GREEN",
        "ORANGE",
        "SETTLEMENT",
        "CITY",
        "ROAD",
        *COLOR_TOKEN_NAMES,
    ]
    for token in named_tokens:
        text = re.sub(rf"<{token}(?![A-Z0-9_]*>)", f"<{token}>", text)

    text = re.sub(
        r"<T(\d{1,2})(?![0-9_]*>)",
        lambda match: f"<T{int(match.group(1)):02d}>",
        text,
    )
    text = re.sub(
        r"<N(\d{1,2})(?![0-9_]*>)",
        lambda match: f"<N{int(match.group(1)):02d}>",
        text,
    )
    text = re.sub(
        r"\bT(\d{1,2})\b",
        lambda match: f"<T{int(match.group(1)):02d}>",
        text,
    )
    text = re.sub(
        r"\bN(\d{1,2})\b",
        lambda match: f"<N{int(match.group(1)):02d}>",
        text,
    )
    text = re.sub(
        r"<E(\d{1,2})[_-](\d{1,2})(?![0-9_]*>)",
        lambda match: _edge_token_from_match(match),
        text,
    )
    text = re.sub(
        r"\bE(\d{1,2})[_-](\d{1,2})\b",
        lambda match: _edge_token_from_match(match),
        text,
    )
    return text


def _edge_token_from_match(match: re.Match[str]) -> str:
    a = int(match.group(1))
    b = int(match.group(2))
    lo, hi = sorted((a, b))
    return f"<E{lo:02d}_{hi:02d}>"


def contains_value(normalized_response: str, expected_token: object) -> bool:
    if expected_token is None:
        return False
    return normalize_text(expected_token) in normalized_response


def number_tokens(normalized_response: str) -> set[str]:
    return set(re.findall(r"\b(?:2|3|4|5|6|8|9|10|11|12)\b", normalized_response))


def edge_tokens(normalized_response: str) -> set[str]:
    return set(re.findall(r"<E\d{2}_\d{2}>", normalized_response))


def contains_labeled_count(
    normalized_response: str,
    labels: Sequence[str],
    expected_count: object,
) -> bool:
    label_forms = set(labels)
    label_forms.update(normalize_text(label) for label in labels)
    count = str(expected_count)
    for label in label_forms:
        escaped = re.escape(label)
        if re.search(rf"{escaped}\D{{0,20}}\b{count}\b", normalized_response):
            return True
        if re.search(rf"\b{count}\b\D{{0,20}}{escaped}", normalized_response):
            return True
    return False


def contains_count(
    normalized_response: str,
    labels: Sequence[str],
    expected_count: object,
) -> bool:
    return contains_labeled_count(
        normalized_response, labels, expected_count
    ) or contains_bare_count(
        normalized_response,
        expected_count,
    )


def contains_bare_count(normalized_response: str, expected_count: object) -> bool:
    numbers = re.findall(r"\b\d+\b", normalized_response)
    return len(numbers) == 1 and numbers[0] == str(expected_count)

