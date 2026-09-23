"""Answer normalization, token repair, and provider message extraction."""

from __future__ import annotations

import re
from collections.abc import Sequence

from cle.players.data import JsonValue
from scripts.board_bench.run.eval_catan_board_bench_openrouter.constants import COLOR_TOKEN_NAMES
from scripts.board_bench.shapes import JsonDict

__all__ = [
    "contains_bare_count",
    "contains_count",
    "contains_labeled_count",
    "contains_value",
    "edge_tokens",
    "extract_content",
    "extract_message_text",
    "model_supports_reasoning_control",
    "normalize_text",
    "number_tokens",
    "repair_merged_tokens",
    "repair_partial_tokens",
]


def extract_content(value: JsonValue) -> str:
    if isinstance(value, list):
        parts = [extract_content(part) for part in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in (
            "text",
            "content",
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "summary",
        ):
            if key in value:
                text = extract_content(value[key])
                if text:
                    return text
        candidates = [
            extract_content(item)
            for item in value.values()
            if isinstance(item, (str, list, dict))
        ]
        return "\n".join(part for part in candidates if part).strip()
    return ""


def extract_message_text(message: JsonDict) -> str:
    """
    Qwen 3.5 may place returned text in reasoning-oriented fields.
    Check all known locations and fallback through nested payload structures.
    """
    if not isinstance(message, dict):
        return ""
    fields = (
        "content",
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "analysis",
        "summary",
    )
    for field in fields:
        text = extract_content(message.get(field))
        if text:
            return text
    return extract_content(message)


def model_supports_reasoning_control(model_id: str) -> bool:
    # Recent Qwen variants may reason by default on OpenRouter routes.
    # Disable it for short-answer visual scoring and bounded completions.
    model_id_lower = model_id.lower()
    return any(family in model_id_lower for family in ("qwen3.5", "qwen3.6", "qwen3.7", "qwen3.8"))


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


def contains_value(normalized_response: str, expected_token: str | None) -> bool:
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
    expected_count: int,
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
    expected_count: int,
) -> bool:
    return contains_labeled_count(
        normalized_response, labels, expected_count
    ) or contains_bare_count(
        normalized_response,
        expected_count,
    )


def contains_bare_count(normalized_response: str, expected_count: int) -> bool:
    numbers = re.findall(r"\b\d+\b", normalized_response)
    return len(numbers) == 1 and numbers[0] == str(expected_count)
