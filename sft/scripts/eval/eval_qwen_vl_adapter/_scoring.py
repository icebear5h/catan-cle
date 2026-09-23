from __future__ import annotations

import importlib
import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from sft.board.board_fluency_scoring import SCHEMA, score_board_fluency
from sft.board.coordinate_comparison import score_coordinate_comparison
from sft.board.spatial_tasks import score_spatial_task
from sft.board.symbolic_board_tasks import score_symbolic_task
from sft.board_state_readout import TASK as TASK
from sft.board_state_readout import score_board_state
from sft.json_types import JsonDict, JsonLike, JsonList, JsonValue, as_dict, as_list, loads_json

if TYPE_CHECKING:  # Heavy; the eval path imports torch/transformers lazily.
    import torch as torch_module
    from transformers import PreTrainedTokenizerBase

FULL_BOARD_TASK = TASK

BOARD_FLUENCY_SCHEMA = SCHEMA


def iter_jsonl(path: Path) -> Iterator[tuple[int, JsonValue]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                yield line_number, loads_json(line)


def _text_parts(content: JsonValue) -> str:
    parts = [as_dict(item) for item in as_list(content)]
    return "\n".join(
        str(item.get("text", "")) for item in parts if item.get("type") == "text"
    ).strip()


def user_text(row: JsonDict) -> str:
    if "conversations" in row:
        value = str(as_dict(as_list(row["conversations"])[0])["value"])
        return value.replace("<image>", "", 1).strip()
    content = as_dict(as_list(row["messages"])[0])["content"]
    if isinstance(content, str):
        return content.replace("<image>", "", 1).strip()
    return _text_parts(content)


def expected_text(row: JsonDict) -> str:
    if "conversations" in row:
        return str(as_dict(as_list(row["conversations"])[1])["value"]).strip()
    content = as_dict(as_list(row["messages"])[1])["content"]
    if isinstance(content, str):
        return content.strip()
    return _text_parts(content)


def _json_value(value: JsonLike) -> JsonValue:
    """Copy a write-side JSON value into the read-side shape records are built from."""

    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return _json_dict(value)
    return [_json_value(item) for item in value]


def _json_dict(value: Mapping[str, JsonLike]) -> JsonDict:
    return {key: _json_value(item) for key, item in value.items()}


LONG_ANSWER_CHARACTERS = 48
LONG_ANSWER_TASK_TYPES = {"terrain_readout", "node_readout", "edge_readout", FULL_BOARD_TASK}


def is_long_answer(row: JsonDict) -> bool:
    """Rows whose expected answer needs the long generation budget.

    Full-board readouts answer with every tile and port keyed by atlas token,
    far past the 16-token budget the one-token and one-phrase heads use.
    """

    if row.get("task_type") in LONG_ANSWER_TASK_TYPES or as_dict(row.get("metadata") or {}).get("task_type") in LONG_ANSWER_TASK_TYPES:
        return True
    return len(expected_text(row)) > LONG_ANSWER_CHARACTERS


def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.split("<|im_end|>", 1)[0].strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(?:answer|assistant)\s*:\s*", "", text, flags=re.I)
    return text.strip()


def extract_json_object(text: str) -> JsonValue:
    text = normalize_text(text)
    try:
        return loads_json(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return loads_json(text[start : end + 1])
    except json.JSONDecodeError:
        return None


READOUT_ITEM_RE = re.compile(r"(<[NETP][0-9_]+>)\s*([^;<]*)")


def readout_items(text: str) -> dict[str, str]:
    """Parse ``<T00> wood 11; <P00> 3:1 port`` into token -> value, whitespace-tolerant."""

    return {token: " ".join(value.split()) for token, value in READOUT_ITEM_RE.findall(text)}


def score_readout(expected: str, response: str) -> JsonDict:
    expected_items = readout_items(expected)
    response_items = readout_items(response)
    matched = sum(1 for token, value in expected_items.items() if response_items.get(token) == value)
    occupied = {token: value for token, value in expected_items.items() if value != "empty"}
    occupied_matched = sum(1 for token, value in occupied.items() if response_items.get(token) == value)
    return {
        "correct": matched == len(expected_items) and len(response_items) == len(expected_items),
        "scoring": "readout_items",
        "expected_normalized": "; ".join(f"{token} {value}" for token, value in expected_items.items()),
        "response_normalized": "; ".join(f"{token} {value}" for token, value in response_items.items()),
        "items_correct": matched,
        "items_total": len(expected_items),
        "items_extra": max(0, len(response_items) - len(expected_items)),
        # A full-list readout is mostly ``empty`` on a real board; the occupied
        # items are the ones that carry the board state, so they are scored apart.
        "occupied_items_correct": occupied_matched,
        "occupied_items_total": len(occupied),
        "empty_items_correct": matched - occupied_matched,
        "empty_items_total": len(expected_items) - len(occupied),
    }


def score_response(
    expected: str, response: str, *, metadata: JsonDict | None = None,
) -> JsonDict:
    if metadata is not None:
        comparison_score = score_coordinate_comparison(expected, response, metadata)
        if comparison_score is not None:
            return _json_dict(comparison_score)
        board_fluency_score = score_board_fluency(expected, response, metadata)
        if board_fluency_score is not None:
            return _json_dict(board_fluency_score)
        symbolic_score = score_symbolic_task(expected, response, metadata)
        if symbolic_score is not None:
            return symbolic_score
    expected_norm = normalize_text(expected)
    response_norm = normalize_text(response)
    if metadata is not None:
        spatial_score = score_spatial_task(expected_norm, response_norm, metadata)
        if spatial_score is not None:
            return _json_dict(spatial_score)
    if "; robber " in expected_norm and len(readout_items(expected_norm)) >= 154:
        return _json_dict(score_board_state(expected_norm, response_norm))
    if len(readout_items(expected_norm)) >= 4:
        return score_readout(expected_norm, response_norm)

    expected_json = extract_json_object(expected_norm)
    if expected_json is not None:
        response_json = extract_json_object(response_norm)
        correct = response_json == expected_json
        return {
            "correct": correct,
            "scoring": "json_exact",
            "expected_normalized": json.dumps(expected_json, sort_keys=True),
            "response_normalized": (
                json.dumps(response_json, sort_keys=True)
                if response_json is not None
                else response_norm
            ),
        }

    return {
        "correct": expected_norm == response_norm,
        "scoring": "text_exact",
        "expected_normalized": expected_norm,
        "response_normalized": response_norm,
    }


def candidate_token_ids(tokenizer: PreTrainedTokenizerBase,
                        candidates: list[str]) -> list[int]:
    """First answer-token id for each candidate; every candidate must be non-empty."""

    ids: list[int] = []
    for candidate in candidates:
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"candidate {candidate!r} produced no tokens")
        ids.append(int(encoded[0]))
    if len(set(ids)) != len(ids):
        raise ValueError("candidates collide on their first token")
    return ids


def score_candidates(
    first_logits: torch_module.Tensor,
    candidates: list[str],
    token_ids: list[int],
    expected: str,
) -> JsonDict:
    """Rank the closed answer set by first-token log-probability."""

    torch = importlib.import_module("torch")
    log_probs = torch.log_softmax(first_logits.float(), dim=-1)
    scores = log_probs[torch.tensor(token_ids)]
    order = torch.argsort(scores, descending=True).tolist()
    expected_norm = normalize_text(expected)
    expected_index = candidates.index(expected_norm)
    ranked = [candidates[index] for index in order]
    top: JsonList = [
        {"answer": candidates[index], "logprob": float(scores[index])}
        for index in order[:3]
    ]
    return {
        "scoring": "first_token_candidates",
        "candidates": len(candidates),
        "predicted": ranked[0],
        "correct": ranked[0] == expected_norm,
        "expected_rank": ranked.index(expected_norm) + 1,
        "expected_logprob": float(scores[expected_index]),
        "top": top,
    }
