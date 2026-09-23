"""Artifact writers plus response selection parsing and validation."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

from cle.harness.reasoning import NativeReasoningRequest, native_reasoning_request
from evals.json_types import JsonDict, JsonValue, as_dict, as_dicts, as_list, as_str
from evals.replay_action_diff.contracts import (
    COMPARISON_PARSER_VERSION,
    SELECTION_CONTRACT,
    list_or_empty,
    object_or_empty,
)


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    return [
        as_dict(json.loads(line), f"{path.name} row")
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _latest_responses(rows: Sequence[JsonDict]) -> dict[Tuple[str, str], JsonDict]:
    latest: dict[Tuple[str, str], JsonDict] = {}
    for row in rows:
        key = (as_str(row["decision_id"], "decision_id"), as_str(row["model_id"], "model_id"))
        latest[key] = row
    return latest


def _parse_shared_action_index(
    raw_response: str,
    action_count: int,
) -> Tuple[Optional[int], Optional[str]]:
    """Parse only the shared indexed-action contract from immutable raw text."""
    action_tags = re.findall(
        r"<action>\s*(.*?)\s*</action>",
        raw_response or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    numeric_tags = [value for value in action_tags if value.strip().isdigit()]
    errors = []
    if len(set(numeric_tags)) > 1:
        errors.append("Model returned conflicting numeric action tags; used the last one")
    action_text = numeric_tags[-1] if numeric_tags else None
    if action_text is None:
        direct = re.findall(
            r"(?:action|move)(?:_index)?\s*[:=]\s*(\d+)",
            raw_response or "",
            flags=re.IGNORECASE,
        )
        action_text = direct[-1] if direct else None
    if action_text is None:
        errors.append("Model did not return a parseable action index")
        return None, "; ".join(errors)
    candidate = int(action_text.strip())
    if not 0 <= candidate < action_count:
        errors.append(
            f"Model selected action {candidate}, outside the valid range 0-{action_count - 1}"
        )
        return None, "; ".join(errors)
    return candidate, "; ".join(errors) if errors else None


def normalize_response_selection(row: JsonDict) -> JsonDict:
    """Preserve player-parser receipts; reparse only historical indexed output."""
    normalized = deepcopy(row)
    normalized["comparison_parser_version"] = COMPARISON_PARSER_VERSION
    if normalized.get("error") or not normalized.get("result"):
        return normalized

    result = as_dict(normalized["result"], "result")
    available_actions = list_or_empty(result.get("available_actions"), "available_actions")
    # Unmarked shared-harness receipts through v10 used indexed XML.
    historical_indexed = (
        "selection_contract" not in result
        and (
            result.get("context_version") in (None, "replay-decision-v2")
            or re.fullmatch(
                r"catan-agent@(?:[1-9]|10)\.[0-9]+\.[0-9]+",
                str(result.get("context_version") or ""),
            ) is not None
        )
    )
    action_index: JsonValue
    if historical_indexed:
        if "raw_response" not in result:
            return normalized
        action_index, parse_error = _parse_shared_action_index(
            str(result.get("raw_response") or ""),
            len(available_actions),
        )
        selected = (
            as_dict(available_actions[action_index], "selected action")
            if action_index is not None
            else None
        )
        result.update(
            {
                "action_index": action_index,
                "action": selected.get("action") if selected else None,
                "action_description": selected.get("description") if selected else None,
                "parse_error": parse_error,
            }
        )
    else:
        # Semantic calls require the original context to parse. Their stored
        # PlayerChoice, not numbers embedded in model text, is authoritative.
        action_index = result.get("action_index")
        if (
            result.get("selection_contract") != SELECTION_CONTRACT
            or "parse_error" not in result
            or result["parse_error"] is not None
            or type(action_index) is not int
            or not 0 <= action_index < len(available_actions)
            or not result.get("action")
        ):
            action_index = None
            result.update(
                {
                    "action_index": None,
                    "action": None,
                    "action_description": None,
                    "requested_action_sequence": [],
                    "knight_destination": None,
                    "parse_error": result.get("parse_error")
                    or "Stored player selection is not a validated action in the call menu",
                }
            )
    normalized["model_action_index"] = action_index
    normalized["agreement"] = (
        action_index is not None and action_index == normalized.get("human_action_index")
    )
    has_followup = result.get("knight_destination") is not None
    normalized["agreement_scope"] = "primary_action"
    normalized["agreement_is_coarse"] = has_followup
    normalized["followup_scoring"] = "unscored" if has_followup else "not_requested"
    normalized["followup_agreement"] = None
    return normalized


def validate_response_compatibility(
    manifest: Sequence[JsonDict],
    response_rows: Sequence[JsonDict],
) -> None:
    """Verify stored calls against current order-invariant decision semantics."""
    exact = {
        row["decision_id"]: row
        for row in manifest
        if row.get("classification") == "exact"
    }
    for response in _latest_responses(response_rows).values():
        decision = exact.get(response.get("decision_id"))
        result = object_or_empty(response.get("result"), "result")
        if decision is None or not result:
            continue
        stored_actions = Counter(
            action.get("action")
            for action in as_dicts(result.get("available_actions", []), "available_actions")
        )
        current_actions = Counter(
            action.get("action")
            for action in as_dicts(decision.get("available_actions", []), "available_actions")
        )
        if stored_actions != current_actions:
            raise RuntimeError(
                f"Stored response menu changed semantically for {response['decision_id']}"
            )
        human_index = response.get("human_action_index")
        stored_menu = as_list(result.get("available_actions", []), "available_actions")
        if not isinstance(human_index, int) or not 0 <= human_index < len(stored_menu):
            raise RuntimeError(
                f"Stored human action index is invalid for {response['decision_id']}"
            )
        stored_human = as_dict(stored_menu[human_index], "stored human action")
        if stored_human.get("action") != as_dict(decision["human"], "human")["action"]:
            raise RuntimeError(
                f"Stored human action changed semantically for {response['decision_id']}"
            )


def reasoning_request_for_model(
    model_id: str,
    effort: str = "xhigh",
) -> NativeReasoningRequest:
    """Return the explicit native-reasoning condition recorded for a model."""
    if not model_id:
        raise ValueError("model_id is required")
    return native_reasoning_request(effort)


def _response_cost_usd(row: JsonDict) -> float:
    result = row.get("result")
    usage = result.get("usage") if isinstance(result, dict) else None
    value = usage.get("cost") if isinstance(usage, dict) else None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return 0.0
    return float(value)

