"""Policy selection validity and recorded-response lookup."""

from __future__ import annotations

from pathlib import Path

from evals.decision_spot_checks.shapes import dict_or_empty
from evals.inspect_archives.config import JsonDict
from evals.inspect_archives.support import _read_jsonl
from evals.json_types import JsonList, JsonValue, as_dict


def is_valid_policy_selection(policy: object, available_actions: object) -> bool:
    """Return whether one archived selection binds exactly to its legal menu."""

    if not isinstance(policy, dict) or not isinstance(available_actions, list):
        return False
    if policy.get("error"):
        return False
    action_index = policy.get("model_action_index")
    if isinstance(action_index, bool) or not isinstance(action_index, int):
        return False
    matches = [
        action
        for action in available_actions
        if isinstance(action, dict) and action.get("index") == action_index
    ]
    if len(matches) != 1:
        return False
    selected_action = policy.get("model_action")
    return selected_action is None or selected_action == matches[0].get("action")


def _policy_score_payload(
    decision: JsonDict,
    raw_response: JsonDict,
) -> JsonDict:
    result = dict_or_empty(raw_response.get("result"), "result")
    actions = _policy_available_actions(decision, raw_response)
    model_index = raw_response.get("model_action_index")
    human_index = raw_response.get("human_action_index")
    model_action = _action_at_index(actions, model_index)
    human_action = _action_at_index(actions, human_index)
    return {
        "agreement": raw_response.get("agreement"),
        "forced": bool(decision.get("forced")),
        "classification": decision.get("classification"),
        "model_action_index": model_index,
        "model_action": result.get("action") or (model_action or {}).get("action"),
        "model_description": result.get("action_description")
        or (model_action or {}).get("description"),
        "human_action_index": human_index,
        "human_action": (human_action or {}).get("action"),
        "human_description": (human_action or {}).get("description"),
        "parse_error": result.get("parse_error"),
        "error": raw_response.get("error"),
    }


def _policy_available_actions(
    decision: JsonDict,
    raw_response: JsonDict,
) -> JsonList:
    result = dict_or_empty(raw_response.get("result"), "result")
    actions = result.get("available_actions")
    if not isinstance(actions, list):
        raise ValueError(
            "original provider response is missing its exact legal menu for "
            f"{decision.get('decision_id')}"
        )
    return actions


def _action_at_index(actions: JsonList, value: JsonValue) -> JsonDict | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    for item in actions:
        action = as_dict(item, "legal menu action")
        if action.get("index") == value:
            return action
    return None


def _latest_policy_responses(path: Path, model_id: str) -> dict[str, JsonDict]:
    latest: dict[str, JsonDict] = {}
    for row in _read_jsonl(path):
        if row.get("model_id") != model_id:
            continue
        decision_id = row.get("decision_id")
        if isinstance(decision_id, str) and decision_id:
            latest[decision_id] = row
    return latest

