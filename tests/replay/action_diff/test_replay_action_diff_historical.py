"""Historical indexed XML comparison behavior is unchanged."""

from copy import deepcopy

import pytest

from evals.replay_action_diff import (
    normalize_response_selection,
)


@pytest.mark.parametrize("schema_version, context_version", [
    ("replay-action-diff-v1", "replay-decision-v2"),
    ("replay-action-diff-v2", "catan-agent@4.0.0"),
    ("replay-action-diff-v2", "catan-agent@5.0.0"),
    ("replay-action-diff-v2", "catan-agent@6.2.1"),
    ("replay-action-diff-v2", "catan-agent@7.0.0"),
    ("replay-action-diff-v2", "catan-agent@8.0.0"),
    ("replay-action-diff-v2", "catan-agent@9.0.0"),
    ("replay-action-diff-v2", "catan-agent@10.0.0"),
])
@pytest.mark.parametrize("raw_response, expected, warning", [
    ("<goals>Build roads</goals><reasoning>Action 1</reasoning><action>0</action>", 0, None),
    ("<action>index</action><action>1</action>", 1, None),
    ("<action>0</action><action>1</action>", 1, "conflicting numeric action tags"),
    ("action_index: 1", 1, None),
    ("move=1", 1, None),
    ("<action>99</action>", None, "outside the valid range"),
    ("No selection", None, "parseable action index"),
])
def test_historical_indexed_xml_comparison_behavior_is_unchanged(
    schema_version: str, context_version: str, raw_response: str, expected: int | None, warning: str | None
) -> None:
    row = {
        "schema_version": schema_version,
        "human_action_index": 1, "model_action_index": 0, "agreement": False,
        "result": {
            "context_version": context_version, "raw_response": raw_response,
            "action_index": 0, "action": "old", "parse_error": "old parser error",
            "available_actions": [
                {"index": 0, "action": "zero", "description": "Zero"},
                {"index": 1, "action": "one", "description": "One"},
            ],
        },
    }
    original = deepcopy(row)
    normalized = normalize_response_selection(row)
    assert row == original
    assert normalized["schema_version"] == row["schema_version"]
    assert normalized["model_action_index"] == expected
    assert normalized["agreement"] is (expected == 1)
    assert normalized["result"]["raw_response"] == raw_response
    if warning:
        assert warning in normalized["result"]["parse_error"]
    else:
        assert normalized["result"]["parse_error"] is None
