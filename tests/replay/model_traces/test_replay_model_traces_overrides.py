"""Curated setup overrides must match an exact attempt ledger row."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from playground.game_viewer.replay.model_traces import (
    ModelTraceArtifactError,
)

from .support import CURATED_TRACE_DIR, _load_curated_override, _write_jsonl


def test_setup_override_must_be_an_exact_attempt_ledger_row(tmp_path: Path) -> None:
    attempts = [
        json.loads(line)
        for line in (CURATED_TRACE_DIR / "setup_strategy_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    original_override = json.loads(
        (CURATED_TRACE_DIR / "setup_strategy_overrides.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    relabeled = deepcopy(attempts[1])
    relabeled["decision_id"] = original_override["decision_id"]
    override_path = tmp_path / "overrides.jsonl"
    attempts_path = tmp_path / "attempts.jsonl"
    _write_jsonl(override_path, [relabeled])
    _write_jsonl(attempts_path, attempts)

    with pytest.raises(ModelTraceArtifactError, match="not in the attempts ledger"):
        _load_curated_override(override_path, attempts_path)


def test_setup_override_rejects_mismatched_causal_and_menu_metadata(tmp_path: Path) -> None:
    original = json.loads(
        (CURATED_TRACE_DIR / "setup_strategy_overrides.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    cases = []

    wrong_result_row = deepcopy(original)
    wrong_result_row["result"]["replay_index"] = 14
    cases.append((wrong_result_row, "result replay row"))

    wrong_stage = deepcopy(original)
    wrong_stage["result"]["setup_stage"] = "second_settlement"
    cases.append((wrong_stage, "override stage"))

    future_activity = deepcopy(original)
    future_activity["result"]["activity_window"]["end_replay_index"] = 1
    cases.append((future_activity, "invalid activity cutoff"))

    changed_menu = deepcopy(original)
    changed_menu["result"]["available_actions"][0]["action"] = "tampered"
    cases.append((changed_menu, "menu order or identity changed"))

    reordered_menu = deepcopy(original)
    actions = reordered_menu["result"]["available_actions"]
    actions[0], actions[1] = actions[1], actions[0]
    actions[0]["index"] = 0
    actions[1]["index"] = 1
    cases.append((reordered_menu, "menu order or identity changed"))

    mismatched_selected_index = deepcopy(original)
    mismatched_selected_index["model_action_index"] = 0
    cases.append((mismatched_selected_index, "top-level selected index"))

    later_override = json.loads(
        (CURATED_TRACE_DIR / "setup_strategy_overrides.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[1]
    )
    substituted_activity = deepcopy(later_override)
    substituted_activity["result"]["recent_activity"][0] = "future activity"
    cases.append((substituted_activity, "recent activity"))

    for case_index, (row, message) in enumerate(cases):
        override_path = tmp_path / f"overrides-{case_index}.jsonl"
        attempts_path = tmp_path / f"attempts-{case_index}.jsonl"
        _write_jsonl(override_path, [row])
        _write_jsonl(attempts_path, [row])
        with pytest.raises(ModelTraceArtifactError, match=message):
            _load_curated_override(override_path, attempts_path)
