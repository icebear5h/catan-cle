"""Suite loading, component validation, and ordering policy."""
import json
from typing import Any

import pytest

from cle.harness import (
    ContextSuite,
    default_suite_path,
    load_context_suite,
)
from cle.harness.response_xml import parse_response_fields

from .support import allows_deprecated_suite


@allows_deprecated_suite
def test_default_yaml_suite_is_strict_and_separately_loadable() -> None:
    first: Any = load_context_suite()
    second = load_context_suite()

    assert first == second
    assert first is not second
    assert first.id == "catan-agent"
    assert first.version == "11.0.0"
    assert first.response.format == "json"
    assert first.response.tags == ("game_plan", "tool", "arguments")
    assert "action_mode" not in first.response.model_dump()
    assert first.context.mode == "components"
    assert first.context.social_context is True
    assert first.context.initial_placement_order == "both_rounds"
    assert first.context.order == (
        "trajectory",
        "strategic_memory",
        "visible_events",
        "recent_table_talk",
        "commitments",
        "phase_info",
        "board_state",
        "resources",
        "opponents",
        "trade_window",
        "phase_guidance",
        "legal_actions",
        "decision_request",
        "response_schema",
    )
    assert first.context.trajectory.max_messages is None
    assert "<action>" not in first.response.instruction
    assert "<game_plan>" not in first.response.instruction
    assert "action_index" not in first.response.instruction
    for name in first.response.tags:
        assert f'"{name}"' in first.response.instruction
    for token in ("<N00>", "<E00_01>", "<T00>"):
        assert token in first.response.instruction

    guidance_word_counts = {
        key: len(value.split())
        for key, value in first.phase_guidance.items()
    }
    assert set(guidance_word_counts) == {
        "initial_settlement_1",
        "initial_settlement_2",
        "initial_road_1",
        "initial_road_2",
        "initial_placement",
        "discarding",
        "robber",
        "main_game",
    }
    assert guidance_word_counts["initial_settlement_1"] == 213
    assert all(
        10 <= count <= 300
        for key, count in guidance_word_counts.items()
        if key != "initial_settlement_1"
    )
    assert 600 <= sum(guidance_word_counts.values()) <= 950

    strategy_variant = json.loads(
        default_suite_path()
        .parents[3]
        .joinpath(
            "evals",
            "suites",
            "catan_initial_settlement_strategy_v2.json",
        )
        .read_text(encoding="utf-8")
    )
    assert first.phase_guidance["initial_settlement_1"] == (
        strategy_variant["guidance"]
    )

    historical_v10 = load_context_suite(
        default_suite_path().with_name("catan_v10.yaml")
    )
    assert historical_v10.response.format == "xml"
    response_fields, _ = parse_response_fields(historical_v10.response.instruction)
    assert set(response_fields) == set(historical_v10.response.tags)
    assert all(len(values) == 1 for values in response_fields.values())
    assert first.system == historical_v10.system
    assert first.context == historical_v10.context
    unchanged_guidance = (
        "initial_settlement_2",
        "initial_road_1",
        "initial_road_2",
        "initial_placement",
    )
    assert all(
        first.phase_guidance[key] == historical_v10.phase_guidance[key]
        for key in unchanged_guidance
    )
    response_changes = {"format", "tags", "instruction"}
    assert first.response.model_dump(exclude=response_changes) == (
        historical_v10.response.model_dump(exclude=response_changes)
    )

    historical_v8 = load_context_suite(
        default_suite_path().with_name("catan_v8.yaml")
    )
    assert historical_v8.version == "8.0.0"
    assert historical_v8.context.initial_placement_order == "both_rounds"
    assert historical_v8.phase_guidance["initial_settlement_1"] != (
        strategy_variant["guidance"]
    )
    assert "Maximizing raw pip count is not the objective" not in (
        historical_v8.phase_guidance["initial_settlement_1"]
    )

    historical_v7 = load_context_suite(
        default_suite_path().with_name("catan_v7.yaml")
    )
    v8_without_order = historical_v8.model_dump(mode="python")
    v8_without_order["version"] = "7.0.0"
    v8_without_order["context"]["initial_placement_order"] = "omit"
    assert ContextSuite.model_validate(v8_without_order) == historical_v7

    guidance_text = "\n".join(first.phase_guidance.values()).lower()
    for implementation_phrase in (
        "game engine",
        "implementation",
        "legal menu",
        "live menu",
        "prompt",
        "rationale",
        "random stream",
        "runtime",
        "seeded",
    ):
        assert implementation_phrase not in guidance_text

    stable_contract = "\n".join(
        (first.system.template, first.response.instruction)
    ).lower()
    assert "rationale" not in stable_contract
    for prescriptive_phrase in (
        "expert",
        "prioritize",
        "prefer",
        "think in sequences",
        "cities > settlements",
        "block the leading opponent",
    ):
        assert prescriptive_phrase not in stable_contract


@allows_deprecated_suite
def test_legacy_decision_suite_remains_loadable() -> None:
    suite = load_context_suite(default_suite_path().with_name("catan_v5.yaml"))

    assert suite.version == "5.0.0"
    assert suite.response.format == "xml"
    assert suite.context.mode == "legacy"
    assert suite.sections["observation"].template is None


def test_component_suite_rejects_unknown_missing_duplicate_and_oversized_strings() -> None:
    suite = load_context_suite()

    unknown = suite.model_dump(mode="python")
    unknown["sections"]["board_state"]["template"] = "{{ hidden_hand }}"
    with pytest.raises(ValueError, match="unknown variables"):
        ContextSuite.model_validate(unknown)

    missing = suite.model_dump(mode="python")
    del missing["sections"]["resources"]
    with pytest.raises(ValueError, match="unknown sections"):
        ContextSuite.model_validate(missing)

    duplicate = suite.model_dump(mode="python")
    duplicate["context"]["order"] = (
        *duplicate["context"]["order"],
        "resources",
    )
    with pytest.raises(ValueError, match="duplicate sections"):
        ContextSuite.model_validate(duplicate)

    oversized = suite.model_dump(mode="python")
    oversized["sections"]["resources"]["template"] = "x" * 12_001
    with pytest.raises(ValueError, match="at most 12000 characters"):
        ContextSuite.model_validate(oversized)


@allows_deprecated_suite
@pytest.mark.parametrize("suite_name", ["catan_v9.yaml", "catan_v10.yaml", "catan_v11.yaml"])
def test_component_order_requires_explicit_social_policy(suite_name: str) -> None:
    suite = load_context_suite(default_suite_path().with_name(suite_name))
    data = suite.model_dump(mode="python")
    data["context"]["social_context"] = not suite.context.social_context
    with pytest.raises(ValueError, match="fixed component order"):
        ContextSuite.model_validate(data)
