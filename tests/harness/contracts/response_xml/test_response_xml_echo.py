"""Trusted suite echo normalization and bounds."""

import pytest

from cle.harness.response_xml import parse_response_fields
from cle.harness.suite import default_suite_path, load_context_suite


@pytest.mark.parametrize("suite_name", ["catan_v4.yaml", "catan_v9.yaml"])
@pytest.mark.parametrize("selection", ["<action>7</action>", "action_index: 7", "7"])
@pytest.mark.parametrize("leading", ["", " \n\t"])
def test_exact_trusted_suite_instruction_echo_is_removed(suite_name: str, selection: str, leading: str) -> None:
    instruction = load_context_suite(
        default_suite_path().with_name(suite_name)
    ).response.instruction
    fields, outside = parse_response_fields(
        f"{leading}{instruction}\n<game_plan>Need 2 roads</game_plan>\n{selection}",
        instruction=instruction,
    )

    assert fields["game_plan"] == ["Need 2 roads"]
    assert "rationale" not in fields
    assert "trade_offer" not in fields
    if selection.startswith("<"):
        assert fields["action"] == ["7"]
        assert outside == ""
    else:
        assert "action" not in fields
        assert outside == selection


@pytest.mark.parametrize("suite_name", ["catan_v4.yaml", "catan_v9.yaml"])
@pytest.mark.parametrize("alteration", ["changed", "prefixed", "repeated", "untrusted"])
def test_echo_normalization_does_not_repair_other_malformed_markup(suite_name: str, alteration: str) -> None:
    instruction = load_context_suite(
        default_suite_path().with_name(suite_name)
    ).response.instruction
    text = instruction + "\n<action>7</action>"
    if alteration == "changed":
        text = text.replace("<trade_offer>", "<trade_offer >", 1)
    elif alteration == "prefixed":
        text = "Here is the schema:\n" + text
    elif alteration == "repeated":
        text = instruction + "\n" + text
    else:
        instruction = ""

    with pytest.raises(ValueError):
        parse_response_fields(text, instruction=instruction)


def test_echo_cannot_hide_malformed_tail_or_unclosed_plan() -> None:
    instruction = load_context_suite(
        default_suite_path().with_name("catan_v9.yaml")
    ).response.instruction
    with pytest.raises(ValueError):
        parse_response_fields(
            instruction + "\n<game_plan><action>7</action>", instruction=instruction
        )


def test_schema_placeholders_and_numeric_examples_are_not_filtered_by_helper() -> None:
    instruction = "<game_plan>plan</game_plan><action>0</action>"
    assert parse_response_fields(instruction + "<action>1</action>", instruction=instruction) == (
        {"game_plan": ["plan"], "action": ["0", "1"]},
        "",
    )
    assert parse_response_fields(
        "<action>zero-based index from VALID ACTIONS</action>7",
        instruction="Return a zero-based action index.",
    ) == ({"action": ["zero-based index from VALID ACTIONS"]}, "7")
    assert parse_response_fields("<action></action>action_index: 7") == (
        {"action": [""]},
        "action_index: 7",
    )


def test_echo_text_inside_comments_and_plans_is_not_normalized() -> None:
    instruction = "Return <action>index</action>"
    fields, outside = parse_response_fields(
        f"<!-- {instruction} --><game_plan>{instruction}</game_plan><action>7</action>",
        instruction=instruction,
    )

    assert fields == {"game_plan": [instruction], "action": ["7"]}
    assert outside == ""


def test_response_length_is_bounded_before_echo_normalization() -> None:
    value = "x" * (128 * 1024 - len("<game_plan></game_plan>"))
    assert parse_response_fields(f"<game_plan>{value}</game_plan>") == (
        {"game_plan": [value]},
        "",
    )
    with pytest.raises(ValueError, match="characters"):
        parse_response_fields(f"<game_plan>{value}x</game_plan>")
    instruction = "<action>index</action>" + " " * (128 * 1024)
    with pytest.raises(ValueError, match="characters"):
        parse_response_fields(instruction + "7", instruction=instruction)


def test_nesting_depth_is_bounded_including_inert_elements() -> None:
    text = "<game_plan>" + "<b>" * 31 + "text" + "</b>" * 31 + "</game_plan>"
    fields, outside = parse_response_fields(text + "<action>7</action>")
    assert fields["action"] == ["7"]
    assert outside == ""
    with pytest.raises(ValueError, match="depth"):
        parse_response_fields("<game_plan>" + text + "</game_plan><action>7</action>")
