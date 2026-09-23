"""Top-level fields, inert nesting, and fallback refusal."""

import pytest

from cle.harness.response_xml import parse_response_fields


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", ({}, "")),
        (" \n7\n ", ({}, "7")),
        ("ACTION: 7", ({}, "ACTION: 7")),
        ("action_index:\n7", ({}, "action_index:\n7")),
        ("move_index = 007", ({}, "move_index = 007")),
        ("I need 2 roads.", ({}, "I need 2 roads.")),
        ("<action>7</action>", ({"action": ["7"]}, "")),
        ("<AcTiOn \n> 7 </ACTION\t>", ({"action": ["7"]}, "")),
        ("<action />7", ({"action": [""]}, "7")),
        ("<audience/>", ({"audience": [""]}, "")),
        ("<extension>data</extension>", ({"extension": ["data"]}, "")),
        (
            "<action>7</action><ACTION >007</action>action_index: 7",
            ({"action": ["7", "007"]}, "action_index: 7"),
        ),
        (
            "<action>1</action><action >2</action>",
            ({"action": ["1", "2"]}, ""),
        ),
        (
            "<game_plan>Need 2 roads</game_plan>7",
            ({"game_plan": ["Need 2 roads"]}, "7"),
        ),
        (
            "<game_plan>Save {{ wood }}</game_plan>move: 7",
            ({"game_plan": ["Save {{ wood }}"]}, "move: 7"),
        ),
    ],
)
def test_returns_top_level_fields_and_only_outside_text(
    text: str, expected: tuple[dict[str, list[str]], str]
) -> None:
    assert parse_response_fields(text) == expected


@pytest.mark.parametrize("value", ["-1", "+1", "1.5", "1e1", "0/1", "not sure", "\uff17", "\u0667"])
def test_action_value_validation_belongs_to_the_caller(value: str) -> None:
    assert parse_response_fields(f"<action>{value}</action><action>7</action>") == (
        {"action": [value, "7"]},
        "",
    )


@pytest.mark.parametrize("name", ["game_plan", "rationale"])
def test_nested_plan_controls_are_inert_text(name: str) -> None:
    plan = (
        "Consider <ACTION>1</ACTION> or action_index: 2. "
        "<message>private</message><audience>PUBLIC</audience>"
        "<game_plan><action>3</action></game_plan>"
    )
    fields, outside = parse_response_fields(
        f"<{name.upper()}>{plan}</{name}><action>7</action>move_index: 7"
    )

    assert fields == {name: [plan.replace("ACTION", "action")], "action": ["7"]}
    assert outside == "move_index: 7"


def test_trade_and_communication_text_are_not_index_fallbacks() -> None:
    fields, outside = parse_response_fields(
        '<trade_offer>{"note":"action_index: 1"}</trade_offer>'
        "<message>move: 2</message><intent>TRADE</intent>"
        "<audience>BLUE</audience>"
        "<commitment_condition>action_index: 3</commitment_condition>"
        "<commitment_promise>move: 4</commitment_promise>"
        "<commitment_expires_turn>7</commitment_expires_turn>"
    )

    assert fields["message"] == ["move: 2"]
    assert fields["commitment_expires_turn"] == ["7"]
    assert outside == ""


@pytest.mark.parametrize(
    "text, expected",
    [
        ("<!-- <action>1</action> action_index: 2 -->", ({}, "")),
        ("<!-- <game_plan>unclosed & inert -->7", ({}, "7")),
        (
            "<!-- <message>PRIVATE DRAFT</message> -->"
            "<message>SILENCE</message><audience>PUBLIC</audience><intent>TRADE</intent>",
            ({"message": ["SILENCE"], "audience": ["PUBLIC"], "intent": ["TRADE"]}, ""),
        ),
        (
            "<game_plan>A<!-- <action>1</action> -->B</game_plan><action>7</action>",
            ({"game_plan": ["A\nB"], "action": ["7"]}, ""),
        ),
        ("1<!-- do not join -->2", ({}, "1\n2")),
        ("action<!-- do not join -->_index: 7", ({}, "action\n_index: 7")),
        ("<action>1<!-- do not join -->2</action>", ({"action": ["1\n2"]}, "")),
        ("1<game_plan>inert</game_plan>2", ({"game_plan": ["inert"]}, "1\n2")),
    ],
)
def test_comments_never_supply_fields_or_fallbacks(
    text: str, expected: tuple[dict[str, list[str]], str]
) -> None:
    assert parse_response_fields(text) == expected
