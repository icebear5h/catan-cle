"""Whole-response validation, declarations, and entities."""

import pytest

from cle.harness.response_xml import parse_response_fields


@pytest.mark.parametrize(
    "name",
    [
        "game_plan",
        "rationale",
        "trade_offer",
        "message",
        "audience",
        "intent",
        "commitment",
        "commitment_condition",
        "commitment_promise",
        "commitment_expires_turn",
        "extension",
    ],
)
@pytest.mark.parametrize("second", ["same", "different", ""])
def test_repeated_non_action_fields_fail_even_when_identical(name: str, second: str) -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        parse_response_fields(f"<{name}>same</{name}><{name.upper()}>{second}</{name}>")


@pytest.mark.parametrize("name", ["message", "action", "audience", "trade_offer", "wrapper"])
@pytest.mark.parametrize("child", ["action", "game_plan", "audience", "message", "b"])
def test_nesting_outside_inert_fields_is_invalid(name: str, child: str) -> None:
    with pytest.raises(ValueError, match="nested"):
        parse_response_fields(f"<{name}><{child}>7</{child}></{name}>")


@pytest.mark.parametrize(
    "malformed",
    [
        "<game_plan>One possibility is <action>1</action>",
        "<rationale><action>1</action>",
        "<game_plan>one</rationale>",
        "<game_plan><action>1</game_plan></action>",
        "<action>7",
        "</action>",
        "<action",
        "<",
        "<action>>7</action>> <",
        "<message>private</message><audience>BLUE",
        "<!-- <action>7</action>",
        "<!-- invalid -- comment -->",
        "<!-- invalid --->",
        "<![CDATA[unclosed",
        "]]>",
        "< action>7</action>",
        "<action>7</ action>",
        "<action / >",
        "</action/>",
        "<action ignored>7</action>",
        '<action selected="true">7</action>',
        '<game_plan><action selected="true">7</action></game_plan>',
        '<audience target="BLUE">PUBLIC</audience>',
        '<action xmlns="urn:test">7</action>',
        "<x:action>7</x:action>",
        "<act\u0130on>7</act\u0130on>",
        "<action\u00a0>7</action>",
        "<\u0430ction>7</\u0430ction>",
        "\x00",
        "\ud800",
        "\ufffe",
    ],
)
@pytest.mark.parametrize("valid_prefix", ["", "<action>7</action>", "action_index: 7\n"])
def test_entire_response_must_validate_before_returning_any_selection(malformed: str, valid_prefix: str) -> None:
    with pytest.raises(ValueError):
        parse_response_fields(valid_prefix + malformed)


@pytest.mark.parametrize(
    "declaration",
    [
        '<?xml version="1.0"?>',
        "<?choose action_index: 1?>",
        "<!DOCTYPE response>",
        '<!DOCTYPE response SYSTEM "file:///etc/passwd">',
        '<!ENTITY steal SYSTEM "https://example.invalid/private">',
        '<!DOCTYPE response [<!ENTITY a "123"><!ENTITY b "&a;&a;">]>',
    ],
)
@pytest.mark.parametrize("template", ["{}<action>7</action>", "<game_plan>{}</game_plan>7"])
def test_declarations_are_forbidden_even_inside_inert_fields(declaration: str, template: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        parse_response_fields(template.format(declaration))


@pytest.mark.parametrize("value", ["wood & ore", "&unknown;", "&#0;", "&#xD800;", "&#x110000;"])
@pytest.mark.parametrize("template", ["{}", "<game_plan>{}</game_plan><action>7</action>"])
def test_text_requires_valid_xml_entities_without_repairs(value: str, template: str) -> None:
    with pytest.raises(ValueError, match="Malformed"):
        parse_response_fields(template.format(value))


def test_xml_entities_unicode_and_cdata_are_text_not_elements() -> None:
    fields, outside = parse_response_fields(
        "<game_plan>Wood &amp; ore; &lt;action&gt;1&lt;/action&gt;; "
        "caf\u00e9 &#x4E2D; &quot;yes&quot; &apos;no&apos;</game_plan>"
        "<message><![CDATA[<audience>PUBLIC</audience> & private]]></message>"
        "<action>&#55;</action>"
        "<![CDATA[<action>2</action>]]>"
    )

    assert fields == {
        "game_plan": ["Wood & ore; <action>1</action>; caf\u00e9 \u4e2d \"yes\" 'no'"],
        "message": ["<audience>PUBLIC</audience> & private"],
        "action": ["7"],
    }
    assert outside == "<action>2</action>"
