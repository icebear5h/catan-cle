import pytest

from cle.harness.response_xml import parse_response_fields
from cle.harness.suite import default_suite_path, load_context_suite


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
def test_returns_top_level_fields_and_only_outside_text(text, expected):
    assert parse_response_fields(text) == expected


@pytest.mark.parametrize("value", ["-1", "+1", "1.5", "1e1", "0/1", "not sure", "\uff17", "\u0667"])
def test_action_value_validation_belongs_to_the_caller(value):
    assert parse_response_fields(f"<action>{value}</action><action>7</action>") == (
        {"action": [value, "7"]},
        "",
    )


@pytest.mark.parametrize("name", ["game_plan", "rationale"])
def test_nested_plan_controls_are_inert_text(name):
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


def test_trade_and_communication_text_are_not_index_fallbacks():
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
def test_comments_never_supply_fields_or_fallbacks(text, expected):
    assert parse_response_fields(text) == expected


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
def test_repeated_non_action_fields_fail_even_when_identical(name, second):
    with pytest.raises(ValueError, match="Duplicate"):
        parse_response_fields(f"<{name}>same</{name}><{name.upper()}>{second}</{name}>")


@pytest.mark.parametrize("name", ["message", "action", "audience", "trade_offer", "wrapper"])
@pytest.mark.parametrize("child", ["action", "game_plan", "audience", "message", "b"])
def test_nesting_outside_inert_fields_is_invalid(name, child):
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
def test_entire_response_must_validate_before_returning_any_selection(malformed, valid_prefix):
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
def test_declarations_are_forbidden_even_inside_inert_fields(declaration, template):
    with pytest.raises(ValueError, match="forbidden"):
        parse_response_fields(template.format(declaration))


@pytest.mark.parametrize("value", ["wood & ore", "&unknown;", "&#0;", "&#xD800;", "&#x110000;"])
@pytest.mark.parametrize("template", ["{}", "<game_plan>{}</game_plan><action>7</action>"])
def test_text_requires_valid_xml_entities_without_repairs(value, template):
    with pytest.raises(ValueError, match="Malformed"):
        parse_response_fields(template.format(value))


def test_xml_entities_unicode_and_cdata_are_text_not_elements():
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


@pytest.mark.parametrize("suite_name", ["catan_v4.yaml", "catan_v9.yaml"])
@pytest.mark.parametrize("selection", ["<action>7</action>", "action_index: 7", "7"])
@pytest.mark.parametrize("leading", ["", " \n\t"])
def test_exact_trusted_suite_instruction_echo_is_removed(suite_name, selection, leading):
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
def test_echo_normalization_does_not_repair_other_malformed_markup(suite_name, alteration):
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


def test_echo_cannot_hide_malformed_tail_or_unclosed_plan():
    instruction = load_context_suite(
        default_suite_path().with_name("catan_v9.yaml")
    ).response.instruction
    with pytest.raises(ValueError):
        parse_response_fields(
            instruction + "\n<game_plan><action>7</action>", instruction=instruction
        )


def test_schema_placeholders_and_numeric_examples_are_not_filtered_by_helper():
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


def test_echo_text_inside_comments_and_plans_is_not_normalized():
    instruction = "Return <action>index</action>"
    fields, outside = parse_response_fields(
        f"<!-- {instruction} --><game_plan>{instruction}</game_plan><action>7</action>",
        instruction=instruction,
    )

    assert fields == {"game_plan": [instruction], "action": ["7"]}
    assert outside == ""


def test_response_length_is_bounded_before_echo_normalization():
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


def test_nesting_depth_is_bounded_including_inert_elements():
    text = "<game_plan>" + "<b>" * 31 + "text" + "</b>" * 31 + "</game_plan>"
    fields, outside = parse_response_fields(text + "<action>7</action>")
    assert fields["action"] == ["7"]
    assert outside == ""
    with pytest.raises(ValueError, match="depth"):
        parse_response_fields("<game_plan>" + text + "</game_plan><action>7</action>")
