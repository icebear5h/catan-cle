"""Action index parsing strictness."""
from typing import Any

import pytest

from cle.harness import (
    ContextSuite,
    ModelResponse,
    PlayerResponseParseError,
    PlayerResponseParser,
    default_suite_path,
    load_context_suite,
)

from .support import _response, _sandbox_and_player


@pytest.mark.parametrize(
    "text, index, fallback",
    [
        ("<action>0</action>", 0, False),
        ("<AcTiOn>\n 7 \n</AcTiOn>", 7, False),
        ("<action>7</action><action>007</action>", 7, False),
        ("<action>7</action>action_index: 7", 7, False),
        ("7", 7, True),
        (" \n7\n ", 7, True),
        ("ACTION: 7", 7, True),
        ("action_index = 7", 7, True),
        ("action_index:\n7", 7, True),
        ("move: 7", 7, True),
        ("MOVE_INDEX=7", 7, True),
        ("action: 7\nmove_index: 007", 7, True),
        ("<game_plan>Need 2 roads</game_plan>action_index: 7", 7, True),
        ("<game_plan>Need 2 roads</game_plan>7", 7, True),
        ("<action>zero-based index from VALID ACTIONS</action>7", 7, True),
        (
            "<game_plan>Consider <action>1</action> or action_index: 2</game_plan>"
            "<action>7</action>",
            7,
            False,
        ),
    ],
)
def test_action_parser_preserves_exact_indices_and_explicit_fallbacks(text: str, index: int, fallback: bool) -> None:
    sandbox, _, _ = _sandbox_and_player([])
    context: Any = sandbox.decision_context()
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))

    choice: Any = PlayerResponseParser(suite).parse(
        context, ModelResponse(content=text)
    )

    assert choice.action_index == index
    assert (choice.parse_warning is not None) == fallback
    assert context.action_at(choice.action_index) == context.legal_actions[index]
    assert sandbox.game_engine.is_action_valid(context.action_at(choice.action_index))


@pytest.mark.parametrize("suite_name", ["catan_v10.yaml", "catan_v9.yaml", "catan_v4.yaml"])
@pytest.mark.parametrize("selection", ["<action>7</action>", "action_index: 7"])
def test_action_parser_ignores_the_suites_echoed_schema(suite_name: str, selection: str) -> None:
    sandbox, _, _ = _sandbox_and_player([])
    suite = load_context_suite(default_suite_path().with_name(suite_name))
    response: Any = ModelResponse(
        content=f"{suite.response.instruction}\n<game_plan>Need 2 roads</game_plan>\n{selection}"
    )

    choice: Any = PlayerResponseParser(suite).parse(sandbox.decision_context(), response)

    assert choice.action_index == 7
    assert choice.raw_response == response.content


@pytest.mark.parametrize("suite_name", ["catan_v4.yaml", "catan_v9.yaml"])
@pytest.mark.parametrize("selection", ["<action>7</action>", "action_index: 7", "7"])
@pytest.mark.parametrize(
    "rationale",
    [
        "<rationale>Best move: settle at node 7.</rationale>",
        "<RATIONALE><action>1</action>\nmove_index: 2</RATIONALE>",
    ],
)
def test_action_parser_treats_historical_rationale_as_inert_data(suite_name: str, selection: str, rationale: str) -> None:
    sandbox, _, _ = _sandbox_and_player([])
    context: Any = sandbox.decision_context()
    suite = load_context_suite(default_suite_path().with_name(suite_name))
    text: Any = f"<game_plan>Expand...</game_plan>{rationale}{selection}"

    choice: Any = PlayerResponseParser(suite).parse(context, ModelResponse(content=text))

    assert choice.action_index == 7
    assert context.action_at(choice.action_index) == context.legal_actions[7]
    assert choice.game_plan == "Expand..."
    assert choice.rationale == ""
    assert choice.native_reasoning == ""
    assert choice.raw_response == text


@pytest.mark.parametrize(
    "instruction",
    [
        "<game_plan>plan</game_plan><action>0</action>",
        "Return a zero-based action index.",
    ],
)
def test_action_parser_keeps_real_selections_when_schema_has_no_placeholder(instruction: str) -> None:
    sandbox, _, _ = _sandbox_and_player([])
    suite_data = load_context_suite(
        default_suite_path().with_name("catan_v10.yaml")
    ).model_dump(mode="python")
    suite_data["response"]["instruction"] = instruction
    parser: Any = PlayerResponseParser(ContextSuite.model_validate(suite_data))
    context = sandbox.decision_context()

    assert parser.parse(context, _response(action=0)).action_index == 0
    with pytest.raises(PlayerResponseParseError, match="conflicting"):
        parser.parse(context, ModelResponse(content="<action>0</action><action>1</action>"))
    with pytest.raises(PlayerResponseParseError, match="whole non-negative integers"):
        parser.parse(context, ModelResponse(content="<action></action>action_index: 7"))


@pytest.mark.parametrize("template", ["<action>{}</action>", "action_index: {}", "{}"])
@pytest.mark.parametrize(
    "value",
    ["-1", "+1", "1.9", "0/1", "1e1", "1 2", "1,2", "0x1", "", "999"],
)
def test_action_parser_rejects_non_integer_or_out_of_range_indices(template: str, value: str) -> None:
    sandbox, _, _ = _sandbox_and_player([])
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))

    with pytest.raises(PlayerResponseParseError):
        PlayerResponseParser(suite).parse(
            sandbox.decision_context(), ModelResponse(content=template.format(value))
        )


@pytest.mark.parametrize(
    "text",
    [
        "<game_plan>Need 2 roads</game_plan>",
        "<game_plan>action_index: 7</game_plan>",
        "<game_plan><action>7</action></game_plan>",
        "<rationale><action>7</action></rationale>",
        "<rationale>move_index: 7</rationale>",
        '<trade_offer>{"give":{"WOOD":1},"receive":{"ORE":2}}</trade_offer>',
        '<trade_offer>{"note":"action_index: 7"}</trade_offer>',
        "I need 2 roads and will decide later.",
        "I need 2 roads.\n7",
        "<game_plan>Need 2 roads</game_plan><action>not sure</action>",
        "<action>-1</action><action>2</action>",
    ],
)
def test_action_parser_does_not_infer_an_index_from_plan_trade_or_prose(text: str) -> None:
    sandbox, _, _ = _sandbox_and_player([])
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))

    with pytest.raises(PlayerResponseParseError):
        PlayerResponseParser(suite).parse(
            sandbox.decision_context(), ModelResponse(content=text)
        )


@pytest.mark.parametrize(
    "text",
    [
        "<action>1</action><action>2</action>",
        "action: 1\nmove_index: 2",
        "<action>1</action>action_index: 2",
        "<action>1</action>2",
    ],
)
def test_action_parser_rejects_conflicting_selections(text: str) -> None:
    sandbox, _, _ = _sandbox_and_player([])
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))

    with pytest.raises(PlayerResponseParseError, match="conflicting"):
        PlayerResponseParser(suite).parse(
            sandbox.decision_context(), ModelResponse(content=text)
        )
