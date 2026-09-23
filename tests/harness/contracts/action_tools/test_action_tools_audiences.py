"""Concrete trade descriptors and audience binding."""
import json
from copy import deepcopy
from dataclasses import replace
from itertools import permutations
from typing import Any

import pytest

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.harness.action_tools import parse_tool_choice, render_action_tools
from cle.players.validation import action_from_choice

from .support import COLORS, OPAQUE_ID, ORE, WOOD, _case, _context, _offer, _turn_engine


def test_concrete_trade_menu_terms_and_audience_are_never_overwritten() -> None:
    context, arguments, _ = _case("offer_trade")
    concrete = _offer(audience=(Color.BLUE,))
    context = replace(context, legal_actions=(Action(Color.RED, ActionType.OFFER_TRADE, concrete),))
    choice = parse_tool_choice(context, "offer_trade", arguments)
    assert choice.trade_offer is None
    assert action_from_choice(context, choice).value == concrete
    for changed in ({**arguments, "receive_any": 1}, {**arguments, "give": {"BRICK": 1}}):
        with pytest.raises(ValueError, match="No legal"):
            parse_tool_choice(context, "offer_trade", changed)
    # Same resource terms with two different audiences cannot be guessed from the call.
    ambiguous: Any = replace(
        context,
        legal_actions=(
            *context.legal_actions,
            Action(Color.RED, ActionType.OFFER_TRADE, _offer(audience=(Color.WHITE,))),
        ),
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        parse_tool_choice(ambiguous, "offer_trade", arguments)


@pytest.mark.parametrize("parameterized", [False, True])
def test_concrete_root_alternatives_render_and_resolve_identical_terms_with_distinct_audiences(
    parameterized: bool,
) -> None:
    engine: Any = _turn_engine()
    actions: Any = tuple(
        Action(Color.RED, ActionType.OFFER_TRADE, _offer(audience=(color,)))
        for color in (Color.BLUE, Color.WHITE)
    )
    assert all(engine.is_action_valid(action) for action in actions)
    context: Any = _context(engine, actions=actions)
    if parameterized:
        meta: Any = next(
            a for a in engine.state.playable_actions if a.action_type == ActionType.OFFER_TRADE
        )
        context = replace(context, legal_actions=(*actions, meta))
    assert context.observation.trade_window is None
    expected: Any = [
        {
            "give": {"WOOD": 1},
            "receive": {"ORE": 1},
            "give_any": 0,
            "receive_any": 0,
            "audience": [color.value],
        }
        for color in (Color.BLUE, Color.WHITE)
    ]
    rendered: Any = render_action_tools(context)
    for menu in permutations(context.legal_actions):
        permuted: Any = replace(context, legal_actions=menu)
        assert render_action_tools(permuted) == rendered
        line: Any = rendered.splitlines()[-1]
        descriptors: Any = json.loads(line.split("exact alternatives: ", 1)[1])
        assert descriptors == expected
        assert "action_index" not in rendered
        assert ("fixed to other seats" in rendered) == parameterized
        for descriptor in descriptors:
            choice: Any = parse_tool_choice(permuted, "offer_trade", descriptor)
            action: Any = action_from_choice(permuted, choice)
            assert choice.trade_offer is None
            assert action in actions
            assert action.value.audience == frozenset({Color(descriptor["audience"][0])})
            branch: Any = deepcopy(engine)
            branch.step(action)
            submitted: Any = branch.state.trade_window.active_offers[0]
            assert submitted.audience == action.value.audience
            assert submitted.give == action.value.give
            assert submitted.receive == action.value.receive
        arguments: Any = {"give": {"WOOD": 1}, "receive": {"ORE": 1}}
        if parameterized:
            assert (
                "without audience, matching parameterized offers take precedence over concrete alternatives"
                in rendered
            )
            assert "required if ambiguous or a matching parameterized offer exists" in rendered
            choice = parse_tool_choice(permuted, "offer_trade", arguments)
            assert permuted.action_at(choice.action_index) == meta
            assert choice.trade_offer == _offer()
            action = action_from_choice(permuted, choice)
            assert engine.is_action_valid(action)
            branch = deepcopy(engine)
            branch.step(action)
            submitted = branch.state.trade_window.active_offers[0]
            assert submitted.audience == frozenset(COLORS[1:])
            assert submitted.give == WOOD
            assert submitted.receive == ORE
        else:
            with pytest.raises(ValueError, match="Ambiguous"):
                parse_tool_choice(permuted, "offer_trade", arguments)


@pytest.mark.parametrize("tool", ["offer_trade", "counter_offer"])
def test_concrete_descriptors_preserve_wildcards_parent_and_round_trip_through_strict_engine(tool: str) -> None:
    engine: Any = _turn_engine()
    parent = None
    actor = Color.RED
    audience = (Color.BLUE, Color.WHITE)
    if tool == "counter_offer":
        context = _context(engine)
        root = parse_tool_choice(
            context, "offer_trade", {"give": {"WOOD": 1}, "receive": {"ORE": 1}}
        )
        engine.step(action_from_choice(context, root))
        parent = engine.state.trade_window.active_offers[0].id
        actor, audience = Color.BLUE, (Color.RED,)
    concrete = _offer(
        actor=actor,
        audience=audience,
        give=ORE,
        receive=WOOD,
        give_any=1,
        receive_any=2,
        parent=parent,
    )
    action_type = ActionType.COUNTER_OFFER if parent else ActionType.OFFER_TRADE
    action = Action(actor, action_type, concrete)
    assert engine.is_action_valid(action)
    context = _context(engine, actor=actor, actions=(action,))
    line = render_action_tools(context).splitlines()[-1]
    descriptors = json.loads(line.split("exact alternatives: ", 1)[1])
    expected: Any = {
        "give": {"ORE": 1},
        "receive": {"WOOD": 1},
        "give_any": 1,
        "receive_any": 2,
        "audience": [color.value for color in audience],
    }
    if parent:
        expected["offer_id"] = parent
    assert descriptors == [expected]
    # Audience is an unordered set of color names, not a player alias or lifecycle override.
    arguments = {
        **expected,
        "audience": [color.lower() for color in reversed(expected["audience"])],
    }
    choice = parse_tool_choice(context, tool, arguments)
    assert choice.trade_offer is None
    assert action_from_choice(context, choice) == action
    engine.step(action_from_choice(context, choice))
    submitted = engine.state.trade_window.active_offers[-1]
    assert submitted.audience == concrete.audience
    assert submitted.give_any == 1
    assert submitted.receive_any == 2
    assert submitted.parent_offer_id == parent


@pytest.mark.parametrize("tool", ["offer_trade", "counter_offer"])
@pytest.mark.parametrize(
    "audience",
    [
        None,
        "BLUE",
        [],
        {},
        0,
        True,
        [None],
        [1],
        [["BLUE"]],
        ["GOLD"],
        ["MYSTIC_BLUE"],
        ["BLUE", "blue"],
        ["WHITE"],
        ["RED", "BLUE", "WHITE", "ORANGE"],
    ],
)
def test_concrete_audience_requires_exact_unique_other_player_colors(tool: str, audience: object) -> None:
    context, arguments, _ = _case(tool)
    concrete = _offer(
        actor=context.actor,
        audience=(Color.RED,) if tool == "counter_offer" else (Color.BLUE,),
        give=(0, 0, 0, 0, 2) if tool == "counter_offer" else WOOD,
        receive=WOOD if tool == "counter_offer" else ORE,
        parent=OPAQUE_ID if tool == "counter_offer" else None,
    )
    action_type = ActionType.COUNTER_OFFER if tool == "counter_offer" else ActionType.OFFER_TRADE
    context: Any = replace(context, legal_actions=(Action(context.actor, action_type, concrete),))
    with pytest.raises(ValueError):
        parse_tool_choice(context, tool, {**arguments, "audience": audience})


@pytest.mark.parametrize("tool", ["offer_trade", "counter_offer"])
@pytest.mark.parametrize("mixed", [False, True])
def test_audience_never_overrides_or_falls_back_to_parameterized_offer_rules(tool: str, mixed: bool) -> None:
    context, arguments, _ = _case(tool)
    expected: Any = frozenset({Color.RED}) if tool == "counter_offer" else frozenset(COLORS[1:])
    if mixed:
        concrete = _offer(
            actor=context.actor,
            audience=tuple(expected),
            give=(0, 1, 0, 0, 0),
            receive=ORE,
            parent=OPAQUE_ID if tool == "counter_offer" else None,
        )
        kind = ActionType.COUNTER_OFFER if tool == "counter_offer" else ActionType.OFFER_TRADE
        context: Any = replace(
            context, legal_actions=(*context.legal_actions, Action(context.actor, kind, concrete))
        )
    for audience in (["WHITE"], [color.value for color in expected]):
        with pytest.raises(ValueError, match="No legal"):
            parse_tool_choice(context, tool, {**arguments, "audience": audience})
    choice: Any = parse_tool_choice(context, tool, arguments)
    assert choice.trade_offer.audience == expected
    assert "(omit audience)" in render_action_tools(context)


@pytest.mark.parametrize("tool", ["offer_trade", "counter_offer"])
@pytest.mark.parametrize(
    "field",
    [
        "id",
        "offered_by",
        "parent_offer_id",
        "created_round",
        "willing_by",
        "declined_by",
        "status",
    ],
)
def test_audience_support_does_not_allow_lifecycle_metadata_overrides(tool: str, field: str) -> None:
    context, arguments, _ = _case(tool)
    with pytest.raises(ValueError, match="Expected arguments"):
        parse_tool_choice(context, tool, {**arguments, field: None})
