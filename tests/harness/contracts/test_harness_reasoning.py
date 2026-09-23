from typing import Any

import pytest

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.observation import PlayerObservation
from cle.harness.reasoning import (
    ReasoningBudget,
    escalate_budget,
    native_reasoning_enabled,
    native_reasoning_request,
    native_reasoning_returned,
    reasoning_budget_for_attempt,
    reasoning_budget_for_context,
    reasoning_budget_instruction,
    reasoning_token_count,
    validate_native_reasoning_request,
)
from cle.players.contracts import PlayerContext


def test_native_reasoning_defaults_are_explicit_and_retain_evidence() -> None:
    assert validate_native_reasoning_request(None) == {
        "effort": "xhigh",
        "exclude": False,
    }
    assert native_reasoning_request("off") == {"enabled": False}
    assert not native_reasoning_enabled({"enabled": False})


def test_native_reasoning_validation_rejects_unobservable_or_ambiguous_requests() -> None:
    with pytest.raises(ValueError, match="exclude must be false"):
        validate_native_reasoning_request({"effort": "high", "exclude": True})
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_native_reasoning_request({"effort": "high", "max_tokens": 100})
    with pytest.raises(ValueError, match="cannot also set"):
        validate_native_reasoning_request({"enabled": False, "effort": "low"})


def test_native_reasoning_evidence_accepts_text_details_or_reported_tokens() -> None:
    usage: Any = {"completion_tokens_details": {"reasoning_tokens": 12}}

    assert reasoning_token_count(usage) == 12
    assert native_reasoning_returned("trace", (), {})
    assert native_reasoning_returned("", ({"type": "reasoning.text"},), {})
    assert native_reasoning_returned("", (), usage)
    assert not native_reasoning_returned("", (), {})


def _observation(
    *,
    phase: str = "main_game",
    my_vp: int = 2,
    opponent_vps: dict[Color, int] | None = None,
) -> PlayerObservation:
    return PlayerObservation(
        my_color=Color.RED,
        my_settlements=[],
        my_cities=[],
        my_roads=[],
        opponent_settlements={},
        opponent_cities={},
        opponent_roads={},
        my_resources={},
        my_dev_cards={},
        opponent_resource_counts={},
        opponent_dev_card_counts={},
        current_turn=1,
        current_phase=phase,
        turn_order=(),
        last_dice_roll=None,
        robber_position=None,
        my_vp=my_vp,
        opponent_vps=opponent_vps or {},
        longest_road_holder=None,
        largest_army_holder=None,
        my_longest_road_length=0,
        valid_actions=[],
        board_map=None,
        buildings_dict={},
        trade_window=None,
        is_my_turn=True,
        turn_player_color=Color.RED,
    )


def _context(
    prompt_key: str,
    *,
    turn_number: int = 20,
    observation: PlayerObservation | None = None,
    action_types: tuple[ActionType, ...] = (ActionType.END_TURN,),
) -> PlayerContext:
    observation = observation or _observation()
    return PlayerContext(
        context_id="test:1:RED",
        actor=Color.RED,
        turn_number=turn_number,
        phase=observation.current_phase,
        observation=observation,
        events=(),
        legal_actions=tuple(Action(Color.RED, action_type, None) for action_type in action_types),
        prompt_key=prompt_key,
    )


def test_budget_holds_high_floor_for_batched_setup() -> None:
    assert reasoning_budget_for_context(_context("initial_settlement_2")).effort == "high"
    assert reasoning_budget_for_context(_context("initial_road_1")).max_tokens is None


def test_budget_holds_high_floor_for_opening_turns() -> None:
    budget = reasoning_budget_for_context(_context("main_game", turn_number=3))
    assert (budget.effort, budget.max_tokens) == ("high", None)


def test_budget_escalates_to_high_in_endgame() -> None:
    observation = _observation(opponent_vps={Color.BLUE: 8})
    budget = reasoning_budget_for_context(_context("main_game", observation=observation))
    assert (budget.effort, budget.max_tokens) == ("high", None)
    own_win = _observation(my_vp=9)
    assert reasoning_budget_for_context(_context("main_game", observation=own_win)).effort == "high"


def test_budget_keeps_medium_for_robber_and_discard() -> None:
    assert reasoning_budget_for_context(_context("robber")).max_tokens is None
    assert reasoning_budget_for_context(_context("discarding")).effort == "medium"


def test_budget_keeps_medium_for_build_and_high_for_trade_menus() -> None:
    budget = reasoning_budget_for_context(
        _context("main_game", action_types=(ActionType.BUILD_CITY, ActionType.END_TURN))
    )
    assert (budget.effort, budget.max_tokens) == ("medium", None)
    trade = reasoning_budget_for_context(
        _context("main_game", action_types=(ActionType.OFFER_TRADE, ActionType.END_TURN))
    )
    assert (trade.effort, trade.max_tokens) == ("high", None)


def test_budget_escalates_one_tier_on_retry() -> None:
    assert escalate_budget(ReasoningBudget("medium", None)) == ReasoningBudget("high", None)
    assert escalate_budget(ReasoningBudget("max", None)) == ReasoningBudget("max", None)
    assert escalate_budget(ReasoningBudget("medium", 1024)) == ReasoningBudget("high", 2048)
    context = _context("main_game")
    assert reasoning_budget_for_attempt(context, None) == ReasoningBudget("medium", None)
    assert reasoning_budget_for_attempt(context, "retry") == ReasoningBudget("high", None)


def test_budget_floors_routine_mid_game_at_medium() -> None:
    budget = reasoning_budget_for_context(_context("main_game"))
    assert (budget.effort, budget.max_tokens) == ("medium", None)


def test_budget_instruction_states_effort_and_cap() -> None:
    capped = reasoning_budget_instruction(ReasoningBudget(effort="medium", max_tokens=1024))
    assert "1024" in capped and "medium" in capped and "commit" in capped
    uncapped = reasoning_budget_instruction(ReasoningBudget(effort="high", max_tokens=None))
    assert "high" in uncapped and "commit" in uncapped
