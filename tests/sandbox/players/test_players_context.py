"""Suite defaults and full context assembly."""
import json
from types import SimpleNamespace
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer
from cle.harness import ModelResponse, default_suite_path, load_context_suite
from cle.players import (
    FirstLegalPlayer,
    PlayerContext,
)
from cle.players.agent import AgentPlayer

from .support import COLORS, AsyncFixedTransport, _context


def test_default_player_suite_uses_semantic_json_call_surface() -> None:
    suite = load_context_suite()

    assert suite.version == "11.0.0"
    assert suite.response.format == "json"
    assert suite.response.tags == ("game_plan", "tool", "arguments")
    assert "rationale" not in suite.response.tags
    assert "<rationale>" not in suite.response.instruction
    assert "action_index" not in suite.response.instruction
    assert "<action>" not in suite.response.instruction


def test_bare_agent_player_defaults_to_shared_fresh_suites() -> None:
    player = AgentPlayer(
        Color.RED, AsyncFixedTransport([]), session_id="default:RED",
    )

    assert player.suite.id == "catan-shared"
    assert player.suite.context.memory_mode == "fresh_notes"
    assert player.suite.response.tags == ("tool", "arguments", "notes")
    assert player.session.context_policy == "fresh_notes"


@pytest.mark.asyncio
async def test_agent_player_assembles_full_context_and_records_only_accepted_attempt() -> None:
    engine: Any = GameEngine(COLORS, seed=9, shuffle_players=False)
    transport = AsyncFixedTransport([
        ModelResponse(
            content=json.dumps({
                "game_plan": "expand toward wheat",
                "tool": "build_settlement",
                "arguments": {"node": f"<N{engine.state.playable_actions[0].value:02d}>"},
            }),
            model="test/model",
        )
    ])
    player = AgentPlayer(
        Color.RED,
        transport,
        session_id="game:RED",
        suite=load_context_suite(),
    )
    context: Any = _context(engine)

    attempt: Any = await player.choose(context)

    assert attempt.choice is not None
    assert attempt.choice.action_index == 0
    assert player.suite.version == "11.0.0"
    assert player.session.messages == []
    assert transport.requests[0].messages[-1].role == "user"

    transition = engine.step(context.action_at(attempt.choice.action_index))
    result = SimpleNamespace(
        context=context,
        transitions=(transition,),
        after_revision=transition.after_revision,
    )
    player.accept(attempt, result)

    assert [message.role for message in player.session.messages] == [
        "user",
        "assistant",
    ]
    assert player.session.strategic_memory == "expand toward wheat"
    assert len(player.session.receipts) == 1


@pytest.mark.asyncio
async def test_agent_player_parses_parameterized_trade_choice() -> None:
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    context = _context(engine, prompt_key="main_game")
    context = PlayerContext(
        context_id=context.context_id,
        actor=context.actor,
        turn_number=context.turn_number,
        phase="main_game",
        observation=context.observation,
        events=context.events,
        legal_actions=(
            Action(Color.RED, ActionType.OFFER_TRADE, "parameterized trade"),
        ),
        prompt_key="main_game",
    )
    player = AgentPlayer(
        Color.RED,
        AsyncFixedTransport([
            ModelResponse(
                content=(
                    "<game_plan>trade</game_plan>"
                    "<action>0</action>"
                    '<trade_offer>{"give":{"WOOD":1},'
                    '"receive":{"ORE":1}}</trade_offer>'
                )
            )
        ]),
        session_id="game:RED",
        suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
    )

    attempt: Any = await player.choose(context)

    assert attempt.choice.trade_offer == TradeOffer(
        offered_by=Color.RED,
        audience=frozenset(COLORS[1:]),
        give=(1, 0, 0, 0, 0),
        receive=(0, 0, 0, 0, 1),
    )


@pytest.mark.asyncio
async def test_agent_player_rejects_positional_trade_tuple() -> None:
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    base = _context(engine, prompt_key="main_game")
    context = PlayerContext(
        context_id=base.context_id,
        actor=base.actor,
        turn_number=base.turn_number,
        phase="main_game",
        observation=base.observation,
        events=base.events,
        legal_actions=(
            Action(Color.RED, ActionType.OFFER_TRADE, "parameterized trade"),
        ),
        prompt_key="main_game",
    )
    player = AgentPlayer(
        Color.RED,
        AsyncFixedTransport([
            ModelResponse(
                content=(
                    "<action>0</action>"
                    "<trade_offer>[1,0,0,0,0,0,0,0,0,1]</trade_offer>"
                )
            )
        ]),
        session_id="game:RED",
        suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
    )

    attempt: Any = await player.choose(context)

    assert attempt.choice is None
    assert "JSON object" in attempt.validation_error


@pytest.mark.asyncio
async def test_agent_player_returns_validation_error_for_bad_menu_index() -> None:
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    player = AgentPlayer(
        Color.RED,
        AsyncFixedTransport([ModelResponse(content="<action>999</action>")]),
        session_id="game:RED",
        suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
    )

    attempt: Any = await player.choose(_context(engine))

    assert attempt.choice is None
    assert "Choose an action index" in attempt.validation_error
    assert player.session.messages == []


@pytest.mark.asyncio
async def test_first_legal_player_is_async_and_snapshot_safe() -> None:
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    player = FirstLegalPlayer(Color.RED)
    context = _context(engine)

    attempt: Any = await player.choose(context)
    snapshot = player.snapshot()

    assert attempt.choice.action_index == 0
    player.accept(attempt, SimpleNamespace(context=context))
    assert player.accepted_choices == 1
    player.restore(snapshot)
    assert player.accepted_choices == 0
