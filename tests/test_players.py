from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from cle.harness import ModelResponse, PlayerSession, load_context_suite
from cle.players import (
    CommunicationMode,
    FirstLegalPlayer,
    PlayerContext,
)
from cle.players.contracts import TalkContext
from cle.players.agent import AgentPlayer
from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


@dataclass
class AsyncFixedTransport:
    responses: list[ModelResponse]
    requests: list = field(default_factory=list)

    async def complete(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def _context(engine, prompt_key="initial_settlement_1"):
    actor = engine.state.current_color()
    return PlayerContext(
        context_id=f"{engine.id}:{engine.revision}:{actor.value}",
        actor=actor,
        turn_number=engine.state.num_turns,
        phase="initial_placement",
        observation=engine.observe(actor),
        events=engine.project_events(actor),
        legal_actions=tuple(engine.state.playable_actions),
        prompt_key=prompt_key,
    )


def test_default_player_suite_uses_single_trade_offer_surface():
    suite = load_context_suite()

    assert suite.version == "9.0.0"
    assert "rationale" not in suite.response.tags
    assert "<rationale>" not in suite.response.instruction
    assert "trade_offer" in suite.response.tags
    assert "trade_terms" not in suite.response.tags


@pytest.mark.asyncio
async def test_agent_player_assembles_full_context_and_records_only_accepted_attempt():
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    transport = AsyncFixedTransport([
        ModelResponse(
            content=(
                "<game_plan>expand toward wheat</game_plan>"
                "<action>0</action>"
            ),
            model="test/model",
        )
    ])
    player = AgentPlayer(
        Color.RED,
        transport,
        session_id="game:RED",
        suite=load_context_suite(),
    )
    context = _context(engine)

    attempt = await player.choose(context)

    assert attempt.choice is not None
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
async def test_agent_player_parses_parameterized_trade_choice():
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
    )

    attempt = await player.choose(context)

    assert attempt.choice.trade_offer == TradeOffer(
        offered_by=Color.RED,
        audience=frozenset(COLORS[1:]),
        give=(1, 0, 0, 0, 0),
        receive=(0, 0, 0, 0, 1),
    )


@pytest.mark.asyncio
async def test_agent_player_rejects_positional_trade_tuple():
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
    )

    attempt = await player.choose(context)

    assert attempt.choice is None
    assert "JSON object" in attempt.validation_error


@pytest.mark.asyncio
async def test_agent_player_returns_validation_error_for_bad_menu_index():
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    player = AgentPlayer(
        Color.RED,
        AsyncFixedTransport([ModelResponse(content="<action>999</action>")]),
        session_id="game:RED",
    )

    attempt = await player.choose(_context(engine))

    assert attempt.choice is None
    assert "Choose an action index" in attempt.validation_error
    assert player.session.messages == []


@pytest.mark.asyncio
async def test_first_legal_player_is_async_and_snapshot_safe():
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    player = FirstLegalPlayer(Color.RED)
    context = _context(engine)

    attempt = await player.choose(context)
    snapshot = player.snapshot()

    assert attempt.choice.action_index == 0
    player.accept(attempt, SimpleNamespace(context=context))
    assert player.accepted_choices == 1
    player.restore(snapshot)
    assert player.accepted_choices == 0


@pytest.mark.asyncio
async def test_agent_player_communication_uses_bounded_structured_contract():
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    transition = engine.step(engine.state.playable_actions[0])
    transport = AsyncFixedTransport([
        ModelResponse(
            content=(
                "<message>Do not block me and I will trade later.</message>"
                "<audience>RED</audience>"
                "<intent>BRIBE</intent>"
                "<commitment_condition>RED avoids BLUE</commitment_condition>"
                "<commitment_promise>BLUE offers ORE</commitment_promise>"
                "<commitment_expires_turn>3</commitment_expires_turn>"
            )
        )
    ])
    player = AgentPlayer(Color.BLUE, transport, session_id="game:BLUE")
    cause = engine.project_events(Color.BLUE)[0]
    context = TalkContext(
        context_id="talk:1:BLUE",
        player=Color.BLUE,
        participants=COLORS,
        cause=cause,
        visible_through_sequence=transition.after_revision - 1,
        game_events=engine.project_game_events(Color.BLUE),
        recent_messages=(),
    )

    choice = await player.communicate(context)

    assert choice.mode == CommunicationMode.SAY
    assert choice.audience == (Color.RED,)
    assert choice.commitment.promise == "BLUE offers ORE"
    assert transport.requests[0].messages[0].content == (
        "You are playing a game of Catan. You are playing as BLUE."
    )
    assert "COMMUNICATION POLICY" in transport.requests[0].messages[-1].content
    assert "COMPLETE VISIBLE GAME EVENTS" in transport.requests[0].messages[-1].content
    assert transport.requests[0].components[1].channel == "environment"


def test_player_session_snapshot_restores_identity_and_continuity():
    session = PlayerSession(Color.RED, "game:RED")
    session.strategic_memory = "build cities"
    snapshot = session.snapshot()
    session.strategic_memory = "changed"

    session.restore(snapshot)

    assert session.strategic_memory == "build cities"
