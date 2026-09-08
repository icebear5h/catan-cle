from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

from cle.harness import ModelResponse, PlayerSession, load_context_suite
from cle.harness.communication import CommunicationSuite, load_communication_suite
from cle.players import (
    CommunicationMode,
    FirstLegalPlayer,
    PlayerContext,
)
from cle.players.contracts import PlayerAttempt, PlayerChoice, TalkContext
from cle.players.agent import AgentPlayer
from cle.players.validation import action_from_choice
from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer, TradeOfferStatus
from cle.sandbox import CatanSandbox


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

    assert suite.version == "10.0.0"
    assert "rationale" not in suite.response.tags
    assert "<rationale>" not in suite.response.instruction
    assert "trade_offer" in suite.response.tags
    assert "trade_terms" not in suite.response.tags
    assert "discard" in suite.response.tags


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


@pytest.mark.asyncio
@pytest.mark.parametrize("echo_dynamic_prompt", [False, True])
async def test_agent_communication_trusts_only_authored_schema_echo_and_keeps_private_audience(echo_dynamic_prompt):
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    engine.append_message(
        speaker=Color.RED, text="DYNAMIC PRIVATE TABLE TALK", audience=(Color.BLUE,),
        intent="TRADE", causation_id="prior-talk",
    )
    data = load_communication_suite().model_dump()
    instruction = "LOCAL AUTHORED SCHEMA:\n" + data["sections"]["response_schema"]["template"]
    data["sections"]["response_schema"]["template"] = instruction
    suite = CommunicationSuite.model_validate(data)
    requests = []

    class EchoTransport:
        async def complete(self, request):
            requests.append(request)
            prefix = request.messages[-1].content if echo_dynamic_prompt else instruction
            return ModelResponse(content=(
                f"{prefix}\n<message>PRIVATE: WOOD for ORE?</message>"
                "<audience>RED</audience><intent>TRADE</intent>"
            ))

    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.BLUE] = AgentPlayer(
        Color.BLUE, EchoTransport(), session_id="private-echo:BLUE", communication_suite=suite,
    )
    sandbox = CatanSandbox(engine, players)
    result = await sandbox.step()

    assert len(requests) == 1
    assert "DYNAMIC PRIVATE TABLE TALK" in requests[0].messages[-1].content
    schema = next(component for component in requests[0].components if component.id == "environment.response_schema")
    assert schema.template == instruction
    assert len(result.messages) == (0 if echo_dynamic_prompt else 1)
    if not echo_dynamic_prompt:
        assert engine.project_messages(Color.RED)[-1].payload["text"] == "PRIVATE: WOOD for ORE?"
        assert engine.project_messages(Color.BLUE)[-1].payload["audience"] == (Color.RED,)
        assert result.messages[0].public_payload is None
    assert engine.project_messages(Color.WHITE) == ()
    assert engine.project_messages(Color.ORANGE) == ()
    assert players[Color.BLUE].session.messages == []
    assert players[Color.BLUE].session.receipts == {}


def test_player_session_snapshot_restores_identity_and_continuity():
    session = PlayerSession(Color.RED, "game:RED")
    session.strategic_memory = "build cities"
    snapshot = session.snapshot()
    session.strategic_memory = "changed"

    session.restore(snapshot)

    assert session.strategic_memory == "build cities"


@pytest.fixture
def trade_context():
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    return replace(
        _context(engine),
        legal_actions=(Action(Color.RED, ActionType.OFFER_TRADE, "parameterized trade"),),
    )


@pytest.mark.parametrize("field_name,value,error", [
    ("id", "injected-id", "lifecycle"),
    ("created_round", 0, "lifecycle"),
    ("willing_by", {Color.BLUE}, "lifecycle"),
    ("declined_by", {Color.BLUE}, "lifecycle"),
    ("status", TradeOfferStatus.WITHDRAWN, "lifecycle"),
    ("status", "active", "lifecycle"),
    ("offered_by", Color.BLUE, "offerer"),
    ("audience", frozenset({Color.BLUE}), "audience"),
    ("parent_offer_id", "some-root", "parent"),
    ("give", [1, 0, 0, 0, 0], "bundles"),
    ("give", (True, 0, 0, 0, 0), "integers"),
    ("give", (-1, 0, 0, 0, 0), "negative"),
    ("receive_any", True, "integers"),
])
def test_action_materializer_revalidates_mutated_trade_fields(trade_context, field_name, value, error):
    offer = TradeOffer(Color.RED, frozenset(COLORS[1:]), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1))
    setattr(offer, field_name, value)
    with pytest.raises(ValueError, match=error):
        action_from_choice(trade_context, PlayerChoice(0, trade_offer=offer))


def test_action_materializer_binds_opaque_counter_parent_and_detaches(trade_context):
    parent = "turn:window:opaque:o2"
    context = replace(
        trade_context,
        actor=Color.BLUE,
        legal_actions=(Action(Color.BLUE, ActionType.COUNTER_OFFER, f"COUNTER_OFFER:{parent}: supply a named trade_offer"),),
    )
    offer = TradeOffer(Color.BLUE, frozenset({Color.RED}), (0, 0, 0, 0, 1), (1, 0, 0, 0, 0), parent_offer_id=parent)
    choice = PlayerChoice(0, trade_offer=offer)
    action = action_from_choice(context, choice)
    assert action.value.parent_offer_id == parent
    assert action.value == offer and action.value is not offer
    offer.parent_offer_id = "turn:window:opaque:o1"
    with pytest.raises(ValueError, match="parent"):
        action_from_choice(context, choice)
    assert action.value.parent_offer_id == parent


def test_action_materializer_cannot_override_a_concrete_trade(trade_context):
    offer = TradeOffer(Color.RED, frozenset(COLORS[1:]), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1))
    context = replace(trade_context, legal_actions=(Action(Color.RED, ActionType.OFFER_TRADE, offer),))
    action = action_from_choice(context, PlayerChoice(0))
    assert action.value == offer and action.value is not offer
    with pytest.raises(ValueError, match="concrete"):
        action_from_choice(context, PlayerChoice(0, trade_offer=replace(offer, give=(2, 0, 0, 0, 0))))


@pytest.mark.parametrize("choice,error", [
    (None, "PlayerChoice"),
    ({"action_index": 0}, "PlayerChoice"),
    (PlayerChoice(True), "integer"),
    (PlayerChoice("0"), "integer"),
    (PlayerChoice(-1), "outside"),
    (PlayerChoice(1), "outside"),
    (PlayerChoice(0, trade_offer={}), "TradeOffer"),
    (PlayerChoice(0, game_plan={}), "game_plan"),
    (PlayerChoice(0, usage={}), "usage"),
    (PlayerChoice(0, reasoning_request=((False, "high"),)), "reasoning_request"),
])
def test_action_materializer_rejects_invalid_choice_shapes(trade_context, choice, error):
    with pytest.raises(ValueError, match=error):
        action_from_choice(trade_context, choice)


def test_action_materializer_supports_exact_discard_and_legacy_none(trade_context):
    context = replace(
        trade_context,
        legal_actions=(Action(Color.RED, ActionType.DISCARD, None),),
        discard_count=4,
    )
    context.observation.my_resources.update(WOOD=3, ORE=5)
    cards = ("WOOD", "ORE", "ORE", "ORE")
    assert action_from_choice(context, PlayerChoice(0, discard_cards=cards)) == Action(Color.RED, ActionType.DISCARD, cards)
    assert action_from_choice(context, PlayerChoice(0)).value is None
    for invalid, error in (
        (["WOOD"] * 4, "tuple"),
        (("WOOD", "ORE", "GOLD", "ORE"), "named resource"),
        (("WOOD", "ORE"), "exactly 4"),
        (("WOOD",) * 4, "does not hold"),
    ):
        with pytest.raises(ValueError, match=error):
            action_from_choice(context, PlayerChoice(0, discard_cards=invalid))
    with pytest.raises(ValueError, match="only allowed"):
        action_from_choice(context, PlayerChoice(0, trade_offer=TradeOffer(
            Color.RED, frozenset(COLORS[1:]), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1)
        )))
    context = replace(context, legal_actions=(Action(Color.RED, ActionType.END_TURN, None),))
    with pytest.raises(ValueError, match="only allowed"):
        action_from_choice(context, PlayerChoice(0, discard_cards=cards))


@pytest.mark.asyncio
async def test_agent_receipts_cached_choices_and_snapshots_are_detached(trade_context):
    player = AgentPlayer(Color.RED, AsyncFixedTransport([]), session_id="snapshot:RED")
    choice = PlayerChoice(
        0,
        trade_offer=TradeOffer(Color.RED, frozenset(COLORS[1:]), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1)),
        native_reasoning_details=({"text": "original"},),
    )
    attempt = PlayerAttempt(trade_context.context_id, choice)
    player.accept(attempt, SimpleNamespace(context=trade_context, after_revision=1))
    choice.trade_offer.give = (2, 0, 0, 0, 0)
    choice.native_reasoning_details[0]["text"] = "edited"
    receipt = player.session.receipts[trade_context.context_id]
    assert receipt.choice.trade_offer.give == (1, 0, 0, 0, 0)
    assert receipt.choice.native_reasoning_details == ({"text": "original"},)
    cached = await player.choose(trade_context)
    cached.choice.trade_offer.give = (3, 0, 0, 0, 0)
    assert receipt.choice.trade_offer.give == (1, 0, 0, 0, 0)
    snapshot = player.snapshot()
    receipt.choice.trade_offer.give = (4, 0, 0, 0, 0)
    player.restore(snapshot)
    assert player.session.receipts[trade_context.context_id].choice.trade_offer.give == (1, 0, 0, 0, 0)
    snapshot.session.receipts[0][1].choice.trade_offer.give = (5, 0, 0, 0, 0)
    assert player.session.receipts[trade_context.context_id].choice.trade_offer.give == (1, 0, 0, 0, 0)


@pytest.mark.asyncio
async def test_agent_detaches_mutable_provider_response_metadata():
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    response = ModelResponse(content="<action>0</action>", native_reasoning_details=({"text": "original"},))
    player = AgentPlayer(Color.RED, AsyncFixedTransport([response]), session_id="provider:RED")
    attempt = await player.choose(_context(engine))
    response.native_reasoning_details[0]["text"] = "edited"
    assert attempt.model_response.native_reasoning_details == ({"text": "original"},)
    assert attempt.choice.native_reasoning_details == ({"text": "original"},)
