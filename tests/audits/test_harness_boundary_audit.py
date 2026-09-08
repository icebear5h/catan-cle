"""Offline boundary regressions, including the admitted custom-player policy."""

import asyncio
import json
import pickle
from dataclasses import FrozenInstanceError, dataclass, field

import httpx
import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import SETTLEMENT, Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer
from cle.harness.communication import default_communication_suite_path
from cle.harness.context import PlayerResponseParseError, PlayerResponseParser
from cle.harness.models import ModelMessage, ModelRequest, ModelResponse
from cle.harness.providers import (
    OpenRouterConfig,
    OpenRouterTransport,
    VLLMConfig,
    VLLMTransport,
)
from cle.harness.suite import default_suite_path, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    CommitmentProposal,
    CommunicationChoice,
    CommunicationMode,
    PlayerAttempt,
    PlayerChoice,
)
from cle.sandbox import CatanSandbox, RetryPolicy
from cle.sandbox.catan import PlayerResponseError, PostActionCommunicationError
from cle.sandbox.communication import CommunicationPolicy
from cle.sandbox.factory import LiveSandboxConfig, create_live_sandbox


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


class BoundaryDefect(AssertionError):
    """A specifically verified boundary regression."""


class QuietCommunication(CommunicationPolicy):
    def pre_action(self, engine):
        return ()

    def after_events(self, engine, events, *, round_number):
        return ()


@dataclass
class LocalTransport:
    responses: list[str]
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request):
        assert self.responses, "Unexpected additional model request"
        self.requests.append(request)
        return ModelResponse(content=self.responses.pop(0))


@pytest.fixture
def sandbox():
    return CatanSandbox(
        GameEngine(COLORS, seed=7, shuffle_players=False),
        {color: FirstLegalPlayer(color) for color in COLORS},
        retry_policy=RetryPolicy(2),
        communication_policy=QuietCommunication(),
    )


@pytest.fixture
def trade_sandbox(sandbox):
    state = sandbox.game_engine.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.player_state["P0_HAS_ROLLED"] = True
    state.player_state["P0_WOOD_IN_HAND"] = 4
    state.resource_freqdeck[0] -= 4
    for index in range(1, 4):
        state.player_state[f"P{index}_ORE_IN_HAND"] = 1
        state.resource_freqdeck[4] -= 1
    state.playable_actions = generate_playable_actions(state)
    return sandbox


def open_root(sandbox, wood=1):
    offer = TradeOffer(Color.RED, frozenset(COLORS[1:]), (wood, 0, 0, 0, 0), (0, 0, 0, 0, 1))
    return sandbox.game_engine.step(
        Action(Color.RED, ActionType.OFFER_TRADE, offer)
    ).resolved_action.value


@pytest.mark.asyncio
async def test_commented_message_must_not_publish_private_draft(sandbox):
    draft = "PRIVATE DRAFT: I have three ORE."
    transport = LocalTransport(
        [
            f"<!-- <message>{draft}</message> -->"
            "<message>SILENCE</message><audience>PUBLIC</audience><intent>TRADE</intent>"
        ]
    )
    sandbox.register_player(AgentPlayer(Color.BLUE, transport, session_id="audit:BLUE"))
    sandbox.communication_policy = CommunicationPolicy()

    result = await sandbox.step()

    assert len(transport.requests) == 1
    assert len(result.transitions) == 1
    if result.messages:
        assert len(result.messages) == 1
        assert result.messages[0].public_payload["text"] == draft
        for color in COLORS:
            assert sandbox.game_engine.project_messages(color)[0].payload["text"] == draft
        raise BoundaryDefect("Commented draft was broadcast to every player")
    assert all(sandbox.game_engine.project_messages(color) == () for color in COLORS)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("<game_plan>One possibility is <action>1</action>", id="unclosed-plan"),
        pytest.param("<action>1</action><action >2</action>", id="conflicting-whitespace-tag"),
    ],
)
def test_malformed_or_conflicting_action_xml_must_be_rejected(sandbox, text):
    context = sandbox.decision_context()
    assert len(context.legal_actions) > 2
    try:
        choice = PlayerResponseParser(load_context_suite()).parse(
            context, ModelResponse(content=text)
        )
    except PlayerResponseParseError:
        return
    assert choice.action_index == 1
    assert sandbox.game_engine.is_action_valid(context.action_at(choice.action_index))
    raise BoundaryDefect("Malformed or conflicting XML selected legal action 1")


def test_duplicate_trade_offer_xml_must_be_rejected(trade_sandbox):
    context = trade_sandbox.decision_context()
    index = next(
        i
        for i, action in enumerate(context.legal_actions)
        if action.action_type == ActionType.OFFER_TRADE
    )
    text = (
        f"<action>{index}</action>"
        '<trade_offer>{"give":{"WOOD":1},"receive":{"ORE":1}}</trade_offer>'
        '<trade_offer>{"give":{"WOOD":4},"receive":{"ORE":1}}</trade_offer>'
    )
    try:
        choice = PlayerResponseParser(load_context_suite()).parse(
            context, ModelResponse(content=text)
        )
    except PlayerResponseParseError:
        return
    assert choice.action_index == index
    assert choice.trade_offer.give == (1, 0, 0, 0, 0)
    assert choice.trade_offer.receive == (0, 0, 0, 0, 1)
    raise BoundaryDefect("Conflicting trade terms silently became the first offer")


@pytest.mark.asyncio
async def test_invalid_commitment_must_not_poison_next_step(sandbox, monkeypatch):
    async def communicate(context):
        return CommunicationChoice(
            mode=CommunicationMode.SAY,
            text="Trade?",
            audience=(Color.RED,),
            intent="BRIBE",
            commitment=CommitmentProposal("condition", "promise", "tomorrow"),
        )

    monkeypatch.setattr(sandbox.players[Color.BLUE], "communicate", communicate)
    sandbox.communication_policy = CommunicationPolicy()
    engine = sandbox.game_engine
    try:
        await sandbox.step()
    except PostActionCommunicationError:
        assert engine.commitments == []
        assert engine.project_messages(Color.RED) == ()
    assert len(engine.state.actions) == 1
    assert sandbox.players[Color.RED].accepted_choices == 1
    sandbox.communication_policy = QuietCommunication()
    before = engine.revision
    try:
        await sandbox.step()
    except TypeError as exc:
        assert str(exc) == "'<=' not supported between instances of 'str' and 'int'"
        assert len(engine.commitments) == 1
        assert engine.commitments[0].expires_turn == "tomorrow"
        assert sandbox.communication_trace[0].accepted is True
        assert len(engine.state.actions) == 2
        assert engine.revision == before + 1
        assert sandbox.players[Color.RED].accepted_choices == 1
        raise BoundaryDefect(
            "Expiry comparison crashed after action/event commit, before accept"
        ) from exc
    assert engine.commitments == []
    assert sandbox.players[Color.RED].accepted_choices == 2


@pytest.mark.asyncio
async def test_mutating_returned_offer_must_not_partially_commit_barrier(
    trade_sandbox, monkeypatch
):
    sandbox = trade_sandbox
    engine = sandbox.game_engine
    root = open_root(sandbox)
    offer = TradeOffer(
        Color.WHITE,
        frozenset({Color.RED}),
        (0, 0, 0, 0, 1),
        (2, 0, 0, 0, 0),
        parent_offer_id=root.id,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    orange_choose = sandbox.players[Color.ORANGE].choose

    async def white_choose(context, feedback=None):
        index = next(
            i
            for i, action in enumerate(context.legal_actions)
            if action.action_type == ActionType.COUNTER_OFFER
        )
        return PlayerAttempt(context.context_id, PlayerChoice(index, trade_offer=offer))

    async def delayed_choose(context, feedback=None):
        entered.set()
        await release.wait()
        return await orange_choose(context, feedback)

    monkeypatch.setattr(sandbox.players[Color.WHITE], "choose", white_choose)
    monkeypatch.setattr(sandbox.players[Color.ORANGE], "choose", delayed_choose)
    before = engine.revision
    pending = asyncio.create_task(sandbox.step())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert engine.is_action_valid(Action(Color.WHITE, ActionType.COUNTER_OFFER, offer))
        offer.give = (0, 0, 0, 0, 99)
        assert not engine.is_action_valid(Action(Color.WHITE, ActionType.COUNTER_OFFER, offer))
        release.set()
        try:
            result = await asyncio.wait_for(pending, 2)
        except PlayerResponseError:
            assert engine.revision == before
            assert all(player.accepted_choices == 0 for player in sandbox.players.values())
            return
        except ValueError as exc:
            assert "not playable right now" in str(exc)
            assert "color=C.WHITE, action_type=AT.COUNTER_OFFER" in str(exc)
            if engine.revision != before:
                assert engine.revision == before + 1
                assert engine.state.actions[-1] == Action(
                    Color.BLUE, ActionType.REJECT_TRADE, root.id
                )
                assert engine.state.trade_window.offers[root.id].declined_by == {Color.BLUE}
                assert [sandbox.players[c].accepted_choices for c in COLORS] == [0, 1, 0, 0]
                raise BoundaryDefect(
                    "BLUE committed before mutated WHITE counteroffer failed"
                ) from exc
            assert all(player.accepted_choices == 0 for player in sandbox.players.values())
            return
        assert len(result.transitions) == 3
        assert result.transitions[1].requested_action.value.give == (0, 0, 0, 0, 1)
        assert [sandbox.players[c].accepted_choices for c in COLORS] == [0, 1, 1, 1]
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize("field_name", ["trade_window", "buildings_dict"])
def test_observation_mutation_must_not_change_engine(trade_sandbox, field_name):
    engine = trade_sandbox.game_engine
    root = open_root(trade_sandbox)
    context = trade_sandbox.decision_context()
    before = engine.revision
    before_engine = pickle.dumps(engine.snapshot())
    if field_name == "trade_window":
        original_give = tuple(engine.state.trade_window.offers[root.id].give)
        assert original_give == (1, 0, 0, 0, 0)
        observed_offer = context.observation.trade_window.offers[root.id]
        assert tuple(observed_offer.give) == original_give
        try:
            observed_offer.give = (4, 0, 0, 0, 0)
        except (FrozenInstanceError, TypeError):
            pass
        except AttributeError as exc:
            assert any(
                text in str(exc).lower()
                for text in ("has no setter", "can't set attribute", "read-only")
            )
        if engine.state.trade_window.offers[root.id].give != original_give:
            assert context.observation.trade_window is engine.state.trade_window
            assert engine.state.trade_window.offers[root.id].give == (4, 0, 0, 0, 0)
            assert engine.events[0].public_payload["give"] == {"WOOD": 1}
            assert engine.revision == before
            raise BoundaryDefect("Editing observation changed live trade terms without an event")
    else:
        tile = next(iter(engine.state.board.map.land_tiles.values()))
        node = next(iter(tile.nodes.values()))
        assert node not in engine.state.board.buildings
        observed_buildings = context.observation.buildings_dict
        assert node not in observed_buildings
        try:
            observed_buildings[node] = (Color.RED, SETTLEMENT)
        except (FrozenInstanceError, TypeError):
            pass
        if node in engine.state.board.buildings:
            assert context.observation.buildings_dict is engine.state.board.buildings
            assert engine.state.board.buildings[node] == (Color.RED, SETTLEMENT)
            assert engine.revision == before
            raise BoundaryDefect("Editing observation inserted a live engine building")
    assert pickle.dumps(engine.snapshot()) == before_engine


@pytest.mark.asyncio
async def test_typed_counter_parent_must_match_selected_menu(trade_sandbox, monkeypatch):
    sandbox = trade_sandbox
    old = open_root(sandbox)
    await sandbox.step()
    assert sandbox.game_engine.state.trade_window.offers[old.id].declined_by == set(COLORS[1:])
    new = open_root(sandbox, wood=2)
    original_choose = sandbox.players[Color.BLUE].choose
    selected = []

    async def choose(context, feedback=None):
        if feedback:
            return await original_choose(context, feedback)
        assert all(old.id not in str(action.value) for action in context.legal_actions)
        index = next(
            i
            for i, action in enumerate(context.legal_actions)
            if action.action_type == ActionType.COUNTER_OFFER
        )
        selected.append(context.legal_actions[index])
        return PlayerAttempt(
            context.context_id,
            PlayerChoice(
                index,
                trade_offer=TradeOffer(
                    Color.BLUE,
                    frozenset({Color.RED}),
                    (0, 0, 0, 0, 1),
                    (3, 0, 0, 0, 0),
                    parent_offer_id=old.id,
                ),
            ),
        )

    monkeypatch.setattr(sandbox.players[Color.BLUE], "choose", choose)
    result = await sandbox.step()
    assert len(selected) == 1
    assert selected[0].value.startswith(f"COUNTER_OFFER:{new.id}:")
    committed = result.transitions[0].requested_action
    assert committed.color == Color.BLUE
    if committed.action_type == ActionType.COUNTER_OFFER:
        if committed.value.parent_offer_id != new.id:
            assert committed.value.parent_offer_id == old.id
            raise BoundaryDefect("Selected counter for o2 committed a counter for answered o1")
    else:
        assert committed == Action(Color.BLUE, ActionType.REJECT_TRADE, new.id)
        assert sandbox.decision_trace[-1].validation_error


@pytest.mark.parametrize(
    "kind", ["attempt-none", "choice-dict", "offer-dict", "offer-list"],
)
@pytest.mark.asyncio
async def test_custom_player_protocol_violation_is_rejected_or_retried(
    trade_sandbox, monkeypatch, kind
):
    """Protocol-violating returns must now receive typed rejection or retry."""
    sandbox = trade_sandbox
    original_choose = sandbox.players[Color.RED].choose
    feedbacks = []

    async def choose(context, feedback=None):
        feedbacks.append(feedback)
        if len(feedbacks) > 1:
            assert pickle.dumps(sandbox.game_engine.snapshot()) == before_engine
            assert tuple(player.snapshot() for player in sandbox.players.values()) == before_players
            return await original_choose(context, feedback)
        if kind == "attempt-none":
            return None
        if kind == "choice-dict":
            return PlayerAttempt(context.context_id, {"action_index": 0})
        index = next(
            i
            for i, action in enumerate(context.legal_actions)
            if action.action_type == ActionType.OFFER_TRADE
        )
        offer = (
            {"give": {"WOOD": 1}, "receive": {"ORE": 1}}
            if kind == "offer-dict"
            else TradeOffer(Color.RED, frozenset(COLORS[1:]), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1))
        )
        if kind == "offer-list":
            offer.give = [1, 0, 0, 0, 0]
        return PlayerAttempt(context.context_id, PlayerChoice(index, trade_offer=offer))

    monkeypatch.setattr(sandbox.players[Color.RED], "choose", choose)
    # Materialize lazy observation entries before measuring player-output effects.
    sandbox.decision_context()
    before_engine = pickle.dumps(sandbox.game_engine.snapshot())
    before_players = tuple(player.snapshot() for player in sandbox.players.values())
    result = await sandbox.step()
    assert len(feedbacks) == 2 and feedbacks[1]
    assert len(sandbox.decision_trace) == 1
    assert sandbox.decision_trace[0].validation_error
    assert sandbox.decision_trace[0].choice is None
    assert result.transitions[0].requested_action.action_type == ActionType.END_TURN
    assert sandbox.players[Color.RED].accepted_choices == 1


@pytest.mark.asyncio
async def test_random_mode_must_not_invoke_injected_transport():
    transport = LocalTransport(["<action>0</action>"])
    sandbox = create_live_sandbox(
        LiveSandboxConfig(
            mode="random",
            seed=7,
            shuffle_players=False,
            palette="canonical_four",
            context_suite_path=str(default_suite_path()),
            communication_suite_path=str(default_communication_suite_path()),
        ),
        transport=transport,
    )
    await sandbox.step()
    assert all(type(sandbox.players[c]) is FirstLegalPlayer for c in COLORS[1:])
    if transport.requests:
        assert len(transport.requests) == 1
        assert isinstance(sandbox.players[Color.RED], AgentPlayer)
        raise BoundaryDefect("Random-mode first step invoked the injected model transport")
    assert type(sandbox.players[Color.RED]) is FirstLegalPlayer


@pytest.mark.asyncio
async def test_decision_prompt_must_retain_visible_table_talk_and_commitments(trade_sandbox):
    sandbox = trade_sandbox
    engine = sandbox.game_engine
    markers = ("AUDIT_PRIVATE_TALK", "AUDIT_COMMITMENT_PROMISE")
    engine.append_message(
        speaker=Color.BLUE,
        text=markers[0],
        audience=(Color.RED,),
        intent="BRIBE",
        causation_id="audit:private-bribe",
        commitment=("Spare BLUE", markers[1], 9),
    )
    assert engine.project_messages(Color.WHITE) == ()
    assert engine.active_commitments(Color.WHITE) == ()
    transport = LocalTransport(["<message>SILENCE</message>", "<action>0</action>"])
    sandbox.register_player(AgentPlayer(Color.RED, transport, session_id="audit:RED"))
    sandbox.communication_policy = CommunicationPolicy()

    await sandbox.step()

    assert len(transport.requests) == 2
    talk, decision = transport.requests
    assert ":talk:" in talk.decision_id and ":talk:" not in decision.decision_id
    talk_text = "\n".join(message.content for message in talk.messages)
    decision_text = "\n".join(message.content for message in decision.messages)
    assert all(marker in talk_text for marker in markers)
    if any(marker not in decision_text for marker in markers):
        assert all(marker not in decision_text for marker in markers)
        raise BoundaryDefect("Both private message and active promise disappeared before choice")


@pytest.mark.parametrize("provider", ["openrouter", "vllm"])
@pytest.mark.asyncio
async def test_shared_provider_isolates_sessions_and_preserves_borrowed_client(provider):
    seen = {}
    both_entered, cancel_entered, never = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def handler(request):
        session = request.headers["x-session-id"]
        seen[session] = json.loads(request.content)["messages"]
        if session in {"RED", "BLUE"}:
            if {"RED", "BLUE"}.issubset(seen):
                both_entered.set()
            await both_entered.wait()
        if session == "cancel":
            cancel_entered.set()
            await never.wait()
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "<action>0</action>"}}]}
        )

    def request(session):
        return ModelRequest(session, session, (ModelMessage("user", f"{session} private context"),))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = (
            OpenRouterTransport(
                OpenRouterConfig(
                    model="audit", endpoint="https://provider.invalid/v1", max_retries=0
                ),
                api_key="audit-not-a-secret",
                client=client,
            )
            if provider == "openrouter"
            else VLLMTransport(
                VLLMConfig(model="audit", base_url="https://provider.invalid/v1", max_retries=0),
                client=client,
            )
        )
        responses = await asyncio.wait_for(
            asyncio.gather(transport.complete(request("RED")), transport.complete(request("BLUE"))),
            2,
        )
        assert [response.content for response in responses] == ["<action>0</action>"] * 2
        pending = asyncio.create_task(transport.complete(request("cancel")))
        try:
            await asyncio.wait_for(cancel_entered.wait(), 2)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await transport.aclose()
        assert not client.is_closed
        assert (await transport.complete(request("after-cancel"))).content == "<action>0</action>"
        assert seen == {
            session: [{"role": "user", "content": f"{session} private context"}]
            for session in ("RED", "BLUE", "cancel", "after-cancel")
        }
