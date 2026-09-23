"""Response parsing and commitment rejection boundaries."""
import pickle
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from cle.game_engine.models.enums import SETTLEMENT
from cle.game_engine.models.player import Color
from cle.harness.context import PlayerResponseParseError, PlayerResponseParser
from cle.harness.models import ModelResponse
from cle.harness.suite import default_suite_path, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.contracts import (
    CommitmentProposal,
    CommunicationChoice,
    CommunicationMode,
    TalkContext,
)
from cle.sandbox import CatanSandbox
from cle.sandbox.catan import PostActionCommunicationError
from cle.sandbox.communication import CommunicationPolicy

from .support import COLORS, BoundaryDefect, LocalTransport, QuietCommunication, open_root


@pytest.mark.asyncio
async def test_commented_message_must_not_publish_private_draft(sandbox: CatanSandbox) -> None:
    draft: Any = "PRIVATE DRAFT: I have three ORE."
    transport = LocalTransport(
        [
            f"<!-- <message>{draft}</message> -->"
            "<message>SILENCE</message><audience>PUBLIC</audience><intent>TRADE</intent>"
        ]
    )
    sandbox.register_player(AgentPlayer(
        Color.BLUE, transport, session_id="audit:BLUE", suite=load_context_suite(),
    ))
    sandbox.communication_policy = CommunicationPolicy()

    result: Any = await sandbox.step()

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
def test_malformed_or_conflicting_action_xml_must_be_rejected(sandbox: CatanSandbox, text: tuple[str]) -> None:
    context: Any = sandbox.decision_context()
    suite: Any = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))
    assert len(context.legal_actions) > 2
    try:
        choice: Any = PlayerResponseParser(suite).parse(
            context, ModelResponse(content=text)
        )
    except PlayerResponseParseError:
        return
    assert choice.action_index == 1
    assert sandbox.game_engine.is_action_valid(context.action_at(choice.action_index))
    raise BoundaryDefect("Malformed or conflicting XML selected legal action 1")


@pytest.mark.asyncio
async def test_invalid_commitment_must_not_poison_next_step(sandbox: CatanSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    async def communicate(context: TalkContext) -> CommunicationChoice:
        return CommunicationChoice(
            mode=CommunicationMode.SAY,
            text="Trade?",
            audience=(Color.RED,),
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


@pytest.mark.parametrize("field_name", ["trade_window", "buildings_dict"])
def test_observation_mutation_must_not_change_engine(trade_sandbox: CatanSandbox, field_name: str) -> None:
    engine: Any = trade_sandbox.game_engine
    root: Any = open_root(trade_sandbox)
    context: Any = trade_sandbox.decision_context()
    before = engine.revision
    before_engine = pickle.dumps(engine.snapshot())
    if field_name == "trade_window":
        original_give: Any = tuple(engine.state.trade_window.offers[root.id].give)
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
