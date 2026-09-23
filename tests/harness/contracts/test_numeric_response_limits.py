import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from cle.game_engine.events import GameEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import player_freqdeck_add
from cle.harness import ModelResponse, PlayerResponseParseError, PlayerResponseParser
from cle.harness.communication import parse_communication_response
from cle.harness.models import ModelRequest
from cle.harness.response_xml import parse_response_fields
from cle.harness.suite import default_suite_path, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import CommunicationChoice, CommunicationMode
from cle.sandbox import CatanSandbox, PlayerResponseError, RetryPolicy
from cle.sandbox.communication import CommunicationOpportunity

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


@dataclass
class QueuedTransport:
    responses: list[ModelResponse] = field(default_factory=list)
    requests: list = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class NoCommunicationPolicy:
    def pre_action(self, engine: GameEngine) -> tuple[CommunicationOpportunity, ...]:
        return ()

    def after_events(self, engine: GameEngine, events: tuple[GameEvent, ...], *, round_number: int) -> tuple[CommunicationOpportunity, ...]:
        return ()


@pytest.fixture
def oversized_integer() -> str:
    digits = "9" * 5000
    # Exercise the interpreter's real guard without changing its process-wide limit.
    with pytest.raises(ValueError, match="limit"):
        int(digits)
    return digits


@pytest.fixture(params=[
    "trade_give", "trade_receive", "trade_give_any", "trade_receive_any",
    "discard", "xml_index", "named_action_index", "named_move_index", "plain_index",
])
def response_case(
    request: pytest.FixtureRequest, oversized_integer: str
) -> tuple[CatanSandbox, AgentPlayer, QueuedTransport, str, str]:
    kind = request.param
    engine: Any = GameEngine(COLORS, seed=7, shuffle_players=False)
    transport = QueuedTransport()
    player = AgentPlayer(
        Color.RED, transport, session_id=f"{engine.id}:RED",
        suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
    )
    players: Any = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = player
    sandbox = CatanSandbox(
        engine, players, retry_policy=RetryPolicy(2),
        communication_policy=NoCommunicationPolicy(),
    )
    if kind.startswith("trade_") or kind == "discard":
        for _ in range(16):
            engine.step(engine.state.playable_actions[0])
        bundle = (4, 0, 0, 0, 4)
        player_freqdeck_add(engine.state, Color.RED, bundle)
        for index, count in enumerate(bundle):
            engine.state.resource_freqdeck[index] -= count
        dice = (3, 4) if kind == "discard" else (1, 1)
        engine.step(Action(Color.RED, ActionType.ROLL, dice), force=True)
        engine.state.playable_actions = generate_playable_actions(engine.state)
    context: Any = sandbox.decision_context()
    if kind.startswith("trade_"):
        index = next(
            index for index, action in enumerate(context.legal_actions)
            if action.action_type == ActionType.OFFER_TRADE
        )
        payload: Any = {"give": {"WOOD": 1}, "receive": {"ORE": 1}}
        valid = f"<action>{index}</action><trade_offer>{json.dumps(payload)}</trade_offer>"
        field_name: Any = kind.removeprefix("trade_")
        if field_name in {"give", "receive"}:
            resource: Any = "WOOD" if field_name == "give" else "ORE"
            payload[field_name][resource] = oversized_integer
        else:
            payload[field_name] = oversized_integer
        invalid_json = json.dumps(payload).replace(f'"{oversized_integer}"', oversized_integer)
        invalid = f"<action>{index}</action><trade_offer>{invalid_json}</trade_offer>"
    elif kind == "discard":
        assert context.legal_actions[0].action_type == ActionType.DISCARD
        payload = {"WOOD": 2, "ORE": context.discard_count - 2}
        valid = f"<action>0</action><discard>{json.dumps(payload)}</discard>"
        invalid = '<action>0</action><discard>{"WOOD":' + oversized_integer + "}</discard>"
    else:
        valid = "<action>0</action>"
        invalid = {
            "xml_index": f"<action>{oversized_integer}</action>",
            "named_action_index": f"action_index: {oversized_integer}",
            "named_move_index": f"move_index = {oversized_integer}",
            "plain_index": oversized_integer,
        }[kind]
    assert 5000 <= len(invalid) < 6000
    # XML admission succeeds: this is a numeric limit, not the response-size cap.
    parse_response_fields(invalid)
    return sandbox, player, transport, invalid, valid


@pytest.mark.filterwarnings("ignore:Prompt suite .* is deprecated:DeprecationWarning")
def test_oversized_numeric_fields_raise_controlled_parse_errors(response_case: tuple[CatanSandbox, AgentPlayer, QueuedTransport, str, str]) -> None:
    sandbox, player, _, invalid, _ = response_case

    with pytest.raises(PlayerResponseParseError) as caught:
        PlayerResponseParser(player.suite).parse(
            sandbox.decision_context(), ModelResponse(content=invalid)
        )

    assert isinstance(caught.value.__cause__, ValueError)
    assert not isinstance(caught.value.__cause__, json.JSONDecodeError)
    assert "limit" in str(caught.value.__cause__)


@pytest.mark.filterwarnings("ignore:Prompt suite .* is deprecated:DeprecationWarning")
@pytest.mark.asyncio
async def test_numeric_rejection_retries_once_and_traces_first_validation(response_case: tuple[CatanSandbox, AgentPlayer, QueuedTransport, str, str]) -> None:
    sandbox, player, transport, invalid, valid = response_case
    transport.responses.extend([ModelResponse(content=invalid), ModelResponse(content=valid)])
    before = sandbox.revision

    result: Any = await sandbox.step()

    assert sandbox.revision == before + 1
    assert len(transport.requests) == 2
    assert transport.responses == []
    assert len(sandbox.decision_trace) == 1
    rejected: Any = sandbox.decision_trace[0]
    assert rejected.choice is None
    assert rejected.validation_error
    assert rejected.model_response.content == invalid
    assert rejected.model_request == transport.requests[0]
    assert rejected.validation_error in transport.requests[1].messages[-1].content
    assert "CORRECTION FROM THE SANDBOX" in transport.requests[1].messages[-1].content
    assert result.attempts[0].validation_error is None
    assert result.attempts[0].choice.raw_response == valid
    assert len(player.session.receipts) == 1
    assert [message.role for message in player.session.messages] == ["user", "assistant"]
    assert player.session.messages[-1].content == valid


@pytest.mark.filterwarnings("ignore:Prompt suite .* is deprecated:DeprecationWarning")
@pytest.mark.asyncio
async def test_numeric_rejection_respects_retry_budget_and_leaves_second_response_queued(response_case: tuple[CatanSandbox, AgentPlayer, QueuedTransport, str, str]) -> None:
    sandbox, player, transport, invalid, valid = response_case
    sandbox.retry_policy = RetryPolicy(1)
    transport.responses.extend([ModelResponse(content=invalid), ModelResponse(content=valid)])
    before = sandbox.revision

    with pytest.raises(PlayerResponseError) as caught:
        await sandbox.step()

    assert sandbox.revision == before
    assert len(transport.requests) == 1
    assert [response.content for response in transport.responses] == [valid]
    assert len(caught.value.attempts) == 1
    assert caught.value.attempts[0] == sandbox.decision_trace[0]
    assert caught.value.attempts[0].validation_error
    assert caught.value.attempts[0].model_response.content == invalid
    assert player.session.messages == []
    assert not player.session.receipts


@pytest.mark.filterwarnings("ignore:Prompt suite .* is deprecated:DeprecationWarning")
@pytest.mark.parametrize("response_case", ["trade_give", "discard"], indirect=True)
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_json_callback_errors_are_not_normalized_as_numeric_rejections(
    response_case: tuple[CatanSandbox, AgentPlayer, QueuedTransport, str, str],
    error_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox, player, _, _, valid = response_case
    error = error_type("unrelated object callback failure")
    original_loads = json.loads

    def fail_object(pairs: list[tuple[str, object]]) -> None:
        raise error

    def decode_with_callback(text: str, **kwargs: object) -> object:
        kwargs["object_pairs_hook"] = fail_object
        return original_loads(text, **kwargs)

    monkeypatch.setattr("cle.harness.context.json.loads", decode_with_callback)

    with pytest.raises(error_type) as caught:
        PlayerResponseParser(player.suite).parse(
            sandbox.decision_context(), ModelResponse(content=valid)
        )

    assert caught.value is error


def test_oversized_communication_commitment_turn_fails_closed(oversized_integer: str) -> None:
    prefix = (
        "<message>Leave BLUE alone and I will offer ORE.</message>"
        "<audience>BLUE</audience><intent>BRIBE</intent>"
        "<commitment_condition>RED avoids BLUE</commitment_condition>"
        "<commitment_promise>BLUE offers ORE</commitment_promise>"
    )
    invalid = prefix + f"<commitment_expires_turn>{oversized_integer}</commitment_expires_turn>"
    assert len(invalid) < 6000
    parse_response_fields(invalid)

    assert parse_communication_response(
        ModelResponse(content=invalid), speaker=Color.RED, participants=COLORS
    ) == CommunicationChoice()
    valid: Any = parse_communication_response(
        ModelResponse(content=prefix + "<commitment_expires_turn>3</commitment_expires_turn>"),
        speaker=Color.RED, participants=COLORS,
    )
    assert valid.mode == CommunicationMode.SAY
    assert valid.commitment.expires_turn == 3


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_communication_commitment_callback_errors_propagate(
    error_type: type[BaseException], monkeypatch: pytest.MonkeyPatch
) -> None:
    error = error_type("unrelated commitment callback failure")

    def fail_commitment(*args: object) -> None:
        raise error

    monkeypatch.setattr("cle.harness.communication.CommitmentProposal", fail_commitment)
    response = ModelResponse(content=(
        "<message>Leave BLUE alone and I will offer ORE.</message>"
        "<audience>BLUE</audience><intent>BRIBE</intent>"
        "<commitment_condition>RED avoids BLUE</commitment_condition>"
        "<commitment_promise>BLUE offers ORE</commitment_promise>"
        "<commitment_expires_turn>3</commitment_expires_turn>"
    ))

    with pytest.raises(error_type) as caught:
        parse_communication_response(response, speaker=Color.RED, participants=COLORS)

    assert caught.value is error
