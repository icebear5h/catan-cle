from dataclasses import dataclass, field

import pytest

from cle.harness import ModelResponse, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox import CatanSandbox
from game_engine.game import GameEngine
from game_engine.models.player import Color


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _response(action=0, plan="expand toward wheat"):
    return ModelResponse(
        content=(
            f"<game_plan>{plan}</game_plan>"
            "<rationale>take an exact legal action</rationale>"
            f"<action>{action}</action>"
        ),
        model="test/model",
        native_reasoning="native analysis",
        native_reasoning_details=({"type": "reasoning.text"},),
        reasoning_request=(("effort", "xhigh"), ("exclude", False)),
        provider_response_id="gen-context-test",
        provider_request_id="req-context-test",
        provider_native_finish_reason="stop",
    )


@dataclass
class FixedTransport:
    responses: list[ModelResponse]
    requests: list = field(default_factory=list)

    async def complete(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


class NoCommunicationPolicy:
    def pre_action(self, engine):
        return ()

    def after_events(self, engine, events, *, round_number):
        return ()


def _sandbox_and_player(responses):
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    transport = FixedTransport(list(responses))
    red = AgentPlayer(Color.RED, transport, session_id=f"{engine.id}:RED")
    players = {Color.RED: red}
    players.update({color: FirstLegalPlayer(color) for color in COLORS[1:]})
    return (
        CatanSandbox(engine, players, communication_policy=NoCommunicationPolicy()),
        red,
        transport,
    )


def test_default_yaml_suite_is_strict_and_separately_loadable():
    first = load_context_suite()
    second = load_context_suite()

    assert first == second
    assert first is not second
    assert first.id == "catan-agent"
    assert first.version == "5.0.0"
    assert first.context.order == (
        "trajectory",
        "strategic_memory",
        "game_events",
        "observation",
        "phase_guidance",
        "legal_actions",
    )
    assert first.context.trajectory.max_messages is None

    authored_text = "\n".join(
        (
            first.system.template,
            first.response.instruction,
            *first.phase_guidance.values(),
        )
    ).lower()
    assert len(authored_text.split()) < 200
    for prescriptive_phrase in (
        "you are",
        "expert",
        "prioritize",
        "prefer",
        "think in sequences",
        "cities > settlements",
        "block the leading opponent",
    ):
        assert prescriptive_phrase not in authored_text


@pytest.mark.asyncio
async def test_agent_player_keeps_full_conversation_and_full_visible_events():
    sandbox, player, transport = _sandbox_and_player(
        [_response(), _response(plan="connect the opening road"), _response()]
    )

    first = await sandbox.step()
    second = await sandbox.step()

    assert first.transitions[0].resolved_action == first.contexts[0].legal_actions[0]
    assert second.transitions[0].resolved_action == second.contexts[0].legal_actions[0]
    assert first.attempts[0].choice.rationale == "take an exact legal action"
    assert first.attempts[0].choice.native_reasoning == "native analysis"
    assert first.attempts[0].choice.native_reasoning_details == (
        {"type": "reasoning.text"},
    )
    assert dict(first.attempts[0].choice.reasoning_request) == {
        "effort": "xhigh",
        "exclude": False,
    }
    assert first.attempts[0].choice.provider_response_id == "gen-context-test"
    assert first.attempts[0].choice.provider_request_id == "req-context-test"
    assert first.attempts[0].choice.provider_native_finish_reason == "stop"
    first_system = transport.requests[0].messages[0].content
    first_user = transport.requests[0].messages[-1].content
    assert "Select the next action for RED." in first_system
    assert "You are" not in first_system
    assert "expert" not in first_system.lower()
    assert "prioritize" not in first_system.lower()
    assert len(first_system.split()) < 150
    assert "DECISION FACTS:" in first_user
    assert "first initial settlement" in first_user
    assert first_user.count("VALID ACTIONS:") == 1
    assert "<valid_actions>" not in first_user
    assert player.session.strategic_memory == "connect the opening road"
    assert [message.role for message in player.session.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message.role for message in transport.requests[1].messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "0. RED: BUILD_SETTLEMENT" in transport.requests[1].messages[-1].content

    while sandbox.current_actor() != Color.RED:
        await sandbox.step()
    await sandbox.step()

    third_user = transport.requests[2].messages[-1].content
    assert "0. RED: BUILD_SETTLEMENT" in third_user
    assert "1. RED: BUILD_ROAD" in third_user


@pytest.mark.asyncio
async def test_sandbox_snapshot_restores_private_player_conversation():
    sandbox, player, _ = _sandbox_and_player([_response(), _response()])
    await sandbox.step()
    snapshot = sandbox.snapshot()

    await sandbox.step()
    assert len(player.session.messages) == 4

    sandbox.restore(snapshot)

    assert len(player.session.messages) == 2
    assert player.session.strategic_memory == "expand toward wheat"
    assert len(player.session.receipts) == 1


@pytest.mark.asyncio
async def test_sandbox_retries_invalid_player_output_with_feedback():
    sandbox, _, transport = _sandbox_and_player(
        [ModelResponse(content="<action>999</action>"), _response()]
    )

    result = await sandbox.step()

    assert result.after_revision == 1
    assert len(transport.requests) == 2
    assert "CORRECTION FROM THE SANDBOX" in transport.requests[1].messages[-1].content
    assert len(sandbox.decision_trace) == 1
