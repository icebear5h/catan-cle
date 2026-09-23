"""Concrete scripted providers and an independent strict-engine setup oracle."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, replace

from cle.game_engine.board_tokens import edge_token, node_token
from cle.game_engine.events import GameEngineSnapshot
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import RESOURCES, Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.observation import PlayerObservation
from cle.harness.models import ModelRequest, ModelResponse
from cle.players.agent import AgentPlayer
from cle.sandbox import CatanSandbox, RetryPolicy
from cle.sandbox.contracts import SandboxSnapshot, SandboxStepResult
from cle.traces.journal import CommandRecord, SQLiteSandboxJournal

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
POLICY = "durable-integration-policy-v1"


class ScriptedTransport:
    """Fail on unexpected calls; optional gates control real task interleavings."""

    def __init__(self, *replies: str | Exception, gated: bool = False) -> None:
        self.replies = deque(replies)
        self.requests: list[ModelRequest] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.completed = asyncio.Event()
        if not gated:
            self.release.set()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        self.entered.set()
        await self.release.wait()
        assert self.replies, f"Unexpected inference: {request.session_id}/{request.channel}"
        reply = self.replies.popleft()
        if isinstance(reply, Exception):
            raise reply
        self.completed.set()
        return ModelResponse(
            reply,
            provider_response_id=f"{request.session_id}:response:{len(self.requests)}",
            provider_request_id=f"{request.session_id}:request:{len(self.requests)}",
            usage=(("prompt_tokens", 17), ("completion_tokens", 5)),
        )


@dataclass(frozen=True)
class SetupPair:
    before: GameEngineSnapshot
    settlement: Action
    road: Action
    after_settlement: GameEngineSnapshot
    after_road: GameEngineSnapshot

    def response(self) -> str:
        node: object = self.settlement.value
        edge: object = self.road.value
        assert isinstance(node, int)
        assert isinstance(edge, tuple) and len(edge) == 2
        assert isinstance(edge[0], int) and isinstance(edge[1], int)
        return json.dumps({
            "actions": [
                {"tool": "build_settlement", "arguments": {"node": node_token(node)}},
                {"tool": "build_road", "arguments": {"edge": edge_token((edge[0], edge[1]))}},
            ],
            "notes": "Keep the admitted setup pair",
        })


def new_engine() -> GameEngine:
    return GameEngine(COLORS, seed=7, shuffle_players=False)


def setup_pair(pair_index: int = 0) -> SetupPair:
    """Select and strictly execute both actions without reading a model prompt."""
    oracle = new_engine()
    for _ in range(pair_index * 2):
        oracle.step(oracle.state.playable_actions[0])
    before = oracle.snapshot()
    settlement = oracle.state.playable_actions[0]
    assert settlement.action_type == ActionType.BUILD_SETTLEMENT
    oracle.step(settlement)
    after_settlement = oracle.snapshot()
    road = oracle.state.playable_actions[0]
    assert road.action_type == ActionType.BUILD_ROAD
    oracle.step(road)
    return SetupPair(before, settlement, road, after_settlement, oracle.snapshot())


def seven_engine(*, discards: bool) -> GameEngine:
    """Legal setup and roll around a reduced, supply-conserving hand fixture."""
    engine = new_engine()
    while engine.state.is_initial_build_phase:
        engine.step(engine.state.playable_actions[0])
    for resource_index, resource in enumerate(RESOURCES):
        total = 0
        for seat in range(4):
            count = {0: 8, 2: 10}.get(seat, 0) if discards and resource == "WOOD" else 0
            engine.state.player_state[f"P{seat}_{resource}_IN_HAND"] = count
            total += count
        engine.state.resource_freqdeck[resource_index] = 19 - total
    engine.state.playable_actions = generate_playable_actions(engine.state)
    engine.rng.seed(1)  # Real RNG yields (2, 5); no forced engine actions.
    transition = engine.step(Action(Color.RED, ActionType.ROLL, None))
    assert transition.resolved_action.value == (2, 5)
    assert engine.state.current_prompt == (
        ActionPrompt.DISCARD if discards else ActionPrompt.MOVE_ROBBER
    )
    return engine


def make_sandbox(
    engine_snapshot: GameEngineSnapshot,
    transports: dict[Color, ScriptedTransport] | None = None,
) -> tuple[CatanSandbox, dict[Color, ScriptedTransport]]:
    engine = new_engine()
    engine.restore(engine_snapshot)
    scripts = {color: ScriptedTransport() for color in COLORS}
    scripts.update(transports or {})
    players = {
        color: AgentPlayer(color, scripts[color], session_id=f"durable:{color.value}")
        for color in COLORS
    }
    return CatanSandbox(engine, players, retry_policy=RetryPolicy(1)), scripts


def fresh_sandbox(
    journal: SQLiteSandboxJournal, sandbox_id: str,
    transports: dict[Color, ScriptedTransport] | None = None,
) -> tuple[CatanSandbox, dict[Color, ScriptedTransport]]:
    # The new runner, engine, players, transports, and store share only disk state.
    return make_sandbox(journal.head(sandbox_id).snapshot.engine, transports)


def successful(record: CommandRecord) -> SandboxStepResult:
    assert record.finished_sequence is not None
    assert record.outcome is not None
    assert record.outcome.status == "succeeded", record.outcome
    assert record.outcome.result is not None
    return record.outcome.result


def agent(sandbox: CatanSandbox, color: Color) -> AgentPlayer:
    player = sandbox.players[color]
    assert isinstance(player, AgentPlayer)
    return player


def assert_observation(actual: PlayerObservation, expected: PlayerObservation) -> None:
    # CatanMap has identity equality. Compare every map field before comparing
    # the remaining dataclass fields, rather than dropping board evidence.
    assert vars(actual.board_map) == vars(expected.board_map)
    assert replace(actual, board_map=expected.board_map) == expected


def assert_result(actual: SandboxStepResult, expected: SandboxStepResult) -> None:
    for context, reference in zip(actual.contexts, expected.contexts, strict=True):
        assert_observation(context.observation, reference.observation)
        assert replace(context, observation=reference.observation) == reference
    assert replace(actual, contexts=expected.contexts) == expected


def assert_engine(engine: GameEngine, expected: GameEngineSnapshot) -> None:
    oracle = new_engine()
    oracle.restore(expected)
    assert engine.id == expected.engine_id
    assert tuple(engine.events) == expected.events
    assert engine.state.actions == oracle.state.actions
    assert engine.state.playable_actions == oracle.state.playable_actions
    assert engine.state.player_state == oracle.state.player_state
    assert engine.state.resource_freqdeck == oracle.state.resource_freqdeck
    assert engine.state.development_listdeck == oracle.state.development_listdeck
    assert engine.rng.getstate() == oracle.rng.getstate()
    for color in COLORS:
        assert_observation(engine.observe(color), oracle.observe(color))


def assert_checkpoint(sandbox: CatanSandbox, expected: SandboxSnapshot) -> None:
    actual = sandbox.snapshot()
    assert_engine(sandbox.game_engine, expected.engine)
    assert actual.player_states == expected.player_states
    assert actual.pending_action_batch == expected.pending_action_batch
    assert actual.pending_decision_revision == expected.pending_decision_revision
    assert actual.pending_reactions == expected.pending_reactions
    assert actual.speech_used == expected.speech_used
    assert actual.speech_calls_remaining == expected.speech_calls_remaining
    assert actual.pre_robber_sequence == expected.pre_robber_sequence
    assert actual.trade_preauthorization == expected.trade_preauthorization


def entry_kinds(journal: SQLiteSandboxJournal, sandbox_id: str) -> list[str]:
    return [entry.kind for entry in journal.entries(sandbox_id)]


async def drain(*tasks: asyncio.Task[CommandRecord]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
