"""Shared helpers for knight tool bundling, retries, and replay preview."""
import pickle
from collections.abc import Sequence
from typing import Any

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import build_settlement
from cle.harness.models import ModelRequest
from cle.harness.suite import load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    AcceptanceResult,
    CommunicationChoice,
    PlayerAttempt,
    PlayerChoice,
    PlayerContext,
    TalkContext,
)
from cle.sandbox import CatanSandbox

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


KNIGHT = Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None)


class _TypedAgent(AgentPlayer):
    """Exercise real agent receipts with typed decisions, never a provider call."""

    def __init__(
        self,
        engine: GameEngine,
        choices: Sequence[PlayerChoice] = (),
        color: Color = Color.RED,
    ) -> None:
        super().__init__(color, transport=self, session_id=f"knight-test:{color.value}", suite=load_context_suite())
        self.engine = engine
        self.choices = list(choices)
        self.calls: list[tuple[PlayerContext, str | None, bytes]] = []
        self.accepted: list[tuple[PlayerAttempt, AcceptanceResult]] = []

    async def complete(self, request: ModelRequest) -> None:
        raise AssertionError("Typed test player must not invoke a model")

    async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
        self.calls.append((context, feedback, pickle.dumps(self.engine.snapshot())))
        assert self.choices, "Unexpected extra choose call"
        return PlayerAttempt(context.context_id, self.choices.pop(0))

    async def communicate(self, context: TalkContext) -> CommunicationChoice:
        return CommunicationChoice()

    def accept(self, attempt: PlayerAttempt, result: AcceptanceResult) -> None:
        assert self.engine.revision >= result.after_revision
        if attempt.choice.knight_destination is not None:
            assert self.engine.revision == result.after_revision
        self.accepted.append((attempt, result))
        super().accept(attempt, result)


def _knight_engine(
    *, rolled: bool = False, victims: int = 0, victory: bool = False
) -> tuple[GameEngine, str]:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False, capture_history=True)
    state = engine.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.player_state["P0_HAS_ROLLED"] = rolled
    state.player_state["P0_KNIGHT_IN_HAND"] = 1
    state.player_state["P0_KNIGHT_OWNED_AT_START"] = True
    state.development_listdeck.remove("KNIGHT")
    if victory:
        # Reduced score boundary: the third Knight awards the last two VP.
        state.player_state["P0_PLAYED_KNIGHT"] = 2
        state.player_state["P0_VICTORY_POINTS"] = 8
        state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 8
        state.development_listdeck.remove("KNIGHT")
        state.development_listdeck.remove("KNIGHT")
    destination = next(
        coordinate
        for coordinate in state.board.map.land_tiles
        if coordinate != state.board.robber_coordinate
    )
    tile = state.board.map.land_tiles[destination]
    for index, color in enumerate(COLORS[1 : 1 + victims], start=1):
        node = next(
            node
            for node in tile.nodes.values()
            if node in state.board.buildable_node_ids(color, initial_build_phase=True)
        )
        state.board.build_settlement(color, node, initial_build_phase=True)
        build_settlement(state, color, node, is_free=True)
        state.player_state[f"P{index}_WOOD_IN_HAND"] = 1
        state.player_state[f"P{index}_ORE_IN_HAND"] = 1
        state.resource_freqdeck[0] -= 1
        state.resource_freqdeck[4] -= 1
    state.playable_actions = generate_playable_actions(state)
    assert KNIGHT in state.playable_actions
    return engine, destination


def _sandbox(
    engine: GameEngine, choices: Sequence[PlayerChoice] = ()
) -> tuple[CatanSandbox, _TypedAgent]:
    red = _TypedAgent(engine, choices)
    players: Any = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = red
    return CatanSandbox(engine, players), red


def _choice(engine: GameEngine, destination: str | None) -> PlayerChoice:
    return PlayerChoice(
        engine.state.playable_actions.index(KNIGHT),
        knight_destination=destination,
        game_plan="Move the robber with this Knight.",
        raw_response="typed Knight decision",
        native_reasoning="One decision selects the card and destination.",
        provider_response_id="local-knight",
    )
