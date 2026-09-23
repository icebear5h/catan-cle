"""Shared helpers for catan sandbox stepping, trading barriers, speech, and discards."""

from dataclasses import dataclass, field
from typing import Any

from cle.game_engine.events import GameEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import ActionPrompt
from cle.game_engine.models.player import Color
from cle.game_engine.models.trade import ResourceBundle
from cle.game_engine.trading import TradeLimits, TradeOffer
from cle.harness.models import ModelRequest, ModelResponse
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    PlayerAttempt,
    PlayerChoice,
    PlayerContext,
    SandboxPlayer,
)
from cle.sandbox import CatanSandbox
from cle.sandbox.communication import CommunicationOpportunity

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _trade_offer(
    give: ResourceBundle = (1, 0, 0, 0, 0), receive: ResourceBundle = (0, 0, 0, 0, 1)
) -> TradeOffer:
    return TradeOffer(
        offered_by=Color.RED,
        audience=frozenset(COLORS[1:]),
        give=give,
        receive=receive,
    )


def _sandbox(
    red: SandboxPlayer | None = None,
) -> tuple[CatanSandbox, dict[Color, Any]]:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    players = {
        color: (red if color == Color.RED and red is not None else FirstLegalPlayer(color))
        for color in COLORS
    }
    return CatanSandbox(engine, players), players


@dataclass
class InvalidThenValidPlayer(FirstLegalPlayer):
    attempts: int = 0

    async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
        self.attempts += 1
        index = 999 if self.attempts == 1 else 0
        return PlayerAttempt(
            context_id=context.context_id,
            choice=PlayerChoice(action_index=index),
        )


class NoCommunicationPolicy:
    def pre_action(self, engine: GameEngine) -> tuple[CommunicationOpportunity, ...]:
        return ()

    def after_events(
        self,
        engine: GameEngine,
        events: tuple[GameEvent, ...],
        *,
        round_number: int,
    ) -> tuple[CommunicationOpportunity, ...]:
        return ()


@dataclass
class FixedTransport:
    responses: list[ModelResponse]
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def _trade_engine(limits: TradeLimits | None = None) -> GameEngine:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False, trade_limits=limits)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.player_state["P0_HAS_ROLLED"] = True
    engine.state.player_state["P0_WOOD_IN_HAND"] = 8
    for index in range(1, 4):
        engine.state.player_state[f"P{index}_ORE_IN_HAND"] = 1
    engine.state.playable_actions = generate_playable_actions(engine.state)
    return engine


def _discarding_engine(hands: dict[int, int]) -> GameEngine:
    """A rolled 7 whose over-limit seats still owe a discard."""
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    state = engine.state
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.DISCARD
    state.is_discarding = True
    for index, wood in hands.items():
        state.player_state[f"P{index}_WOOD_IN_HAND"] = wood
        state.resource_freqdeck[0] -= wood
    state.current_player_index = min(hands)
    state.playable_actions = generate_playable_actions(state)
    return engine
