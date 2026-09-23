"""Legacy synchronous policy players kept outside the game engine."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Callable, Iterable

from cle.game_engine.models.enums import Action
from cle.game_engine.models.player import Color

if TYPE_CHECKING:
    from cle.game_engine.game import GameEngine


class Player:
    """Legacy synchronous policy interface used by compatibility environments."""

    def __init__(self, color: Color, is_bot: bool = True) -> None:
        self.color = color
        self.is_bot = is_bot

    def decide(self, game: GameEngine, playable_actions: Iterable[Action]) -> Action:
        raise NotImplementedError

    def reset_state(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}:{self.color.value}"


class SimplePlayer(Player):
    """Legacy policy that always chooses the first legal action."""

    def decide(self, game: GameEngine, playable_actions: Iterable[Action]) -> Action:
        return next(iter(playable_actions))


class HumanPlayer(Player):
    """Legacy terminal-input policy."""

    def __init__(
        self,
        color: Color,
        is_bot: bool = False,
        input_fn: Callable[[str], str] = builtins.input,
    ) -> None:
        super().__init__(color, is_bot)
        self.input_fn = input_fn

    def decide(self, game: GameEngine, playable_actions: Iterable[Action]) -> Action:
        actions = tuple(playable_actions)
        for index, action in enumerate(actions):
            print(f"{index}: {action.action_type} {action.value}")
        selected = None
        while selected is None or selected < 0 or selected >= len(actions):
            try:
                selected = int(self.input_fn(">>> "))
            except ValueError:
                selected = None
        return actions[selected]
