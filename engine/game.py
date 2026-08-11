"""
Contains Game class which is a thin-wrapper around the State class.
"""

import uuid
import random
import sys
from typing import Sequence, Union, Optional

from engine.models.enums import Action, ActionPrompt, ActionType
from engine.state import State, apply_action, assert_forced_action_is_explicit
from engine.state_functions import player_key, player_has_rolled
from engine.models.map import CatanMap
from engine.models.player import Color, Player

# To timeout RandomRobots from getting stuck...
TURNS_LIMIT = 1000


def is_valid_action(state, action):
    """True if its a valid action right now. An action is valid
    if its in playable_actions or if its a OFFER_TRADE/COUNTER_OFFER in the right time."""
    if action.action_type == ActionType.OFFER_TRADE:
        return (
            state.current_color() == action.color
            and state.current_prompt == ActionPrompt.PLAY_TURN
            and player_has_rolled(state, action.color)
            and is_valid_trade(action.value)
        )

    if action.action_type == ActionType.COUNTER_OFFER:
        # Counter-offers valid in legacy DECIDE_TRADE or async PLAY_TURN (if there are active trades)
        valid_prompt = (
            state.current_prompt == ActionPrompt.DECIDE_TRADE
            or (state.current_prompt == ActionPrompt.PLAY_TURN and len(state.active_trades) > 0)
        )
        return (
            action.color != state.colors[state.current_turn_index]  # Not the turn player
            and valid_prompt
            and is_valid_trade(action.value)
        )

    return action in state.playable_actions


def is_valid_trade(action_value):
    """Checks the value of a OFFER_TRADE does not
    give away resources or trade matching resources.
    """
    # Meta-actions have string values (format descriptions) - not valid trades
    if not isinstance(action_value, (tuple, list)) or len(action_value) < 10:
        return False

    offering = action_value[:5]
    asking = action_value[5:10]
    if sum(offering) == 0 or sum(asking) == 0:
        return False  # cant give away cards

    for i, j in zip(offering, asking):
        if i > 0 and j > 0:
            return False  # cant trade same resources
    return True


class Game:
    """
    Initializes a map, decides player seating order, and exposes two main
    methods for executing the game (play and play_tick; to advance until
    completion or just by one decision by a player respectively).
    """

    def __init__(
        self,
        players: Sequence[Player],
        seed: Optional[int] = None,
        discard_limit: int = 7,
        vps_to_win: int = 10,
        catan_map: Optional[CatanMap] = None,
        initialize: bool = True,
        shuffle_players: bool = True,
    ):
        """Creates a game (doesn't run it).

        Args:
            players (List[Player]): list of players, should be at most 4.
            seed (int, optional): Random seed to use (for reproducing games). Defaults to None.
            discard_limit (int, optional): Discard limit to use. Defaults to 7.
            vps_to_win (int, optional): Victory Points needed to win. Defaults to 10.
            catan_map (CatanMap, optional): Map to use. Defaults to None.
            initialize (bool, optional): Whether to initialize. Defaults to True.
            shuffle_players (bool, optional): Whether to shuffle player order. Defaults to True.
        """
        if initialize:
            self.seed = seed if seed is not None else random.randrange(sys.maxsize)
            random.seed(self.seed)

            self.id = str(uuid.uuid4())
            self.vps_to_win = vps_to_win
            self.state = State(players, catan_map, discard_limit=discard_limit, shuffle_players=shuffle_players)
            self.history = []  # Stack of (state_copy, action) for undo

    def play(self, decide_fn=None):
        """Executes game until a player wins or exceeded TURNS_LIMIT.

        Args:
            decide_fn (function, optional): Function to overwrite current player's decision with.
                Defaults to None.
        Returns:
            Color: winning color or None if game exceeded TURNS_LIMIT
        """
        while self.winning_color() is None and self.state.num_turns < TURNS_LIMIT:
            self.play_tick(decide_fn=decide_fn)
        return self.winning_color()

    def play_tick(self, decide_fn=None):
        """Advances game by one ply (player decision).

        Args:
            decide_fn (function, optional): Function to overwrite current player's decision with.
                Defaults to None.

        Returns:
            Action: Final action (modified to be used as Log)
        """
        player = self.state.current_player()
        actions = self.state.playable_actions

        action = (
            decide_fn(player, self, actions)
            if decide_fn is not None
            else player.decide(self, actions)
        )
        return self.execute(action)

    def execute(
        self,
        action: Action,
        validate_action: bool = True,
        save_history: bool = True,
        force: bool = False,
    ) -> Action:
        """Internal call that carries out decided action by player"""
        if force:
            assert_forced_action_is_explicit(action)
            validate_action = False

        if validate_action and not is_valid_action(self.state, action):
            raise ValueError(
                f"{action} not playable right now. playable_actions={self.state.playable_actions}"
            )

        # Save state before action for undo
        if save_history and hasattr(self, 'history'):
            self.history.append((self.state.copy(), action))

        return apply_action(self.state, action, force=force)

    def undo(self) -> Optional[Action]:
        """Undo the last action, restoring the previous game state.

        Returns:
            Action: The action that was undone, or None if no history.
        """
        if not hasattr(self, 'history') or not self.history:
            return None

        prev_state, action = self.history.pop()
        self.state = prev_state
        return action

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return hasattr(self, 'history') and len(self.history) > 0

    def winning_color(self) -> Union[Color, None]:
        """Gets winning color

        Returns:
            Union[Color, None]: Might be None if game truncated by TURNS_LIMIT
        """
        result = None
        for color in self.state.colors:
            key = player_key(self.state, color)
            if (
                self.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
                >= self.vps_to_win
            ):
                result = color

        return result

    def copy(self) -> "Game":
        """Creates a copy of this Game, that can be modified without
        repercusions on this one (useful for simulations).

        Returns:
            Game: Game copy.
        """
        game_copy = Game(players=[], initialize=False)
        game_copy.seed = self.seed
        game_copy.id = self.id
        game_copy.vps_to_win = self.vps_to_win
        game_copy.state = self.state.copy()
        game_copy.history = [(s.copy(), a) for s, a in self.history] if hasattr(self, 'history') else []
        return game_copy
