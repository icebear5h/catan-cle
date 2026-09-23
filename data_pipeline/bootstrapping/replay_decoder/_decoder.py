"""The `ReplayDecoder` facade that walks a replay file event by event."""

import json

from data_pipeline.bootstrapping.replay_decoder._events import apply_event
from data_pipeline.bootstrapping.replay_decoder._geometry import (
    corner_tiles,
    corners_adjacent,
    edge_connects_corner,
    edges_connected,
)
from data_pipeline.bootstrapping.replay_decoder._init_state import initialize_state
from data_pipeline.bootstrapping.replay_decoder._models import GameState, PlayerState
from data_pipeline.bootstrapping.replay_decoder._moves import valid_moves
from data_pipeline.bootstrapping.replay_decoder._placement import (
    can_afford_city,
    can_afford_dev_card,
    can_afford_road,
    can_afford_settlement,
    can_place_road,
    can_place_settlement,
)
from data_pipeline.bootstrapping.replay_decoder._snapshot import board_state, snapshot_state
from data_pipeline.bootstrapping.replay_decoder._types import (
    ActionInfo,
    BoardState,
    Move,
    ReplayStep,
    StateSnapshot,
)
from data_pipeline.json_coerce import as_dict, as_dict_list, as_int, as_list
from data_pipeline.json_types import JsonDict, JsonValue


class ReplayDecoder:
    def __init__(self, replay_path: str) -> None:
        with open(replay_path) as f:
            payload: JsonValue = json.load(f)

        self.raw_data = as_dict(payload)
        self.data = as_dict(self.raw_data['data'])
        self.event_history = as_dict(self.data['eventHistory'])
        self.events = as_dict_list(self.event_history['events'])
        self.initial_state = as_dict(self.event_history.get('initialState', {}))
        self.play_order = [as_int(color) for color in as_list(self.data.get('playOrder', []))]

        self.state = GameState()
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize game state from initialState"""
        initialize_state(self.state, self.initial_state, self.play_order)

    def apply_event(self, event: JsonDict) -> ActionInfo:
        """Apply a single event to the game state, return action info"""
        return apply_event(self.state, event)

    def get_valid_moves(self) -> list[Move]:
        """Get all valid moves for current game state"""
        return valid_moves(self.state)

    def _can_place_settlement(self, corner_id: int, is_setup: bool = False) -> bool:
        return can_place_settlement(self.state, corner_id, is_setup)

    def _can_place_road(self, edge_id: int) -> bool:
        return can_place_road(self.state, edge_id)

    def _corners_adjacent(self, c1: int, c2: int) -> bool:
        return corners_adjacent(self.state.corners[c1], self.state.corners[c2])

    def _edge_connects_corner(self, edge_id: int, corner_id: int) -> bool:
        return edge_connects_corner(self.state.edges[edge_id], self.state.corners[corner_id])

    def _edges_connected(self, e1: int, e2: int) -> bool:
        return edges_connected(self.state, e1, e2)

    def _can_afford_settlement(self, player: PlayerState) -> bool:
        return can_afford_settlement(player)

    def _can_afford_city(self, player: PlayerState) -> bool:
        return can_afford_city(player)

    def _can_afford_road(self, player: PlayerState) -> bool:
        return can_afford_road(player)

    def _can_afford_dev_card(self, player: PlayerState) -> bool:
        return can_afford_dev_card(player)

    def replay_with_states(self) -> list[ReplayStep]:
        """Replay the game and yield state + valid moves at each step"""
        results: list[ReplayStep] = []

        # Initial state
        results.append({
            'event_idx': -1,
            'state': self._snapshot_state(),
            'valid_moves': self.get_valid_moves(),
            'action_taken': None
        })

        # Apply each event
        for i, event in enumerate(self.events):
            action_info = self.apply_event(event)

            # Skip chat messages and other non-game actions
            if action_info['action_type'] is not None:
                results.append({
                    'event_idx': i,
                    'state': self._snapshot_state(),
                    'valid_moves': self.get_valid_moves(),
                    'action_taken': action_info
                })

        return results

    def _get_corner_tiles(self, corner_id: int) -> list[int]:
        """Get tile indices that a corner touches (for resource production)"""
        return corner_tiles(self.state, corner_id)

    def _snapshot_state(self) -> StateSnapshot:
        """Create a serializable snapshot of current state"""
        return snapshot_state(self.state)

    def get_board_state(self) -> BoardState:
        """Get the static board layout for replay inspection."""
        return board_state(self.state)


__all__ = ["ReplayDecoder"]
