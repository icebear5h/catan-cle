"""Transactional checkpoints for authoritative replay steps."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from cle.game_engine.communication import SocialCommitment
from cle.game_engine.events import GameEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.state import GameState
from cle.replay.colonist.types import TradeLedgerRecord
from cle.replay.contracts import ReplayRuntimeState
from cle.replay.runtime.access import get_game_engine, set_game_engine


@dataclass
class ReplayStepCheckpoint:
    """State required to atomically undo one parsed replay action."""

    game: GameEngine
    game_state: GameState
    game_events: tuple[GameEvent, ...]
    game_commitments: tuple[SocialCommitment, ...]
    game_history_length: int
    replay_index: int
    replay_actions_length: int
    game_log_length: int
    semantic_issues_length: int
    first_divergence_step: dict[str, int]
    game_running: bool
    replay_final_state_synced: bool
    replay_pending_dev_card: dict[str, object] | None
    replay_trade_ledger: dict[object, TradeLedgerRecord]

    @classmethod
    def capture(cls, state: ReplayRuntimeState) -> ReplayStepCheckpoint:
        game = get_game_engine(state)
        if game is None:
            # Callers gate on a live engine; keep the attribute error they saw.
            raise AttributeError("'NoneType' object has no attribute 'state'")
        return cls(
            game=game,
            game_state=game.state.copy(),
            game_events=deepcopy(tuple(game.events)),
            game_commitments=deepcopy(tuple(game.commitments)),
            game_history_length=len(getattr(game, "history", [])),
            replay_index=state.replay_index,
            replay_actions_length=len(state.replay_actions_per_step),
            game_log_length=len(state.game_log),
            semantic_issues_length=len(state.replay_semantic_issues),
            first_divergence_step=deepcopy(state.first_divergence_step),
            game_running=state.game_running,
            replay_final_state_synced=state.replay_final_state_synced,
            replay_pending_dev_card=deepcopy(state.replay_pending_dev_card),
            replay_trade_ledger=deepcopy(
                getattr(state, "replay_trade_ledger", {})
            ),
        )

    def restore(self, state: ReplayRuntimeState) -> None:
        """Restore both engine state and replay-owned metadata."""
        set_game_engine(state, self.game)
        self.game.state = self.game_state.copy()
        self.game.rng = self.game.state.rng
        self.game.events = list(deepcopy(self.game_events))
        self.game.commitments = list(deepcopy(self.game_commitments))
        if hasattr(self.game, "history"):
            del self.game.history[self.game_history_length :]

        state.replay_index = self.replay_index
        del state.replay_actions_per_step[self.replay_actions_length :]
        del state.game_log[self.game_log_length :]
        del state.replay_semantic_issues[self.semantic_issues_length :]
        state.first_divergence_step = deepcopy(self.first_divergence_step)
        state.game_running = self.game_running
        state.replay_final_state_synced = self.replay_final_state_synced
        state.replay_pending_dev_card = deepcopy(self.replay_pending_dev_card)
        state.replay_trade_ledger = deepcopy(self.replay_trade_ledger)


def ensure_replay_checkpoint_state(state: ReplayRuntimeState) -> None:
    """Initialize checkpoint storage on older ServerState instances."""
    if not hasattr(state, "replay_step_checkpoints"):
        state.replay_step_checkpoints = []
