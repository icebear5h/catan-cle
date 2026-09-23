"""Detached snapshots and complete restore preflight before live mutation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.sandbox import catan
from cle.sandbox.action_batches import PendingActionBatch
from cle.sandbox.communication import CommunicationOpportunity
from cle.sandbox.contracts import SandboxSnapshot
from cle.sandbox.trade_preauthorization import TradePreauthorization

from .support import MissingPlayerError, SandboxError

if TYPE_CHECKING:
    from . import CatanSandbox


def snapshot(self: CatanSandbox) -> SandboxSnapshot:
    return SandboxSnapshot(
        engine=self.game_engine.snapshot(),
        player_states=tuple(
            (color, catan.deepcopy(player.snapshot())) for color, player in self.players.items()
        ),
        pending_decision_revision=self._pending_decision_revision,
        speech_used=self._speech_used,
        speech_calls_remaining=self._speech_calls_remaining,
        pending_reactions=catan.deepcopy(self._pending_reactions),
        pre_robber_sequence=self._pre_robber_sequence,
        trade_preauthorization=self._trade_preauthorization,
        pending_action_batch=catan.deepcopy(self._pending_action_batch),
    )


def restore(self: CatanSandbox, snapshot: SandboxSnapshot) -> None:
    if self._step_state is not None:
        raise SandboxError("Cannot restore a sandbox while a step is in flight")
    snapshot_colors = set(snapshot.engine.state.colors)
    if set(self.players) != snapshot_colors:
        raise MissingPlayerError("Snapshot colors do not match registered sandbox players")
    if (
        len(snapshot.player_states) != len(snapshot_colors)
        or {color for color, _ in snapshot.player_states} != snapshot_colors
    ):
        raise MissingPlayerError("Snapshot player states do not match registered sandbox players")
    pending = snapshot.pending_decision_revision
    if pending is not None and (type(pending) is not int or pending != len(snapshot.engine.events)):
        raise ValueError("Pending decision does not match the saved engine revision")
    if type(snapshot.speech_used) is not bool or (
        snapshot.speech_calls_remaining is not None and (
            type(snapshot.speech_calls_remaining) is not int or snapshot.speech_calls_remaining < 0
        )
    ):
        raise ValueError("Invalid saved speech budget")
    if not isinstance(snapshot.pending_reactions, tuple) or any(
        not isinstance(item, CommunicationOpportunity) or item.player not in snapshot_colors
        or item.cause.sequence >= len(snapshot.engine.events) or item.round < 0
        for item in snapshot.pending_reactions
    ):
        raise ValueError("Invalid saved reaction queue")
    if snapshot.pre_robber_sequence is not None and (
        type(snapshot.pre_robber_sequence) is not int
        or not 0 <= snapshot.pre_robber_sequence < len(snapshot.engine.events)
    ):
        raise ValueError("Invalid saved pre-robber window")
    authorization = snapshot.trade_preauthorization
    batch = snapshot.pending_action_batch
    if batch is not None:
        if not isinstance(batch, PendingActionBatch) or authorization is not None:
            raise ValueError("Invalid saved action batch")
        batch.validate_snapshot(snapshot.engine)
    if authorization is not None:
        if not isinstance(authorization, TradePreauthorization):
            raise ValueError("Invalid saved trade preauthorization")
        authorization.validate_snapshot(snapshot.engine)
    for color, player_state in snapshot.player_states:
        validator = getattr(self.players[color], "validate_restore", None)
        if validator is not None:
            validator(catan.deepcopy(player_state))
    self.game_engine.restore(snapshot.engine)
    for color, player_state in snapshot.player_states:
        self.players[color].restore(catan.deepcopy(player_state))
    self._pending_decision_revision = pending
    self._speech_used = snapshot.speech_used
    self._speech_calls_remaining = snapshot.speech_calls_remaining
    self._pending_reactions = catan.deepcopy(snapshot.pending_reactions)
    self._pre_robber_sequence = snapshot.pre_robber_sequence
    self._trade_preauthorization = authorization
    self._pending_action_batch = catan.deepcopy(batch)
