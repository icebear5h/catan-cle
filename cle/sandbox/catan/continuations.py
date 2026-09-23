"""Synchronous consumption of admitted batches and trade preauthorizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine.events import EngineTransition, GameEvent
from cle.game_engine.models.enums import Action, ActionType
from cle.sandbox import catan
from cle.sandbox.action_batches import AutomaticBatchAction, PendingActionBatch
from cle.sandbox.contracts import SandboxStepResult
from cle.sandbox.trade_preauthorization import AutomaticTradeAction

if TYPE_CHECKING:
    from . import CatanSandbox


def _batch_boundary(self: CatanSandbox, batch: PendingActionBatch, action: Action) -> str | None:
    if self.game_engine.winning_color() is not None:
        return "Game ended"
    if batch.phase == "initial_placement" and action.action_type == ActionType.BUILD_ROAD:
        return "Setup settlement-road pair completed; each pair requires a separate decision"
    if action.action_type == ActionType.END_TURN:
        return "Turn ended"
    if self.current_actor() != batch.actor or self.game_engine.state.num_turns != batch.turn_number:
        return "Actor or turn changed"
    if self.game_engine.observe(batch.actor).current_phase != batch.phase:
        return "Decision phase changed"
    return None


def _pause_action_batch(self: CatanSandbox, batch: PendingActionBatch, reason: str) -> GameEvent:
    self._pending_action_batch = None
    # A pending speech decision must follow the new private feedback revision.
    event = self.game_engine.publish_event(
        "ACTION_BATCH_PAUSED", batch.actor,
        {"reason": reason, "completed_actions": batch.next_index,
         "discarded_actions": len(batch.actions) - batch.next_index,
         "instruction": "Committed prefix remains. Remainder discarded. Decide from current state; do not repeat completed actions."},
        visible_to=(batch.actor,), causation_id=batch.origin_context_id,
    )
    if self._pending_decision_revision is not None:
        self._pending_decision_revision = self.revision
    return event


def _advance_action_batch(
    self: CatanSandbox, batch: PendingActionBatch, transition: EngineTransition,
) -> EngineTransition:
    consumed = catan.replace(batch, next_index=batch.next_index + 1, expected_revision=self.revision)
    self._pending_action_batch = None
    if consumed.next_index == len(batch.actions):
        return transition
    reason = self._batch_boundary(batch, transition.requested_action)
    if reason:
        event = self._pause_action_batch(consumed, reason)
        return catan.replace(transition, after_revision=self.revision, events=(*transition.events, catan.deepcopy(event)))
    self._pending_action_batch = consumed
    return transition


def _resolve_action_batch(self: CatanSandbox) -> SandboxStepResult | None:
    batch = self._pending_action_batch
    assert batch is not None
    reason = None
    if self.revision != batch.expected_revision:
        reason = "New events arrived after the admitted action; fresh observation required"
    elif self._pending_reactions or self._trade_barrier_contexts():
        reason = "Speech or player-trade response barrier requires a fresh decision"
    elif self.current_actor() != batch.actor or self.game_engine.state.num_turns != batch.turn_number:
        reason = "Actor or turn changed"
    elif self.game_engine.observe(batch.actor).current_phase != batch.phase:
        reason = "Decision phase changed"
    if reason:
        self._pause_action_batch(batch, reason)
        return None
    call = batch.actions[batch.next_index]
    try:
        # Never bind to the original menu: earlier actions may unlock this one.
        context = catan.build_decision_context(
            self.game_engine, advertised_actions=tuple(catan.generate_playable_actions(self.game_engine.state)),
        )
        tool, arguments = call["tool"], call["arguments"]
        assert isinstance(tool, str) and isinstance(arguments, dict)
        choice = catan.parse_tool_choice(context, tool, arguments, shared=True)
        action = catan.action_from_choice(context, choice)
        if not self.game_engine.is_action_valid(action):
            raise ValueError("Action is no longer legal or affordable")
    except ValueError as exc:
        self._pause_action_batch(batch, f"Action {batch.next_index + 1} ({call['tool']}) failed: {exc}")
        return None
    # No await between consumption and strict commit; cancellation cannot replay it.
    transition = catan.deepcopy(self.game_engine.step(action))
    transition = self._advance_action_batch(batch, transition)
    return SandboxStepResult(
        contexts=(), attempts=(), transitions=(transition,),
        automatic_action=AutomaticBatchAction(catan.deepcopy(batch), transition.events[0].sequence),
    )


def _resolve_trade_preauthorization(self: CatanSandbox) -> SandboxStepResult | None:
    authorization = self._trade_preauthorization
    assert authorization is not None
    action, reason = authorization.resolve(self.game_engine)
    # Resolution is synchronous; consumption and transfer cannot be split by
    # cancellation. A later retry can never repeat this authorization.
    self._trade_preauthorization = None
    if action is None:
        self.game_engine.publish_event(
            "TRADE_PREAUTHORIZATION_PAUSED", authorization.actor,
            {"offer_id": authorization.offer_id, "reason": reason,
             "instruction": "Authorization consumed. Decide again; no automatic trade occurred."},
            visible_to=(authorization.actor,),
            causation_id=authorization.origin_context_id,
        )
        return None
    transition = self.game_engine.step(action)
    return SandboxStepResult(
        contexts=(), attempts=(), transitions=(catan.deepcopy(transition),),
        automatic_action=AutomaticTradeAction(authorization, transition.events[0].sequence),
    )
