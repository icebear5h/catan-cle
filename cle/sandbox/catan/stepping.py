"""Single-player admission and synchronous action-bundle commitment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine.events import GameEvent
from cle.game_engine.models.enums import Action, ActionType
from cle.players.contracts import CommunicationChoice, PlayerChoice
from cle.sandbox import catan
from cle.sandbox.action_batches import PendingActionBatch
from cle.sandbox.communication import (
    CommunicationAdmission,
    CommunicationOpportunity,
    ReactionReason,
)
from cle.sandbox.contracts import SandboxStepResult
from cle.sandbox.trade_preauthorization import TradePreauthorization

from .support import (
    _DISCARD_BARRIER_ACTIONS,
    MissingPlayerError,
    SandboxError,
    TerminalSandboxError,
)

if TYPE_CHECKING:
    from . import CatanSandbox


async def _step(self: CatanSandbox) -> SandboxStepResult:
    if self.game_engine.winning_color() is not None:
        self._pending_action_batch = None
        raise TerminalSandboxError("Cannot step a terminal game")

    if self._pending_action_batch is not None:
        automatic = self._resolve_action_batch()
        if automatic is not None:
            if automatic.winner is not None:
                return automatic
            return await self._post_action_communication(
                automatic,
                max_rounds=self.game_engine.communication_limits.max_general_reaction_rounds,
            )

    # An admitted instruction survives policy rebinding. Resolve it without
    # inference, notes updates, speech polling, or a fabricated player receipt.
    authorization = self._trade_preauthorization
    if authorization is not None:
        reason = authorization.invalid_reason(self.game_engine)
        if reason is not None or authorization.response_revision is not None:
            automatic = self._resolve_trade_preauthorization()
            if automatic is not None:
                return automatic

    self._refresh_inference_policy()
    barrier_contexts = self._trade_barrier_contexts()
    if barrier_contexts:
        return await self._step_barrier(barrier_contexts)
    if self._trade_preauthorization is not None:
        self._resolve_trade_preauthorization()

    discard_contexts = self._discard_barrier_contexts()
    if discard_contexts:
        return await self._step_barrier(
            discard_contexts, allowed=_DISCARD_BARRIER_ACTIONS, label="discard",
        )

    playable_actions = tuple(self.game_engine.state.playable_actions)
    forced_roll = (
        len(playable_actions) == 1 and playable_actions[0].action_type == ActionType.ROLL
    )
    pre_messages: tuple[GameEvent, ...] = ()
    if self._pending_decision_revision is not None:
        if self._pending_decision_revision != self.revision:
            raise SandboxError("Pending decision no longer matches the game revision")
    elif not forced_roll and not self._reactive(self.current_actor()):
        pre_opportunities = self.communication_policy.pre_action(self.game_engine)
        pre_messages = await self._run_communication(pre_opportunities, max_rounds=1)
        if pre_opportunities and getattr(self.players[self.current_actor()], "context_policy", "legacy") == "fresh_notes":
            # A failed decision resumes after its already admitted pre-action speech.
            self._pending_decision_revision = self.revision

    self._open_pre_robber_window()
    pre_messages = (*pre_messages, *await self._run_reactive())
    self._refresh_inference_policy()
    context = self.decision_context(self.current_actor())
    revision = self.revision
    player = self.players.get(context.actor)
    if player is None:
        raise MissingPlayerError(f"No player is registered for {context.actor}")

    action: Action | CommunicationChoice
    if forced_roll:
        action = context.legal_actions[0]
        attempt = None
    else:
        action, attempt = await self._get_action_from_player(player, context, revision=revision)

    if isinstance(action, CommunicationChoice):
        # This is a model choice, never an engine action or a consumed turn.
        assert attempt is not None
        self._validate_context_update(player, attempt.model_request, context)
        self._start_speech_budget()
        event = self._append_speech(context.actor, action, context.context_id)
        self._speech_used = True
        assert self._speech_calls_remaining is not None
        self._speech_calls_remaining -= 1
        self._pending_decision_revision = self.revision
        speech_result = SandboxStepResult((context,), (attempt,), (), (event,))
        player.accept(*catan.deepcopy((attempt, speech_result)))
        projected = catan.project_event(event, context.actor)
        assert projected is not None and context.visible_through_sequence is not None
        self.communication_trace.append(CommunicationAdmission(
            CommunicationOpportunity(
                context.actor, projected, context.visible_through_sequence,
                ReactionReason.STANDALONE_SPEECH, 0,
            ),
            catan.replace(action, model_request=attempt.model_request, model_response=attempt.model_response),
            accepted=True,
        ))
        self._pending_reactions += self.communication_policy.addressed(self.game_engine, (event,))
        pre_messages = (*pre_messages, event, *await self._run_reactive())
        self._refresh_inference_policy()
        context = self.decision_context(self.current_actor())
        revision = self.revision
        player = self.players[context.actor]
        action, attempt = await self._get_action_from_player(player, context, revision=revision)

    self._check_revision(revision)
    choice = None
    if attempt is not None:
        self._validate_context_update(player, attempt.model_request, context)
        assert isinstance(attempt.choice, PlayerChoice)
        choice = attempt.choice
    followup = catan.choice_followup_action(context, choice) if choice is not None else None
    # No awaits or player callbacks may split an admitted Knight bundle.
    assert isinstance(action, Action)
    transitions = [catan.deepcopy(self.game_engine.step(action))]
    if attempt is not None and choice is not None and choice.batch_actions:
        batch = PendingActionBatch(
            actor=context.actor, actions=catan.deepcopy(choice.batch_actions),
            next_index=0, expected_revision=revision,
            turn_number=context.turn_number, phase=context.phase,
            origin_sequence=transitions[0].events[0].sequence,
            origin_context_id=attempt.context_id,
            provider_response_id=choice.provider_response_id,
            provider_request_id=choice.provider_request_id,
        )
        transitions[0] = self._advance_action_batch(batch, transitions[0])
    if attempt is not None and choice is not None and choice.confirm_if_accepted_by is not None:
        offer = transitions[0].resolved_action.value
        window = self.game_engine.state.trade_window
        assert window is not None
        requested = choice.confirm_if_accepted_by
        if requested == "ANY":
            priority = tuple(color for color in self.game_engine.state.colors if color in offer.audience)
        else:
            assert isinstance(requested, tuple)
            priority = requested
        self._trade_preauthorization = TradePreauthorization(
            actor=context.actor, window_id=window.id, offer_id=offer.id,
            turn_number=self.game_engine.state.num_turns, round=window.round,
            give=offer.give, receive=offer.receive, audience=offer.audience,
            priority=priority, offer_sequence=transitions[0].events[0].sequence,
            origin_context_id=attempt.context_id,
            provider_response_id=choice.provider_response_id,
            provider_request_id=choice.provider_request_id,
        )
    if followup is not None and transitions[-1].winner is None:
        transitions.append(catan.deepcopy(self.game_engine.step(followup)))
    self._pending_decision_revision = None
    self._speech_used = False
    self._speech_calls_remaining = None
    result = SandboxStepResult(
        contexts=() if attempt is None else (context,),
        attempts=() if attempt is None else (attempt,),
        transitions=tuple(transitions),
        messages=pre_messages,
    )
    if attempt is not None:
        accepted_attempt, accepted_result = catan.deepcopy((attempt, result))
        player.accept(accepted_attempt, accepted_result)
        if choice is not None and choice.batch_actions and result.winner is not None:
            return result
    return await self._post_action_communication(
        result,
        max_rounds=self.game_engine.communication_limits.max_general_reaction_rounds,
    )
