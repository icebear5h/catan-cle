"""Lightweight asynchronous orchestration for one Catan game engine."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Iterable, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import replace
from typing import Any

from cle.harness.action_tools import legal_tool_names, parse_tool_choice

from cle.players.contracts import (
    CommunicationChoice,
    CommunicationMode,
    PlayerAttempt,
    PlayerContext,
    SandboxPlayer,
    TalkContext,
)
from cle.players.validation import (
    action_from_choice,
    choice_followup_action,
    validate_communication_choice,
    validate_player_attempt,
)
from cle.players.notes import MAX_NOTES_CHARS
from cle.sandbox.communication import (
    CommunicationAdmission,
    CommunicationOpportunity,
    CommunicationPolicy,
    ReactionReason,
)
from cle.sandbox.contracts import (
    RetryPolicy,
    SandboxSnapshot,
    SandboxStepResult,
    SandboxView,
)
from cle.sandbox.decision import build_decision_context
from cle.sandbox.trade_preauthorization import AutomaticTradeAction, TradePreauthorization
from cle.sandbox.action_batches import AutomaticBatchAction, PendingActionBatch
from cle.game_engine.events import EngineTransition, GameEvent, project_event
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions, trade_response_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state import GameState, ensure_trade_window


class SandboxError(RuntimeError):
    pass


class MissingPlayerError(SandboxError):
    pass


class TerminalSandboxError(SandboxError):
    pass


class PlayerResponseError(SandboxError):
    def __init__(
        self,
        player: Color,
        attempts: tuple[PlayerAttempt, ...],
        validation_error: str,
    ) -> None:
        self.player = player
        self.attempts = deepcopy(attempts)
        self.validation_error = validation_error
        super().__init__(
            f"Player {player} failed to choose a valid action after {len(attempts)} attempts"
        )


class PostActionCommunicationError(SandboxError):
    """Speech failed after the action and player history were committed."""

    def __init__(self, result: SandboxStepResult) -> None:
        self.result = result
        super().__init__("Game action committed, but post-action communication failed")


class PostActionCommunicationCancelled(asyncio.CancelledError):
    """Preserve cancellation semantics and the already-committed result."""

    def __init__(self, result: SandboxStepResult) -> None:
        self.result = result
        super().__init__("Game action committed before communication was cancelled")


async def _gather_or_cancel(calls: Iterable[Coroutine[Any, Any, Any]]) -> list[Any]:
    tasks = [asyncio.create_task(call) for call in calls]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class CatanSandbox:
    """Compose one game engine with one independently stateful player per color."""

    def __init__(
        self,
        game_engine: GameEngine,
        players: Mapping[Color, SandboxPlayer],
        *,
        retry_policy: RetryPolicy | None = None,
        communication_policy: CommunicationPolicy | None = None,
        refresh_players: Callable[["CatanSandbox"], None] | None = None,
    ) -> None:
        self.game_engine = game_engine
        self.players = dict(players)
        self.retry_policy = retry_policy or RetryPolicy()
        self.communication_policy = communication_policy or CommunicationPolicy()
        self._refresh_players = refresh_players
        self.decision_trace: list[PlayerAttempt] = []
        self.communication_trace: list[CommunicationAdmission] = []
        self._step_state: GameState | None = None
        self._pending_decision_revision: int | None = None
        self._speech_used = False
        self._speech_calls_remaining: int | None = None
        self._pending_reactions: tuple[CommunicationOpportunity, ...] = ()
        self._pre_robber_sequence: int | None = None
        self._trade_preauthorization: TradePreauthorization | None = None
        self._pending_action_batch: PendingActionBatch | None = None
        self._validate_players()

    @classmethod
    def create(
        cls,
        colors: Sequence[Color],
        players: Mapping[Color, SandboxPlayer],
        *,
        seed: int | None = None,
        discard_limit: int = 7,
        vps_to_win: int = 10,
        shuffle_players: bool = True,
        retry_policy: RetryPolicy | None = None,
        communication_policy: CommunicationPolicy | None = None,
    ) -> "CatanSandbox":
        engine = GameEngine(
            colors,
            seed=seed,
            discard_limit=discard_limit,
            vps_to_win=vps_to_win,
            shuffle_players=shuffle_players,
        )
        return cls(
            engine,
            players,
            retry_policy=retry_policy,
            communication_policy=communication_policy,
        )

    @property
    def revision(self) -> int:
        return self.game_engine.revision

    def current_actor(self) -> Color:
        return self.game_engine.state.current_color()

    def register_player(self, player: SandboxPlayer) -> None:
        if self._step_state is not None:
            raise SandboxError("Cannot replace a player while a step is in flight")
        if player.color not in self.game_engine.state.colors:
            raise ValueError(f"Player color {player.color} is not in this game")
        self.players[player.color] = player
        self._validate_players()

    def view(self, observer: Color | None = None) -> SandboxView:
        observer = observer or self.current_actor()
        observation = self.game_engine.observe(observer)
        current_actor = self.current_actor()
        return deepcopy(SandboxView(
            revision=self.revision,
            observer=observer,
            current_actor=current_actor,
            turn_number=self.game_engine.state.num_turns,
            phase=observation.current_phase,
            observation=observation,
            events=self.game_engine.project_events(observer),
            legal_actions=(
                tuple(self.game_engine.state.playable_actions)
                if observer == current_actor and self.game_engine.winning_color() is None
                else ()
            ),
            winner=self.game_engine.winning_color(),
        ))

    async def step(self) -> SandboxStepResult:
        """Obtain one valid player action, apply it strictly, and acknowledge it."""
        if self._step_state is not None:
            raise SandboxError("A sandbox already has a step in flight")
        self._step_state = self.game_engine.state
        try:
            return await self._step()
        finally:
            self._step_state = None

    async def _step(self) -> SandboxStepResult:
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

        playable_actions = tuple(self.game_engine.state.playable_actions)
        forced_roll = (
            len(playable_actions) == 1 and playable_actions[0].action_type == ActionType.ROLL
        )
        pre_messages = ()
        if self._pending_decision_revision is not None:
            if self._pending_decision_revision != self.revision:
                raise SandboxError("Pending decision no longer matches the game revision")
        elif not forced_roll and not self._reactive(self.current_actor()):
            pre_opportunities = self.communication_policy.pre_action(self.game_engine)
            pre_messages = await self._run_communication(
                pre_opportunities,
                max_rounds=1,
            )
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

        if forced_roll:
            action = context.legal_actions[0]
            attempt = None
        else:
            action, attempt = await self._get_action_from_player(player, context, revision=revision)

        if isinstance(action, CommunicationChoice):
            # This is a model choice, never an engine action or a consumed turn.
            self._validate_context_update(player, attempt.model_request, context)
            self._start_speech_budget()
            event = self._append_speech(context.actor, action, context.context_id)
            self._speech_used = True
            self._speech_calls_remaining -= 1
            self._pending_decision_revision = self.revision
            speech_result = SandboxStepResult((context,), (attempt,), (), (event,))
            player.accept(*deepcopy((attempt, speech_result)))
            self.communication_trace.append(CommunicationAdmission(
                CommunicationOpportunity(
                    context.actor, project_event(event, context.actor),
                    context.visible_through_sequence, ReactionReason.STANDALONE_SPEECH, 0,
                ),
                replace(action, model_request=attempt.model_request, model_response=attempt.model_response),
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
        if attempt is not None:
            self._validate_context_update(player, attempt.model_request, context)
        followup = choice_followup_action(context, attempt.choice) if attempt is not None else None
        # No awaits or player callbacks may split an admitted Knight bundle.
        transitions = [deepcopy(self.game_engine.step(action))]
        if attempt is not None and attempt.choice.batch_actions:
            batch = PendingActionBatch(
                actor=context.actor, actions=deepcopy(attempt.choice.batch_actions),
                next_index=0, expected_revision=revision,
                turn_number=context.turn_number, phase=context.phase,
                origin_sequence=transitions[0].events[0].sequence,
                origin_context_id=attempt.context_id,
                provider_response_id=attempt.choice.provider_response_id,
                provider_request_id=attempt.choice.provider_request_id,
            )
            transitions[0] = self._advance_action_batch(batch, transitions[0])
        if attempt is not None and attempt.choice.confirm_if_accepted_by is not None:
            offer = transitions[0].resolved_action.value
            window = self.game_engine.state.trade_window
            requested = attempt.choice.confirm_if_accepted_by
            priority = (
                tuple(color for color in self.game_engine.state.colors if color in offer.audience)
                if requested == "ANY" else requested
            )
            self._trade_preauthorization = TradePreauthorization(
                actor=context.actor, window_id=window.id, offer_id=offer.id,
                turn_number=self.game_engine.state.num_turns, round=window.round,
                give=offer.give, receive=offer.receive, audience=offer.audience,
                priority=priority, offer_sequence=transitions[0].events[0].sequence,
                origin_context_id=attempt.context_id,
                provider_response_id=attempt.choice.provider_response_id,
                provider_request_id=attempt.choice.provider_request_id,
            )
        if followup is not None and transitions[-1].winner is None:
            transitions.append(deepcopy(self.game_engine.step(followup)))
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
            accepted_attempt, accepted_result = deepcopy((attempt, result))
            player.accept(accepted_attempt, accepted_result)
            if attempt.choice.batch_actions and result.winner is not None:
                return result
        return await self._post_action_communication(
            result,
            max_rounds=self.game_engine.communication_limits.max_general_reaction_rounds,
        )

    async def _step_barrier(
        self,
        contexts: tuple[PlayerContext, ...],
    ) -> SandboxStepResult:
        revision = self.revision
        players = tuple(self.players[context.actor] for context in contexts)
        failures: list[list[PlayerAttempt]] = [[] for _ in contexts]
        pending: dict[Color, PlayerAttempt] = {}

        async def acquire(index: int) -> tuple[Action, PlayerAttempt]:
            action, attempt = await self._get_action_from_player(
                players[index], contexts[index], revision=revision, failed_attempts=failures[index]
            )
            pending[contexts[index].actor] = attempt
            return action, attempt

        try:
            choices = await _gather_or_cancel(acquire(index) for index in range(len(contexts)))
            # Revalidate the complete batch after every retry, including engine
            # legality and funding, not only trade-window capacity.
            while True:
                self._check_revision(revision)
                staged = deepcopy(self.game_engine)
                for index, (context, (_, attempt)) in enumerate(zip(contexts, choices)):
                    try:
                        action = action_from_choice(
                            context, attempt.choice,
                            max_notes_chars=getattr(players[index], "max_notes_chars", MAX_NOTES_CHARS),
                        )
                        self._validate_context_update(players[index], attempt.model_request, context)
                        if action.action_type not in {
                            ActionType.COUNTER_OFFER,
                            ActionType.ACCEPT_TRADE,
                            ActionType.REJECT_TRADE,
                        }:
                            raise ValueError("Unsupported trade barrier response")
                        if action.action_type == ActionType.COUNTER_OFFER:
                            ensure_trade_window(staged.state).validate_offer(action.value)
                        staged.step(deepcopy(action))
                    except ValueError as exc:
                        pending.pop(context.actor)
                        rejected = replace(attempt, validation_error=str(exc))
                        self.decision_trace.append(rejected)
                        failures[index].append(rejected)
                        choices[index] = await acquire(index)
                        break
                    choices[index] = (action, attempt)
                else:
                    break

            self._check_revision(revision)
        except BaseException as exc:
            # Acquisition has already cancelled and awaited unfinished children.
            # Only completed, not-yet-rejected replies remain pending.
            self.decision_trace.extend(
                replace(
                    pending[context.actor],
                    validation_error=(
                        "Decision withheld: trade barrier did not commit: "
                        f"{str(exc) or type(exc).__name__}"
                    ),
                )
                for context in contexts
                if context.actor in pending
            )
            raise
        # No player callbacks or awaits may split an admitted live batch.
        transitions = tuple(
            deepcopy(self.game_engine.step(action)) for action, _ in choices
        )
        window = self.game_engine.state.trade_window
        if window is not None and window.active_offers:
            window.advance_round()
            self.game_engine.state.playable_actions = generate_playable_actions(
                self.game_engine.state
            )
        if self._trade_preauthorization is not None:
            self._trade_preauthorization = replace(
                self._trade_preauthorization, response_revision=self.revision,
            )
        result = SandboxStepResult(
            contexts=contexts,
            attempts=tuple(attempt for _, attempt in choices),
            transitions=transitions,
        )
        for player, context, (_, attempt), transition in zip(players, contexts, choices, transitions):
            accepted_attempt, accepted_result = deepcopy((
                attempt,
                SandboxStepResult(
                    contexts=(context,),
                    attempts=(attempt,),
                    transitions=(transition,),
                ),
            ))
            player.accept(accepted_attempt, accepted_result)
        return await self._post_action_communication(
            result,
            max_rounds=self.game_engine.communication_limits.max_trade_reaction_rounds,
        )

    async def _post_action_communication(
        self, result: SandboxStepResult, *, max_rounds: int
    ) -> SandboxStepResult:
        messages: list[GameEvent] = []
        try:
            await self._run_communication(
                tuple(item for item in self.communication_policy.after_events(
                    self.game_engine,
                    tuple(
                        event for transition in result.transitions for event in transition.events
                    ),
                    round_number=0,
                ) if not self._reactive(item.player)),
                max_rounds=max_rounds,
                emitted=messages,
            )
        except asyncio.CancelledError as exc:
            raise PostActionCommunicationCancelled(
                replace(result, messages=(*result.messages, *deepcopy(messages)))
            ) from exc
        except Exception as exc:
            raise PostActionCommunicationError(
                replace(result, messages=(*result.messages, *deepcopy(messages)))
            ) from exc
        return replace(result, messages=(*result.messages, *deepcopy(messages)))

    def _batch_boundary(self, batch: PendingActionBatch, action: Action) -> str | None:
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

    def _pause_action_batch(self, batch: PendingActionBatch, reason: str) -> GameEvent:
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

    def _advance_action_batch(self, batch: PendingActionBatch, transition: EngineTransition) -> EngineTransition:
        consumed = replace(batch, next_index=batch.next_index + 1, expected_revision=self.revision)
        self._pending_action_batch = None
        if consumed.next_index == len(batch.actions):
            return transition
        reason = self._batch_boundary(batch, transition.requested_action)
        if reason:
            event = self._pause_action_batch(consumed, reason)
            return replace(transition, after_revision=self.revision, events=(*transition.events, deepcopy(event)))
        self._pending_action_batch = consumed
        return transition

    def _resolve_action_batch(self) -> SandboxStepResult | None:
        batch = self._pending_action_batch
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
            context = build_decision_context(
                self.game_engine, advertised_actions=tuple(generate_playable_actions(self.game_engine.state)),
            )
            choice = parse_tool_choice(context, call["tool"], call["arguments"], shared=True)
            action = action_from_choice(context, choice)
            if not self.game_engine.is_action_valid(action):
                raise ValueError("Action is no longer legal or affordable")
        except ValueError as exc:
            self._pause_action_batch(batch, f"Action {batch.next_index + 1} ({call['tool']}) failed: {exc}")
            return None
        # No await between consumption and strict commit; cancellation cannot replay it.
        transition = deepcopy(self.game_engine.step(action))
        transition = self._advance_action_batch(batch, transition)
        return SandboxStepResult(
            contexts=(), attempts=(), transitions=(transition,),
            automatic_action=AutomaticBatchAction(deepcopy(batch), transition.events[0].sequence),
        )

    def _resolve_trade_preauthorization(self) -> SandboxStepResult | None:
        authorization = self._trade_preauthorization
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
            contexts=(), attempts=(), transitions=(deepcopy(transition),),
            automatic_action=AutomaticTradeAction(authorization, transition.events[0].sequence),
        )

    def _reactive(self, color: Color) -> bool:
        return bool(getattr(self.players[color], "reactive_speech", False))

    def _start_speech_budget(self) -> None:
        if self._speech_calls_remaining is None:
            self._speech_calls_remaining = self.game_engine.communication_limits.max_messages_per_window

    def _append_speech(self, speaker: Color, choice: CommunicationChoice, causation_id: str) -> GameEvent:
        proposal = choice.commitment
        return deepcopy(self.game_engine.append_message(
            speaker=speaker, text=choice.text, audience=choice.audience,
            causation_id=causation_id,
            respondents=choice.respondents,
            commitment=(proposal.condition, proposal.promise, proposal.expires_turn) if proposal else None,
        ))

    def _open_pre_robber_window(self) -> None:
        actions = self.game_engine.state.playable_actions
        if not actions or any(action.action_type != ActionType.MOVE_ROBBER for action in actions):
            return
        # Most recent cause distinguishes a seven from historical split Knights.
        cause = next((event for event in reversed(self.game_engine.events) if event.event_type in {
            ActionType.ROLL.value, ActionType.PLAY_KNIGHT_CARD.value, ActionType.MOVE_ROBBER.value,
        }), None)
        if cause is None or cause.event_type != ActionType.ROLL.value or sum(cause.public_payload) != 7:
            return
        if cause.sequence == self._pre_robber_sequence:
            return
        self._pre_robber_sequence = cause.sequence
        self._start_speech_budget()
        self._pending_reactions += tuple(
            CommunicationOpportunity(
                color, project_event(cause, color), self.revision - 1,
                ReactionReason.PRE_ROBBER, 0,
            )
            for color in self.game_engine.state.colors
            if color != self.current_actor() and self._reactive(color)
        )

    async def _run_reactive(self) -> tuple[GameEvent, ...]:
        """Durable bounded queue; successful replies are removed before another await."""
        emitted = []
        limits = self.game_engine.communication_limits
        while self._pending_reactions:
            self._refresh_inference_policy()
            self._start_speech_budget()
            if self._speech_calls_remaining <= 0:
                self._pending_reactions = ()
                break
            opportunity = self._pending_reactions[0]
            player = self.players[opportunity.player]
            session = getattr(player, "session", None)
            received = max(getattr(session, "action_next_sequence", 0), getattr(session, "talk_next_sequence", 0))
            if opportunity.round >= limits.max_general_reaction_rounds or (
                opportunity.reason == ReactionReason.ADDRESSED_SPEECH and opportunity.cause.sequence < received
            ):
                self._pending_reactions = self._pending_reactions[1:]
                continue
            # A failed provider call consumes its slot too, so retries cannot reset a window.
            self._speech_calls_remaining -= 1
            messages = await self._run_communication((opportunity,), max_rounds=1)
            self._pending_reactions = self._pending_reactions[1:]
            emitted.extend(messages)
            self._pending_reactions += self.communication_policy.addressed(
                self.game_engine, messages, round_number=opportunity.round + 1,
            )
        return tuple(emitted)

    async def _run_communication(
        self,
        initial: tuple[CommunicationOpportunity, ...],
        *,
        max_rounds: int,
        emitted: list[GameEvent] | None = None,
    ) -> tuple[GameEvent, ...]:
        opportunities = deepcopy(initial)
        if emitted is None:
            emitted = []
        round_number = 0
        limits = self.game_engine.communication_limits
        while opportunities and round_number < max_rounds:
            self._refresh_inference_policy()
            opportunities = tuple(item for item in opportunities if not self._reactive(item.player) or item.reason in {
                ReactionReason.PRE_ROBBER, ReactionReason.ADDRESSED_SPEECH,
            })
            if not opportunities:
                break
            revision = self.revision
            cutoff = max(item.visible_through_sequence for item in opportunities)
            by_player = {}
            for opportunity in opportunities:
                by_player.setdefault(
                    opportunity.player,
                    replace(
                        opportunity,
                        visible_through_sequence=(
                            revision - 1
                            if getattr(self.players[opportunity.player], "context_policy", "legacy") == "fresh_notes"
                            else cutoff
                        ),
                    ),
                )
            ordered = tuple(
                by_player[color] for color in self.game_engine.state.colors if color in by_player
            )
            contexts = tuple(self._talk_context(item) for item in ordered)
            # Retain completed replies even if a sibling fails before admission.
            acquired: dict[Color, CommunicationChoice] = {}
            invalid: dict[Color, str] = {}

            async def communicate(
                item: CommunicationOpportunity, context: TalkContext
            ) -> CommunicationChoice:
                returned = await self.players[item.player].communicate(deepcopy(context))
                try:
                    validate_communication_choice(
                        returned, speaker=item.player, participants=context.participants,
                        max_notes_chars=getattr(self.players[item.player], "max_notes_chars", MAX_NOTES_CHARS),
                    )
                    self._validate_context_update(self.players[item.player], returned.model_request, context)
                except ValueError as exc:
                    invalid[item.player] = str(exc)
                try:
                    choice = (
                        deepcopy(returned)
                        if isinstance(returned, CommunicationChoice)
                        else CommunicationChoice()
                    )
                except TypeError as exc:
                    invalid[item.player] = f"Communication choice cannot be detached: {exc}"
                    choice = CommunicationChoice()
                acquired[item.player] = choice
                return choice

            try:
                choices = await _gather_or_cancel(
                    communicate(item, context) for item, context in zip(ordered, contexts)
                )
                self._check_revision(revision)
            except BaseException as exc:
                self.communication_trace.extend(
                    CommunicationAdmission(
                        item,
                        acquired[item.player],
                        accepted=False,
                        validation_error=(
                            f"Communication withheld: {str(exc) or type(exc).__name__}"
                        ),
                    )
                    for item in ordered
                    if item.player in acquired
                )
                raise
            round_events = []
            failure: Exception | None = None
            withheld_error = None
            for opportunity, context, choice in zip(ordered, contexts, choices):
                validation_error = withheld_error
                if validation_error is None and opportunity.player not in invalid:
                    try:
                        self._validate_context_update(self.players[opportunity.player], choice.model_request, context)
                    except ValueError as exc:
                        invalid[opportunity.player] = str(exc)
                if validation_error is None and opportunity.player in invalid:
                    validation_error = invalid[opportunity.player]
                    failure = ValueError(validation_error)
                    withheld_error = (
                        f"Communication withheld after {opportunity.player.value} "
                        f"rejection: {validation_error}"
                    )
                if validation_error is None and choice.mode != CommunicationMode.SILENCE:
                    if len(emitted) >= limits.max_messages_per_window:
                        validation_error = "Communication withheld: message limit reached"
                    else:
                        commitment = (
                            (
                                choice.commitment.condition,
                                choice.commitment.promise,
                                choice.commitment.expires_turn,
                            )
                            if choice.commitment is not None
                            else None
                        )
                        try:
                            event = self.game_engine.append_message(
                                speaker=opportunity.player,
                                text=choice.text,
                                audience=choice.audience,
                                causation_id=opportunity.cause.causation_id,
                                commitment=commitment,
                                respondents=choice.respondents,
                            )
                        except Exception as exc:
                            failure = exc
                            validation_error = str(exc) or type(exc).__name__
                            withheld_error = (
                                f"Communication withheld after {opportunity.player.value} "
                                f"rejection: {validation_error}"
                            )
                        else:
                            round_events.append(deepcopy(event))
                            emitted.append(deepcopy(event))
                            if self._pending_decision_revision is not None or self._reactive(opportunity.player):
                                self._pending_decision_revision = self.revision
                self.communication_trace.append(
                    CommunicationAdmission(
                        opportunity,
                        choice,
                        accepted=validation_error is None,
                        validation_error=validation_error,
                    )
                )
                if validation_error is None:
                    accept_communication = getattr(self.players[opportunity.player], "accept_communication", None)
                    if accept_communication is not None:
                        accept_communication(deepcopy(context), deepcopy(choice))
                    self.players[opportunity.player].acknowledge_events(
                        opportunity.visible_through_sequence + 1
                    )
            if failure is not None:
                raise failure
            round_number += 1
            opportunities = self.communication_policy.after_events(
                self.game_engine,
                tuple(round_events),
                round_number=round_number,
            )
        return tuple(emitted)

    def _refresh_inference_policy(self) -> None:
        # Only call with no outstanding acquisition/admission. Concurrent speech
        # and trade batches, including retries, retain their original players.
        if self._refresh_players is not None:
            self._refresh_players(self)

    def _talk_context(self, opportunity: CommunicationOpportunity) -> TalkContext:
        color = opportunity.player
        cutoff = opportunity.visible_through_sequence
        visible = tuple(
            event for event in self.game_engine.project_events(color) if event.sequence <= cutoff
        )
        return deepcopy(TalkContext(
            context_id=(
                f"{self.game_engine.id}:talk:{opportunity.cause.causation_id}:"
                f"{opportunity.round}:{color.value}"
                + (f":{opportunity.cause.sequence}" if self._reactive(color) else "")
            ),
            player=color,
            participants=self.game_engine.state.colors,
            cause=opportunity.cause,
            visible_through_sequence=opportunity.visible_through_sequence,
            game_events=tuple(
                event for event in visible if event.event_type != "MESSAGE_SENT"
            ),
            recent_messages=tuple(
                event for event in visible if event.event_type == "MESSAGE_SENT"
            )[-self.game_engine.communication_limits.recent_message_window :],
            active_commitments=tuple(
                item for item in self.game_engine.active_commitments(color)
                if item.created_sequence <= cutoff and item.source_message_sequence <= cutoff
            ),
            observation=(
                self.game_engine.observe(color)
                if getattr(self.players[color], "context_policy", "legacy") == "fresh_notes"
                else None
            ),
            visible_messages=tuple(
                event for event in visible if event.event_type == "MESSAGE_SENT"
            ),
            trigger_reason=opportunity.reason.value,
        ))

    @staticmethod
    def _validate_context_update(
        player: SandboxPlayer, request: Any, context: PlayerContext | TalkContext,
    ) -> None:
        validator = getattr(player, "validate_context_update", None)
        if validator is not None:
            validator(deepcopy(request), deepcopy(context))

    async def _prompt_player(
        self,
        player: SandboxPlayer,
        context: PlayerContext,
        feedback: str | None,
    ) -> PlayerAttempt:
        return await player.choose(deepcopy(context), feedback)

    @staticmethod
    def _retry_feedback(
        player: SandboxPlayer,
        context: PlayerContext,
        error: str,
    ) -> str:
        """Validation error plus the actor's exact legal tools.

        Base prompts intentionally omit the concrete menu; after a failure the
        model has proven it cannot infer legality, so the correction names it.
        """
        tools = ", ".join(legal_tool_names(context))
        if not tools:
            return error
        batches = getattr(
            getattr(getattr(player, "suite", None), "context", None),
            "deterministic_batches", False,
        )
        suffix = f" Your currently legal tools: {tools}."
        if batches:
            suffix += " Or return one bounded actions batch starting with one of them."
        return f"{error}{suffix}"

    async def _get_action_from_player(
        self,
        player: SandboxPlayer,
        context: PlayerContext,
        *,
        revision: int,
        failed_attempts: list[PlayerAttempt] | None = None,
    ) -> tuple[Action | CommunicationChoice, PlayerAttempt]:
        if failed_attempts is None:
            failed_attempts = []
        feedback = (
            self._retry_feedback(player, context, failed_attempts[-1].validation_error)
            if failed_attempts and failed_attempts[-1].validation_error else None
        )
        for _ in range(len(failed_attempts), self.retry_policy.max_decision_attempts):
            self._check_revision(revision)
            returned = await self._prompt_player(player, context, feedback)
            action = None
            try:
                action = validate_player_attempt(
                    context, returned,
                    max_notes_chars=getattr(player, "max_notes_chars", MAX_NOTES_CHARS),
                )
            except ValueError as exc:
                attempt = PlayerAttempt(
                    context.context_id,
                    None,
                    str(exc),
                    model_request=returned.model_request if isinstance(returned, PlayerAttempt) else None,
                    model_response=returned.model_response if isinstance(returned, PlayerAttempt) else None,
                )
            else:
                attempt = returned
            try:
                attempt = deepcopy(attempt)
            except TypeError as exc:
                action = None
                attempt = PlayerAttempt(context.context_id, None, f"Player attempt cannot be detached: {exc}")
            try:
                self._check_revision(revision)
            except SandboxError as exc:
                self.decision_trace.append(replace(attempt, choice=None, validation_error=str(exc)))
                raise
            try:
                if action is None:
                    raise ValueError(attempt.validation_error)
                self._validate_context_update(player, attempt.model_request, context)
                if isinstance(action, CommunicationChoice):
                    if not self._reactive(context.actor):
                        raise ValueError("Standalone speech requires the reactive contract")
                    return action, attempt
                if attempt.choice.batch_actions:
                    if not getattr(getattr(getattr(player, "suite", None), "context", None), "deterministic_batches", False):
                        raise ValueError("Action batches require an opted-in shared/fresh player")
                    first = attempt.choice.batch_actions[0]
                    bound = parse_tool_choice(context, first["tool"], first["arguments"], shared=True)
                    if action_from_choice(context, bound) != action:
                        raise ValueError("Batch first action does not match the selected action")
                    if action not in generate_playable_actions(self.game_engine.state):
                        raise ValueError("Batch first action is no longer legal or affordable")
                if action.action_type in {ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER}:
                    staged = copy(self.game_engine.state)
                    ensure_trade_window(staged).validate_offer(action.value)
                if not self.game_engine.is_action_valid(action):
                    raise ValueError(
                        "The selected action parameters are no longer legal or "
                        "affordable. Choose again from the current context."
                    )
                followup = choice_followup_action(context, attempt.choice)
                if followup is not None:
                    staged = deepcopy(self.game_engine)
                    transition = staged.step(action)
                    if transition.winner is None:
                        staged.step(followup)
            except ValueError as exc:
                feedback = self._retry_feedback(player, context, str(exc))
                rejected = replace(
                    attempt,
                    context_id=context.context_id,
                    choice=attempt.choice if action is not None else None,
                    validation_error=feedback,
                )
                self.decision_trace.append(rejected)
                failed_attempts.append(rejected)
                continue
            return action, attempt
        raise PlayerResponseError(
            player.color,
            tuple(failed_attempts),
            feedback or "Return one valid action index.",
        )

    def _check_revision(self, revision: int) -> None:
        if self.revision != revision or self.game_engine.state is not self._step_state:
            raise SandboxError(
                f"Stale context: game state changed; expected revision {revision}, "
                f"found {self.revision}"
            )

    def snapshot(self) -> SandboxSnapshot:
        return SandboxSnapshot(
            engine=self.game_engine.snapshot(),
            player_states=tuple(
                (color, deepcopy(player.snapshot())) for color, player in self.players.items()
            ),
            pending_decision_revision=self._pending_decision_revision,
            speech_used=self._speech_used,
            speech_calls_remaining=self._speech_calls_remaining,
            pending_reactions=deepcopy(self._pending_reactions),
            pre_robber_sequence=self._pre_robber_sequence,
            trade_preauthorization=self._trade_preauthorization,
            pending_action_batch=deepcopy(self._pending_action_batch),
        )

    def restore(self, snapshot: SandboxSnapshot) -> None:
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
                validator(deepcopy(player_state))
        self.game_engine.restore(snapshot.engine)
        for color, player_state in snapshot.player_states:
            self.players[color].restore(deepcopy(player_state))
        self._pending_decision_revision = pending
        self._speech_used = snapshot.speech_used
        self._speech_calls_remaining = snapshot.speech_calls_remaining
        self._pending_reactions = deepcopy(snapshot.pending_reactions)
        self._pre_robber_sequence = snapshot.pre_robber_sequence
        self._trade_preauthorization = authorization
        self._pending_action_batch = deepcopy(batch)

    def _trade_barrier_contexts(self) -> tuple[PlayerContext, ...]:
        state = self.game_engine.state
        contexts = []
        window = state.trade_window
        for color in state.colors:
            if window is not None and color == window.turn_player:
                continue
            actions = tuple(trade_response_actions(state, color))
            if actions:
                contexts.append(self.decision_context(color, actions))
        return tuple(contexts)

    def decision_context(
        self,
        actor: Color | None = None,
        advertised_actions: tuple[Action, ...] | None = None,
    ) -> PlayerContext:
        """Build the shared perspective-safe context for one exact decision."""
        context = build_decision_context(
            self.game_engine,
            actor,
            advertised_actions,
            context_revision=self.revision,
        )
        return replace(context, speech_allowed=(
            advertised_actions is None and self._reactive(context.actor) and not self._speech_used
            and (self._speech_calls_remaining is None or self._speech_calls_remaining > 0)
        ))

    def _validate_players(self) -> None:
        engine_colors = set(self.game_engine.state.colors)
        player_colors = set(self.players)
        if player_colors != engine_colors:
            missing = engine_colors - player_colors
            extra = player_colors - engine_colors
            raise ValueError(
                f"Sandbox players must exactly match engine colors; "
                f"missing={sorted(color.value for color in missing)}, "
                f"extra={sorted(color.value for color in extra)}"
            )
        mismatched = [color for color, player in self.players.items() if player.color != color]
        if mismatched:
            raise ValueError(f"Player mapping keys do not match players: {mismatched}")
