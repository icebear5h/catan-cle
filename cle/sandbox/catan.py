"""Lightweight asynchronous orchestration for one Catan game engine."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Iterable, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import replace
from typing import Any

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
    validate_communication_choice,
    validate_player_attempt,
)
from cle.sandbox.communication import (
    CommunicationAdmission,
    CommunicationOpportunity,
    CommunicationPolicy,
)
from cle.sandbox.contracts import (
    RetryPolicy,
    SandboxSnapshot,
    SandboxStepResult,
    SandboxView,
)
from cle.sandbox.decision import build_decision_context
from cle.game_engine.events import GameEvent
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
    ) -> None:
        self.game_engine = game_engine
        self.players = dict(players)
        self.retry_policy = retry_policy or RetryPolicy()
        self.communication_policy = communication_policy or CommunicationPolicy()
        self.decision_trace: list[PlayerAttempt] = []
        self.communication_trace: list[CommunicationAdmission] = []
        self._step_state: GameState | None = None
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
            raise TerminalSandboxError("Cannot step a terminal game")

        barrier_contexts = self._trade_barrier_contexts()
        if barrier_contexts:
            return await self._step_barrier(barrier_contexts)

        playable_actions = tuple(self.game_engine.state.playable_actions)
        forced_roll = (
            len(playable_actions) == 1 and playable_actions[0].action_type == ActionType.ROLL
        )
        pre_messages = ()
        if not forced_roll:
            pre_messages = await self._run_communication(
                self.communication_policy.pre_action(self.game_engine),
                max_rounds=1,
            )

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

        self._check_revision(revision)
        transition = deepcopy(self.game_engine.step(action))
        result = SandboxStepResult(
            contexts=() if attempt is None else (context,),
            attempts=() if attempt is None else (attempt,),
            transitions=(transition,),
            messages=pre_messages,
        )
        if attempt is not None:
            accepted_attempt, accepted_result = deepcopy((attempt, result))
            player.accept(accepted_attempt, accepted_result)
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
                        action = action_from_choice(context, attempt.choice)
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
                self.communication_policy.after_events(
                    self.game_engine,
                    tuple(
                        event for transition in result.transitions for event in transition.events
                    ),
                    round_number=0,
                ),
                max_rounds=max_rounds,
                emitted=messages,
            )
        except Exception as exc:
            raise PostActionCommunicationError(
                replace(result, messages=(*result.messages, *deepcopy(messages)))
            ) from exc
        return replace(result, messages=(*result.messages, *deepcopy(messages)))

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
            revision = self.revision
            cutoff = max(item.visible_through_sequence for item in opportunities)
            by_player = {}
            for opportunity in opportunities:
                by_player.setdefault(
                    opportunity.player,
                    replace(opportunity, visible_through_sequence=cutoff),
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
                        returned, speaker=item.player, participants=context.participants
                    )
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
            for opportunity, choice in zip(ordered, choices):
                validation_error = withheld_error
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
                                intent=choice.intent,
                                causation_id=opportunity.cause.causation_id,
                                commitment=commitment,
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
                self.communication_trace.append(
                    CommunicationAdmission(
                        opportunity,
                        choice,
                        accepted=validation_error is None,
                        validation_error=validation_error,
                    )
                )
                if validation_error is None:
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
        ))

    async def _prompt_player(
        self,
        player: SandboxPlayer,
        context: PlayerContext,
        feedback: str | None,
    ) -> PlayerAttempt:
        return await player.choose(deepcopy(context), feedback)

    async def _get_action_from_player(
        self,
        player: SandboxPlayer,
        context: PlayerContext,
        *,
        revision: int,
        failed_attempts: list[PlayerAttempt] | None = None,
    ) -> tuple[Action, PlayerAttempt]:
        if failed_attempts is None:
            failed_attempts = []
        feedback = failed_attempts[-1].validation_error if failed_attempts else None
        for _ in range(len(failed_attempts), self.retry_policy.max_decision_attempts):
            self._check_revision(revision)
            returned = await self._prompt_player(player, context, feedback)
            action = None
            try:
                action = validate_player_attempt(context, returned)
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
                if action.action_type in {ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER}:
                    staged = copy(self.game_engine.state)
                    ensure_trade_window(staged).validate_offer(action.value)
                if not self.game_engine.is_action_valid(action):
                    raise ValueError(
                        "The selected action parameters are no longer legal or "
                        "affordable. Choose again from the current context."
                    )
            except ValueError as exc:
                feedback = str(exc)
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
        )

    def restore(self, snapshot: SandboxSnapshot) -> None:
        if self._step_state is not None:
            raise SandboxError("Cannot restore a sandbox while a step is in flight")
        snapshot_colors = set(snapshot.engine.state.colors)
        if set(self.players) != snapshot_colors:
            raise MissingPlayerError("Snapshot colors do not match registered sandbox players")
        self.game_engine.restore(snapshot.engine)
        for color, player_state in snapshot.player_states:
            self.players[color].restore(deepcopy(player_state))

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
        return build_decision_context(
            self.game_engine,
            actor,
            advertised_actions,
            context_revision=self.revision,
        )

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
