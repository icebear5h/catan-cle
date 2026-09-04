"""Lightweight asynchronous orchestration for one Catan game engine."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from cle.players.contracts import (
    CommunicationMode,
    PlayerAttempt,
    PlayerContext,
    SandboxPlayer,
    TalkContext,
)
from cle.sandbox.communication import CommunicationOpportunity, CommunicationPolicy
from cle.sandbox.contracts import (
    RetryPolicy,
    SandboxSnapshot,
    SandboxStepResult,
    SandboxView,
)
from cle.sandbox.decision import build_decision_context
from cle.game_engine.events import GameEvent
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import trade_response_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color


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
        self.attempts = attempts
        self.validation_error = validation_error
        super().__init__(
            f"Player {player} failed to choose a valid action after "
            f"{len(attempts)} attempts"
        )


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
        self.communication_trace: list[Any] = []
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
        if player.color not in self.game_engine.state.colors:
            raise ValueError(f"Player color {player.color} is not in this game")
        self.players[player.color] = player
        self._validate_players()

    def view(self, observer: Color | None = None) -> SandboxView:
        observer = observer or self.current_actor()
        observation = self.game_engine.observe(observer)
        current_actor = self.current_actor()
        return SandboxView(
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
        )

    async def step(self) -> SandboxStepResult:
        """Obtain one valid player action, apply it strictly, and acknowledge it."""
        if self.game_engine.winning_color() is not None:
            raise TerminalSandboxError("Cannot step a terminal game")

        barrier_contexts = self._trade_barrier_contexts()
        if barrier_contexts:
            return await self._step_barrier(barrier_contexts)

        playable_actions = tuple(self.game_engine.state.playable_actions)
        forced_roll = (
            len(playable_actions) == 1
            and playable_actions[0].action_type == ActionType.ROLL
        )
        pre_messages = ()
        if not forced_roll:
            pre_messages = await self._run_communication(
                self.communication_policy.pre_action(self.game_engine),
                max_rounds=1,
            )

        context = self.decision_context(self.current_actor())
        player = self.players.get(context.actor)
        if player is None:
            raise MissingPlayerError(f"No player is registered for {context.actor}")

        if forced_roll:
            action = context.legal_actions[0]
            attempt = None
        else:
            action, attempt = await self._get_action_from_player(player, context)

        transition = self.game_engine.step(action)
        post_messages = await self._run_communication(
            self.communication_policy.after_events(
                self.game_engine,
                transition.events,
                round_number=0,
            ),
            max_rounds=self.game_engine.communication_limits.max_general_reaction_rounds,
        )
        result = SandboxStepResult(
            contexts=() if attempt is None else (context,),
            attempts=() if attempt is None else (attempt,),
            transitions=(transition,),
            messages=(*pre_messages, *post_messages),
        )
        if attempt is not None:
            player.accept(attempt, result)
        return result

    async def _step_barrier(
        self,
        contexts: tuple[PlayerContext, ...],
    ) -> SandboxStepResult:
        players = tuple(self.players[context.actor] for context in contexts)
        choices = await asyncio.gather(
            *(
                self._get_action_from_player(player, context)
                for player, context in zip(players, contexts)
            )
        )
        transitions = tuple(
            self.game_engine.step(action)
            for action, _ in choices
        )
        window = self.game_engine.state.trade_window
        if window is not None and window.active_offers:
            window.advance_round()
        attempts = tuple(attempt for _, attempt in choices)
        message_events = await self._run_communication(
            self.communication_policy.after_events(
                self.game_engine,
                tuple(
                    event
                    for transition in transitions
                    for event in transition.events
                ),
                round_number=0,
            ),
            max_rounds=self.game_engine.communication_limits.max_trade_reaction_rounds,
        )
        result = SandboxStepResult(
            contexts=contexts,
            attempts=attempts,
            transitions=transitions,
            messages=message_events,
        )
        for player, context, attempt, transition in zip(
            players,
            contexts,
            attempts,
            transitions,
        ):
            player.accept(
                attempt,
                SandboxStepResult(
                    contexts=(context,),
                    attempts=(attempt,),
                    transitions=(transition,),
                ),
            )
        return result

    async def _run_communication(
        self,
        initial: tuple[CommunicationOpportunity, ...],
        *,
        max_rounds: int,
    ) -> tuple[GameEvent, ...]:
        opportunities = initial
        emitted: list[GameEvent] = []
        round_number = 0
        limits = self.game_engine.communication_limits
        while opportunities and round_number < max_rounds:
            by_player = {}
            for opportunity in opportunities:
                by_player.setdefault(opportunity.player, opportunity)
            ordered = tuple(
                by_player[color]
                for color in self.game_engine.state.colors
                if color in by_player
            )
            contexts = tuple(self._talk_context(item) for item in ordered)
            choices = await asyncio.gather(
                *(
                    self.players[item.player].communicate(context)
                    for item, context in zip(ordered, contexts)
                )
            )
            round_events = []
            for opportunity, choice in zip(ordered, choices):
                self.communication_trace.append((opportunity, choice))
                self.players[opportunity.player].acknowledge_events(
                    opportunity.visible_through_sequence + 1
                )
                if choice.mode == CommunicationMode.SILENCE:
                    continue
                if len(emitted) + len(round_events) >= limits.max_messages_per_window:
                    break
                commitment = (
                    (
                        choice.commitment.condition,
                        choice.commitment.promise,
                        choice.commitment.expires_turn,
                    )
                    if choice.commitment is not None
                    else None
                )
                round_events.append(
                    self.game_engine.append_message(
                        speaker=opportunity.player,
                        text=choice.text,
                        audience=choice.audience,
                        intent=choice.intent,
                        causation_id=opportunity.cause.causation_id,
                        commitment=commitment,
                    )
                )
            emitted.extend(round_events)
            round_number += 1
            opportunities = self.communication_policy.after_events(
                self.game_engine,
                tuple(round_events),
                round_number=round_number,
            )
        return tuple(emitted)

    def _talk_context(self, opportunity: CommunicationOpportunity) -> TalkContext:
        color = opportunity.player
        return TalkContext(
            context_id=(
                f"{self.game_engine.id}:talk:{opportunity.cause.causation_id}:"
                f"{opportunity.round}:{color.value}"
            ),
            player=color,
            participants=self.game_engine.state.colors,
            cause=opportunity.cause,
            visible_through_sequence=opportunity.visible_through_sequence,
            game_events=tuple(
                event
                for event in self.game_engine.project_game_events(color)
                if event.sequence <= opportunity.visible_through_sequence
            ),
            recent_messages=tuple(
                event
                for event in self.game_engine.project_messages(color)
                if event.sequence <= opportunity.visible_through_sequence
            ),
            active_commitments=self.game_engine.active_commitments(color),
        )

    async def _prompt_player(
        self,
        player: SandboxPlayer,
        context: PlayerContext,
        feedback: str | None,
    ) -> PlayerAttempt:
        return await player.choose(context, feedback)

    async def _get_action_from_player(
        self,
        player: SandboxPlayer,
        context: PlayerContext,
    ) -> tuple[Action, PlayerAttempt]:
        feedback = None
        failed_attempts = []
        for _ in range(self.retry_policy.max_decision_attempts):
            attempt = await self._prompt_player(player, context, feedback)
            if attempt.choice is None:
                self.decision_trace.append(attempt)
                failed_attempts.append(attempt)
                feedback = attempt.validation_error or "Return one valid action index."
                continue
            try:
                action = context.action_at(attempt.choice.action_index)
            except IndexError as exc:
                self.decision_trace.append(attempt)
                failed_attempts.append(attempt)
                feedback = str(exc)
                continue
            if (
                action.action_type
                in {ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER}
                and isinstance(action.value, str)
            ):
                if attempt.choice.trade_offer is None:
                    self.decision_trace.append(attempt)
                    failed_attempts.append(attempt)
                    feedback = "Selected trade action requires an exact trade_offer."
                    continue
                action = Action(
                    action.color,
                    action.action_type,
                    attempt.choice.trade_offer,
                )
            if not self.game_engine.is_action_valid(action):
                self.decision_trace.append(attempt)
                failed_attempts.append(attempt)
                feedback = (
                    "The selected action parameters are no longer legal or "
                    "affordable. Choose again from the current context."
                )
                continue
            return action, attempt
        raise PlayerResponseError(
            player.color,
            tuple(failed_attempts),
            feedback or "Return one valid action index.",
        )

    def snapshot(self) -> SandboxSnapshot:
        return SandboxSnapshot(
            engine=self.game_engine.snapshot(),
            player_states=tuple(
                (color, player.snapshot())
                for color, player in self.players.items()
            ),
        )

    def restore(self, snapshot: SandboxSnapshot) -> None:
        snapshot_colors = set(snapshot.engine.state.colors)
        if set(self.players) != snapshot_colors:
            raise MissingPlayerError(
                "Snapshot colors do not match registered sandbox players"
            )
        self.game_engine.restore(snapshot.engine)
        for color, player_state in snapshot.player_states:
            self.players[color].restore(player_state)

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
        mismatched = [
            color
            for color, player in self.players.items()
            if player.color != color
        ]
        if mismatched:
            raise ValueError(f"Player mapping keys do not match players: {mismatched}")
