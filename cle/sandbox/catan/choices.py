"""Detached decision contexts, retries, and strict action preflight."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state import GameState
from cle.harness.models import ModelRequest
from cle.players.contracts import (
    CommunicationChoice,
    PlayerAttempt,
    PlayerChoice,
    PlayerContext,
    SandboxPlayer,
    TalkContext,
)
from cle.players.notes import MAX_NOTES_CHARS
from cle.sandbox import catan

from .support import PlayerResponseError, SandboxError

if TYPE_CHECKING:
    from . import CatanSandbox


def _validate_context_update(
    player: SandboxPlayer, request: ModelRequest | None, context: PlayerContext | TalkContext,
) -> None:
    validator = getattr(player, "validate_context_update", None)
    if validator is not None:
        validator(catan.deepcopy(request), catan.deepcopy(context))


async def _prompt_player(
    self: CatanSandbox,
    player: SandboxPlayer,
    context: PlayerContext,
    feedback: str | None,
) -> PlayerAttempt:
    return await player.choose(catan.deepcopy(context), feedback)


def _retry_feedback(player: SandboxPlayer, context: PlayerContext, error: str) -> str:
    """Validation error plus the actor's exact legal tools.

    Base prompts intentionally omit the concrete menu; after a failure the
    model has proven it cannot infer legality, so the correction names it.
    """
    tools = ", ".join(catan.legal_tool_names(context))
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
    self: CatanSandbox,
    player: SandboxPlayer,
    context: PlayerContext,
    *,
    revision: int,
    failed_attempts: list[PlayerAttempt] | None = None,
) -> tuple[Action | CommunicationChoice, PlayerAttempt]:
    if failed_attempts is None:
        failed_attempts = []
    last_error = failed_attempts[-1].validation_error if failed_attempts else None
    feedback = self._retry_feedback(player, context, last_error) if last_error else None
    for _ in range(len(failed_attempts), self.retry_policy.max_decision_attempts):
        self._check_revision(revision)
        returned = await self._prompt_player(player, context, feedback)
        action = None
        try:
            action = catan.validate_player_attempt(
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
            attempt = catan.deepcopy(attempt)
        except TypeError as exc:
            action = None
            attempt = PlayerAttempt(context.context_id, None, f"Player attempt cannot be detached: {exc}")
        try:
            self._check_revision(revision)
        except SandboxError as exc:
            self.decision_trace.append(catan.replace(attempt, choice=None, validation_error=str(exc)))
            raise
        try:
            if action is None:
                raise ValueError(attempt.validation_error)
            self._validate_context_update(player, attempt.model_request, context)
            if isinstance(action, CommunicationChoice):
                if not self._reactive(context.actor):
                    raise ValueError("Standalone speech requires the reactive contract")
                return action, attempt
            assert isinstance(attempt.choice, PlayerChoice)
            if attempt.choice.batch_actions:
                if not getattr(getattr(getattr(player, "suite", None), "context", None), "deterministic_batches", False):
                    raise ValueError("Action batches require an opted-in shared/fresh player")
                first = attempt.choice.batch_actions[0]
                tool, arguments = first["tool"], first["arguments"]
                assert isinstance(tool, str) and isinstance(arguments, dict)
                bound = catan.parse_tool_choice(context, tool, arguments, shared=True)
                if catan.action_from_choice(context, bound) != action:
                    raise ValueError("Batch first action does not match the selected action")
                if action not in catan.generate_playable_actions(self.game_engine.state):
                    raise ValueError("Batch first action is no longer legal or affordable")
            if action.action_type in {ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER}:
                staged_state = catan.copy(self.game_engine.state)
                catan.ensure_trade_window(staged_state).validate_offer(action.value)
            if not self._is_action_valid(action):
                raise ValueError(
                    "The selected action parameters are no longer legal or "
                    "affordable. Choose again from the current context."
                )
            followup = catan.choice_followup_action(context, attempt.choice)
            if followup is not None:
                staged = catan.deepcopy(self.game_engine)
                transition = staged.step(action)
                if transition.winner is None:
                    staged.step(followup)
        except ValueError as exc:
            feedback = self._retry_feedback(player, context, str(exc))
            rejected = catan.replace(
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
        player.color, tuple(failed_attempts), feedback or "Return one valid action index.",
    )


def _is_action_valid(self: CatanSandbox, action: Action) -> bool:
    state = self.game_engine.state
    if (
        action.action_type == ActionType.DISCARD
        and state.current_prompt == ActionPrompt.DISCARD
        and action.color in state.colors[state.current_player_index + 1:]
    ):
        # The engine asks discarders one seat at a time. A later discarder in
        # the barrier is checked from their own seat; the barrier still replays
        # every discard in engine order before anything commits.
        staged = catan.copy(state)
        staged.current_player_index = state.colors.index(action.color)
        validate: Callable[[GameState, Action], bool] = catan.is_valid_action
        return self.game_engine.winning_color() is None and validate(staged, action)
    return self.game_engine.is_action_valid(action)


def decision_context(
    self: CatanSandbox,
    actor: Color | None = None,
    advertised_actions: tuple[Action, ...] | None = None,
) -> PlayerContext:
    context = catan.build_decision_context(
        self.game_engine, actor, advertised_actions, context_revision=self.revision,
    )
    return catan.replace(context, speech_allowed=(
        advertised_actions is None and self._reactive(context.actor) and not self._speech_used
        and (self._speech_calls_remaining is None or self._speech_calls_remaining > 0)
    ))
