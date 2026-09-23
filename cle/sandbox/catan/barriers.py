"""Concurrent trade/discard acquisition with ordered atomic admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.players.contracts import PlayerAttempt, PlayerChoice, PlayerContext
from cle.players.notes import MAX_NOTES_CHARS
from cle.sandbox import catan
from cle.sandbox.contracts import SandboxStepResult

from .support import _TRADE_BARRIER_ACTIONS

if TYPE_CHECKING:
    from . import CatanSandbox


async def _step_barrier(
    self: CatanSandbox,
    contexts: tuple[PlayerContext, ...],
    *,
    allowed: frozenset[ActionType] = _TRADE_BARRIER_ACTIONS,
    label: str = "trade",
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
        assert isinstance(action, Action)
        return action, attempt

    try:
        choices = await catan._gather_or_cancel(acquire(index) for index in range(len(contexts)))
        # Revalidate the complete batch after every retry, including engine
        # legality and funding, not only trade-window capacity.
        while True:
            self._check_revision(revision)
            staged = catan.deepcopy(self.game_engine)
            for index, (context, (_, attempt)) in enumerate(zip(contexts, choices)):
                try:
                    assert isinstance(attempt.choice, PlayerChoice)
                    action = catan.action_from_choice(
                        context, attempt.choice,
                        max_notes_chars=getattr(players[index], "max_notes_chars", MAX_NOTES_CHARS),
                    )
                    self._validate_context_update(players[index], attempt.model_request, context)
                    if action.action_type not in allowed:
                        raise ValueError(f"Unsupported {label} barrier response")
                    if action.action_type == ActionType.COUNTER_OFFER:
                        catan.ensure_trade_window(staged.state).validate_offer(action.value)
                    staged.step(catan.deepcopy(action))
                except ValueError as exc:
                    pending.pop(context.actor)
                    rejected = catan.replace(attempt, validation_error=str(exc))
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
            catan.replace(
                pending[context.actor],
                validation_error=(
                    f"Decision withheld: {label} barrier did not commit: "
                    f"{str(exc) or type(exc).__name__}"
                ),
            )
            for context in contexts
            if context.actor in pending
        )
        raise
    # No player callbacks or awaits may split an admitted live batch.
    transitions = tuple(
        catan.deepcopy(self.game_engine.step(action)) for action, _ in choices
    )
    window = self.game_engine.state.trade_window
    if window is not None and window.active_offers:
        window.advance_round()
        self.game_engine.state.playable_actions = catan.generate_playable_actions(self.game_engine.state)
    if label == "trade" and self._trade_preauthorization is not None:
        self._trade_preauthorization = catan.replace(
            self._trade_preauthorization, response_revision=self.revision,
        )
    result = SandboxStepResult(
        contexts=contexts,
        attempts=tuple(attempt for _, attempt in choices),
        transitions=transitions,
    )
    for player, context, (_, attempt), transition in zip(players, contexts, choices, transitions):
        accepted_attempt, accepted_result = catan.deepcopy((
            attempt,
            SandboxStepResult(contexts=(context,), attempts=(attempt,), transitions=(transition,)),
        ))
        player.accept(accepted_attempt, accepted_result)
    limits = self.game_engine.communication_limits
    return await self._post_action_communication(
        result,
        max_rounds=(
            limits.max_trade_reaction_rounds if label == "trade"
            else limits.max_general_reaction_rounds
        ),
    )


def _discard_barrier_contexts(self: CatanSandbox) -> tuple[PlayerContext, ...]:
    """Prompt every remaining discarder of one 7 together, as at a real table.

    Each discard touches only that player's hand and the bank, so the choices
    are independent. They still commit in the engine's seat order.
    """
    state = self.game_engine.state
    if state.current_prompt != ActionPrompt.DISCARD or self._pending_decision_revision is not None:
        return ()
    # Seats before the current one already discarded and may still hold > limit.
    discarders = tuple(
        color for color in state.colors[state.current_player_index:]
        if catan.player_num_resource_cards(state, color) > state.discard_limit
    )
    if len(discarders) < 2:
        return ()
    return tuple(
        self.decision_context(color, (Action(color, ActionType.DISCARD, None),))
        for color in discarders
    )


def _trade_barrier_contexts(self: CatanSandbox) -> tuple[PlayerContext, ...]:
    state = self.game_engine.state
    contexts = []
    window = state.trade_window
    for color in state.colors:
        if window is not None and color == window.turn_player:
            continue
        actions = tuple(catan.trade_response_actions(state, color))
        if actions:
            contexts.append(self.decision_context(color, actions))
    return tuple(contexts)
