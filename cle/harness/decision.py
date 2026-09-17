"""Shared non-mutating model-decision entry point."""

from __future__ import annotations

from cle.harness.communication import CommunicationSuite
from cle.harness.models import CompletionTransport
from cle.harness.suite import ContextSuite, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.contracts import PlayerAttempt, PlayerContext
from cle.players.notes import validate_notes


async def request_player_attempt(
    context: PlayerContext,
    transport: CompletionTransport,
    *,
    game_plan: str = "",
    session_id: str | None = None,
    suite: ContextSuite | None = None,
    communication_suite: CommunicationSuite | None = None,
    notes_seed: str | None = None,
) -> PlayerAttempt:
    """Request one unaccepted choice; an explicit notes seed belongs to context.actor."""
    active_suite = suite or load_context_suite()
    player = AgentPlayer(
        context.actor,
        transport,
        session_id=session_id or context.context_id,
        suite=active_suite,
        communication_suite=communication_suite,
    )
    if active_suite.context.memory_mode == "fresh_notes":
        if notes_seed is not None:
            player.session.strategic_memory = validate_notes(
                notes_seed, active_suite.context.max_notes_chars,
            )
    else:
        player.session.strategic_memory = game_plan
    return await player.choose(context)
