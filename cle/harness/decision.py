"""Shared non-mutating model-decision entry point."""

from __future__ import annotations

from cle.harness.models import CompletionTransport
from cle.harness.suite import ContextSuite, load_context_suite
from cle.players.agent import AgentPlayer
from cle.players.contracts import PlayerAttempt, PlayerContext


async def request_player_attempt(
    context: PlayerContext,
    transport: CompletionTransport,
    *,
    game_plan: str = "",
    session_id: str | None = None,
    suite: ContextSuite | None = None,
) -> PlayerAttempt:
    """Request one choice through the same agent contract used in live games."""
    active_suite = suite or load_context_suite()
    player = AgentPlayer(
        context.actor,
        transport,
        session_id=session_id or context.context_id,
        suite=active_suite,
    )
    player.session.strategic_memory = game_plan
    return await player.choose(context)
