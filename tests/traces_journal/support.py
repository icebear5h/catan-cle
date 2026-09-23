import asyncio
from dataclasses import replace

from cle.game_engine.models.player import Color
from cle.harness.models import PlayerSession
from cle.players.agent import AgentPlayerSnapshot
from cle.players.data import PlayerSnapshot
from cle.sandbox import CatanSandbox
from cle.sandbox.contracts import SandboxSnapshot
from cle.traces.journal import CommandOutcome, CommandTicket, SQLiteSandboxJournal

POLICY = "first-legal-v1"
REQUEST_JSON = '{ "model": "local", "messages": [{"role":"user","content":"choose"}] }'
REQUEST_SHA = "1234567890abcdef" * 4


def ticket_for(
    journal: SQLiteSandboxJournal, sandbox: CatanSandbox, command_id: str = "step-1"
) -> CommandTicket:
    head = journal.head(sandbox.game_engine.id)
    return CommandTicket(
        head.sandbox_id, command_id, head.generation, head.checkpoint_sequence, POLICY,
    )


def advance(sandbox: CatanSandbox) -> CommandOutcome:
    before = sandbox.revision
    result = asyncio.run(sandbox.step())
    return CommandOutcome("succeeded", sandbox.snapshot(), before, result=result)


def with_notes(snapshot: SandboxSnapshot, text: str, revision: int) -> SandboxSnapshot:
    color = snapshot.player_states[0][0]
    session = PlayerSession(
        color, "notes-session", strategic_memory=text, memory_revision=revision,
        context_policy="fresh_notes",
    )
    states: tuple[tuple[Color, PlayerSnapshot], ...] = (
        (color, AgentPlayerSnapshot(session.snapshot())), *snapshot.player_states[1:],
    )
    return replace(snapshot, player_states=states)
