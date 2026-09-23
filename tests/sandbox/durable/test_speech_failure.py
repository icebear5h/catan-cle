from pathlib import Path

import pytest

from cle.game_engine.models.player import Color
from cle.sandbox.durable import DurableSandbox
from cle.traces.journal import SQLiteSandboxJournal

from .support import (
    POLICY,
    ScriptedTransport,
    agent,
    assert_checkpoint,
    assert_engine,
    fresh_sandbox,
    make_sandbox,
    seven_engine,
)


@pytest.mark.asyncio
async def test_same_revision_failure_preserves_accepted_notes_and_resumes_remaining_speech(
    tmp_path: Path,
) -> None:
    oracle = seven_engine(discards=False)
    before = oracle.snapshot()
    blue = ScriptedTransport('{"mode":"pass","notes":"Keep this silent observation"}')
    white = ScriptedTransport(RuntimeError("speech provider unavailable"))
    sandbox, scripts = make_sandbox(before, {Color.BLUE: blue, Color.WHITE: white})
    journal = SQLiteSandboxJournal(tmp_path / "speech.sqlite3")
    runner = DurableSandbox.start(sandbox, journal, policy_identity=POLICY)
    failed = await runner.step("speech-failure")
    outcome = failed.outcome
    assert outcome is not None and outcome.status == "failed"
    assert outcome.error_type == "RuntimeError" and outcome.result is None
    assert outcome.before_revision == sandbox.revision == len(before.events)
    assert_engine(sandbox.game_engine, before)
    assert len(outcome.communications) == 1
    assert outcome.communications[0].accepted
    assert outcome.communications[0].opportunity.player == Color.BLUE
    accepted_blue = agent(sandbox, Color.BLUE).snapshot()
    assert accepted_blue.session.strategic_memory == "Keep this silent observation"
    assert accepted_blue.session.memory_revision == 1
    assert accepted_blue.session.talk_next_sequence == sandbox.revision
    assert accepted_blue.session.action_next_sequence == 0
    assert len(accepted_blue.session.communication_receipts) == 1
    checkpoint = outcome.snapshot
    assert [item.player for item in checkpoint.pending_reactions] == [Color.WHITE, Color.ORANGE]
    assert checkpoint.speech_calls_remaining == 10
    assert len(blue.requests) == len(white.requests) == 1
    assert not scripts[Color.RED].requests and not scripts[Color.ORANGE].requests

    reopened = SQLiteSandboxJournal(journal.path)
    fresh, resumed_scripts = fresh_sandbox(reopened, oracle.id, {
        Color.WHITE: ScriptedTransport('{"mode":"pass","notes":"white recovered"}'),
        Color.ORANGE: ScriptedTransport('{"mode":"pass","notes":"orange heard"}'),
        Color.RED: ScriptedTransport(RuntimeError("action provider unavailable")),
    })
    resumed = DurableSandbox.resume(fresh, reopened, policy_identity=POLICY)
    assert_checkpoint(fresh, checkpoint)
    replay = await resumed.step("speech-failure")
    assert replay.finished_sequence == failed.finished_sequence
    assert replay.outcome is not None and replay.outcome.status == "failed"
    assert not any(script.requests for script in resumed_scripts.values())

    # An intentional new command continues after BLUE's accepted pass. Even a
    # later action failure must retain those additional notes at this revision.
    continued = await resumed.step("continue-after-speech-failure")
    assert continued.outcome is not None and continued.outcome.status == "failed"
    assert continued.outcome.error_type == "RuntimeError"
    assert continued.finished_sequence is not None and failed.finished_sequence is not None
    assert continued.finished_sequence > failed.finished_sequence
    assert_engine(fresh.game_engine, before)
    assert agent(fresh, Color.BLUE).snapshot() == accepted_blue
    assert not resumed_scripts[Color.BLUE].requests
    assert len(resumed_scripts[Color.WHITE].requests) == 1
    assert len(resumed_scripts[Color.ORANGE].requests) == 1
    assert len(resumed_scripts[Color.RED].requests) == 1
    assert agent(fresh, Color.WHITE).session.strategic_memory == "white recovered"
    assert agent(fresh, Color.ORANGE).session.strategic_memory == "orange heard"
    assert fresh.snapshot().pending_reactions == ()
    assert fresh.snapshot().speech_calls_remaining == 8
    assert [item.opportunity.player for item in continued.outcome.communications] == [
        Color.WHITE, Color.ORANGE,
    ]
    assert all(item.accepted for item in continued.outcome.communications)
    assert_checkpoint(fresh, reopened.head(oracle.id).snapshot)
