import asyncio
from pathlib import Path

import pytest

from cle.game_engine.models.player import Color
from cle.harness.catan_board_surface import ImageBoardPresenter
from cle.harness.suite import default_suite_path
from cle.sandbox.durable import DurableSandbox, create_durable_sandbox
from cle.sandbox.factory import LiveSandboxConfig
from cle.traces.journal import CommandTicket, JournalConflict, SQLiteSandboxJournal

from .support import (
    POLICY,
    ScriptedTransport,
    SetupPair,
    agent,
    assert_checkpoint,
    fresh_sandbox,
    make_sandbox,
    seven_engine,
    successful,
)


@pytest.mark.asyncio
async def test_settled_delivery_retry_keeps_original_identity_after_policy_change(
    tmp_path: Path, pair: SetupPair,
) -> None:
    sandbox, _ = make_sandbox(pair.before, {Color.RED: ScriptedTransport(pair.response())})
    journal = SQLiteSandboxJournal(tmp_path / "changed-policy.sqlite3")
    old = DurableSandbox.start(sandbox, journal, policy_identity=POLICY)
    settled = await old.step("placement")
    fresh, scripts = fresh_sandbox(journal, sandbox.game_engine.id)
    new = DurableSandbox.resume(fresh, journal, policy_identity="changed-model")
    replay = await new.step("placement")
    assert replay.ticket.policy_identity == POLICY
    assert replay.finished_sequence == settled.finished_sequence
    assert not any(script.requests for script in scripts.values())
    assert successful(await new.step("road")).transitions[0].requested_action == pair.road
    assert journal.command(fresh.game_engine.id, "road").ticket.policy_identity == "changed-model"


@pytest.mark.parametrize("change", ["model", "endpoint"])
def test_factory_fences_environment_selected_policy_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    # Construct real local HTTP transports but never dispatch a network request.
    monkeypatch.setenv("CATAN_LLM_MODEL", "immutable-checkpoint-a")
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:19191/v1")
    journal = SQLiteSandboxJournal(tmp_path / "environment.sqlite3")
    config = LiveSandboxConfig(mode="llm", seed=7, reasoning={"enabled": False})
    runner = create_durable_sandbox(config, journal)
    sandbox_id = runner.sandbox.game_engine.id
    journal.begin(CommandTicket(sandbox_id, "pending", 1, 0, runner.policy_identity))
    if change == "model":
        monkeypatch.setenv("CATAN_LLM_MODEL", "immutable-checkpoint-b")
    else:
        monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:19192/v1")
    with pytest.raises(JournalConflict, match="original policy"):
        create_durable_sandbox(config, journal, sandbox_id=sandbox_id, recover_pending=True)
    assert journal.head(sandbox_id).generation == 1


@pytest.mark.asyncio
async def test_barrier_provider_exception_is_not_embedded_in_sibling_diagnostics(tmp_path: Path) -> None:
    secret = "https://provider.invalid/?api_key=must-not-be-stored"
    oracle = seven_engine(discards=True)
    red = ScriptedTransport(RuntimeError(secret), gated=True)
    white = ScriptedTransport('{"tool":"discard","arguments":{"cards":{"WOOD":5}}}')
    sandbox, _ = make_sandbox(oracle.snapshot(), {Color.RED: red, Color.WHITE: white})
    journal = SQLiteSandboxJournal(tmp_path / "exception.sqlite3")
    runner = DurableSandbox.start(sandbox, journal, policy_identity=POLICY)
    task = asyncio.create_task(runner.step("discard"))
    await asyncio.wait_for(white.completed.wait(), 3)
    red.release.set()
    record = await asyncio.wait_for(task, 3)
    assert record.outcome is not None and record.outcome.status == "failed"
    assert record.outcome.attempts
    for attempt in record.outcome.attempts:
        assert secret not in (attempt.validation_error or "")
    assert secret not in str(journal.command(oracle.id, "discard").outcome)


@pytest.mark.asyncio
async def test_image_provenance_is_logged_without_copying_png_into_outcome(
    tmp_path: Path, pair: SetupPair,
) -> None:
    script = ScriptedTransport(pair.response())
    sandbox, _ = make_sandbox(pair.before, {Color.RED: script})
    agent(sandbox, Color.RED)._assembler.board_presenter = ImageBoardPresenter(image_size=128)
    journal = SQLiteSandboxJournal(tmp_path / "image.sqlite3")
    runner = DurableSandbox.start(sandbox, journal, policy_identity=POLICY)
    record = await runner.step("placement")
    request = successful(record).attempts[0].model_request
    assert request is not None and request.board_presentation is None
    assert script.requests[0].board_presentation is not None
    started = next(entry for entry in journal.entries(sandbox.game_engine.id) if entry.kind == "call_started")
    assert script.requests[0].board_presentation.content_sha256 in started.payload_json
    assert started.call_key is not None
    receipt = journal.get_call(sandbox.game_engine.id, "placement", started.call_key)
    assert receipt.response is not None and receipt.response.content == pair.response()
    assert_checkpoint(sandbox, journal.head(sandbox.game_engine.id).snapshot)


@pytest.mark.asyncio
async def test_idle_factory_migration_survives_claim_and_next_commit(
    tmp_path: Path, pair: SetupPair, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CATAN_PROMPT_SUITE_FOLLOW_LATEST", raising=False)
    store = SQLiteSandboxJournal(tmp_path / "migration.sqlite3")
    config = LiveSandboxConfig(
        mode="llm", seed=7, shuffle_players=False, palette="canonical_four",
        context_suite_path=str(default_suite_path().with_name("catan_v9.yaml")),
    )
    original = create_durable_sandbox(config, store, transport=ScriptedTransport())
    assert agent(original.sandbox, Color.RED).session.context_policy == "legacy"
    current = LiveSandboxConfig(mode="llm", seed=7, shuffle_players=False, palette="canonical_four")
    restored = create_durable_sandbox(
        current, store, sandbox_id=original.sandbox.game_engine.id,
        transport=ScriptedTransport(pair.response()),
    )
    assert agent(restored.sandbox, Color.RED).session.context_policy == "fresh_notes"
    assert agent(restored.sandbox, Color.RED).session.memory_revision == 1
    assert successful(await restored.step("placement")).transitions[0].requested_action == pair.settlement
    assert_checkpoint(restored.sandbox, store.head(restored.sandbox.game_engine.id).snapshot)
