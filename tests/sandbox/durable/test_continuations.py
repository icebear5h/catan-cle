from pathlib import Path

import pytest

from cle.sandbox.action_batches import AutomaticBatchAction
from cle.sandbox.durable import DurableSandbox, create_durable_sandbox
from cle.sandbox.factory import LiveSandboxConfig
from cle.traces.journal import SQLiteSandboxJournal

from .faults import FaultJournal, PersistenceFault
from .support import (
    POLICY,
    ScriptedTransport,
    SetupPair,
    agent,
    assert_checkpoint,
    assert_engine,
    assert_result,
    entry_kinds,
    fresh_sandbox,
    make_sandbox,
    setup_pair,
    successful,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("pair_index", [0, 3, 7])
async def test_pending_automatic_road_recovers_as_one_step_without_inference(
    tmp_path: Path, pair_index: int,
) -> None:
    pair = setup_pair(pair_index)
    actor = pair.settlement.color
    transport = ScriptedTransport(pair.response())
    sandbox, _ = make_sandbox(pair.before, {actor: transport})
    path = tmp_path / "automatic.sqlite3"
    runner = DurableSandbox.start(sandbox, SQLiteSandboxJournal(path), policy_identity=POLICY)
    first = await runner.step("settlement")
    first_result = successful(first)
    checkpoint = sandbox.snapshot()
    batch = checkpoint.pending_action_batch
    assert batch is not None and batch.next_index == 1
    assert batch.origin_context_id == first_result.attempts[0].context_id
    assert agent(sandbox, actor).session.memory_revision == 1

    # Crash after the queued road is applied but before its step outcome is saved.
    faulty = FaultJournal(path, "before_settle")
    interim, interim_scripts = fresh_sandbox(faulty, sandbox.game_engine.id)
    advancing = DurableSandbox.resume(interim, faulty, policy_identity=POLICY)
    with pytest.raises(PersistenceFault, match="before_settle"):
        await advancing.step("automatic-road")
    assert faulty.attempted_outcome is not None
    attempted = faulty.attempted_outcome.result
    assert attempted is not None
    assert attempted.transitions[0].requested_action == pair.road
    assert not any(script.requests for script in interim_scripts.values())
    assert_checkpoint(interim, checkpoint)

    reopened = SQLiteSandboxJournal(path)
    fresh, scripts = fresh_sandbox(reopened, sandbox.game_engine.id)
    resumed = DurableSandbox.resume(
        fresh, reopened, policy_identity=POLICY, recover_pending=True,
    )
    assert resumed.pending_command_id == "automatic-road"
    road = await resumed.step("automatic-road")
    result = successful(road)
    assert result.contexts == ()
    assert result.attempts == ()
    assert len(result.transitions) == 1
    assert result.transitions[0].requested_action == pair.road
    assert isinstance(result.automatic_action, AutomaticBatchAction)
    assert result.automatic_action.batch == batch
    assert result.automatic_action.to_payload()["action_number"] == 2
    assert result.automatic_action.to_payload()["provider_response_id"] == batch.provider_response_id
    assert fresh.snapshot().pending_action_batch is None
    assert fresh.snapshot().player_states == checkpoint.player_states
    assert_engine(fresh.game_engine, pair.after_road)
    assert not any(script.requests for script in scripts.values())
    assert len(transport.requests) == 1

    latest = fresh.snapshot()
    assert_result(successful(await resumed.step("automatic-road")), result)
    assert_result(successful(await resumed.step("settlement")), first_result)
    assert_checkpoint(fresh, latest)
    kinds = entry_kinds(reopened, fresh.game_engine.id)
    assert kinds.count("step_started") == kinds.count("step_succeeded") == 2
    assert kinds.count("call_started") == kinds.count("call_completed") == 1


@pytest.mark.asyncio
async def test_factory_resume_restores_agent_and_batch_without_new_inference(
    tmp_path: Path, pair: SetupPair,
) -> None:
    config = LiveSandboxConfig(
        mode="llm", seed=7, shuffle_players=False, palette="canonical_four",
        max_decision_attempts=1,
    )
    transport = ScriptedTransport(pair.response())
    journal = SQLiteSandboxJournal(tmp_path / "factory.sqlite3")
    runner = create_durable_sandbox(config, journal, transport=transport)
    first = await runner.step("settlement")
    first_result = successful(first)
    sandbox_id = runner.sandbox.game_engine.id
    checkpoint = runner.sandbox.snapshot()
    assert checkpoint.pending_action_batch is not None
    assert first_result.transitions[0].requested_action == pair.settlement
    assert transport.requests[0].prompt_sources

    unused = ScriptedTransport()
    resumed = create_durable_sandbox(
        config, SQLiteSandboxJournal(journal.path), sandbox_id=sandbox_id, transport=unused,
    )
    assert resumed.policy_identity == runner.policy_identity
    assert_checkpoint(resumed.sandbox, checkpoint)
    road = successful(await resumed.step("road"))
    assert road.transitions[0].requested_action == pair.road
    assert resumed.sandbox.snapshot().player_states == checkpoint.player_states
    assert resumed.sandbox.snapshot().pending_action_batch is None
    assert len(transport.requests) == 1 and not unused.requests
