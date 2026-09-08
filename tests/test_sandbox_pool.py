import asyncio

import pytest

from cle.players.baseline import FirstLegalPlayer
from cle.sandbox import CatanSandbox, SandboxPool
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


@pytest.mark.asyncio
async def test_pool_runs_at_least_64_sandboxes_concurrently():
    active = 0
    peak = 0

    class MeasuredPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return await super().choose(context, feedback)

    sandboxes = []
    for seed in range(64):
        engine = GameEngine(COLORS, seed=seed, shuffle_players=False)
        players = {color: FirstLegalPlayer(color) for color in COLORS}
        players[Color.RED] = MeasuredPlayer(Color.RED)
        sandboxes.append(CatanSandbox(engine, players))

    results = await SandboxPool().step_many(sandboxes)

    assert peak == 64
    assert len(results) == 64
    assert all(sandbox.revision == 1 for sandbox in sandboxes)


@pytest.mark.asyncio
async def test_pool_rejects_duplicate_sandbox_in_one_batch():
    engine = GameEngine(COLORS, seed=1, shuffle_players=False)
    sandbox = CatanSandbox(
        engine,
        {color: FirstLegalPlayer(color) for color in COLORS},
    )

    with pytest.raises(ValueError, match="only once"):
        await SandboxPool().step_many((sandbox, sandbox))


@pytest.mark.asyncio
async def test_pool_reserves_identities_before_scheduling_children():
    entered = asyncio.Event()
    release = asyncio.Event()

    class PendingPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            entered.set()
            await release.wait()
            return await super().choose(context, feedback)

    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = PendingPlayer(Color.RED)
    sandbox = CatanSandbox(engine, players)
    pool = SandboxPool()
    first = asyncio.create_task(pool.step_many((sandbox,)))
    second = asyncio.create_task(pool.step_many((sandbox,)))
    with pytest.raises(RuntimeError, match="in flight"):
        await second
    await asyncio.wait_for(entered.wait(), 1)
    release.set()
    await first

    assert engine.revision == 1
    assert players[Color.RED].accepted_choices == 1
    assert pool._active == set()
    await pool.step_many((sandbox,))
    assert engine.revision == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_pool_failure_cleans_up_siblings_before_releasing_reservations(cancel):
    entered = asyncio.Event()
    never = asyncio.Event()
    cleaned_up = asyncio.Event()

    class PendingPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            entered.set()
            try:
                await never.wait()
            finally:
                await asyncio.sleep(0)
                cleaned_up.set()

    class FailingPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            await entered.wait()
            if cancel:
                await never.wait()
            raise RuntimeError("provider failed")

    sandboxes = []
    for player_type in (PendingPlayer, FailingPlayer):
        engine = GameEngine(COLORS, seed=7, shuffle_players=False)
        players = {color: FirstLegalPlayer(color) for color in COLORS}
        players[Color.RED] = player_type(Color.RED)
        sandboxes.append(CatanSandbox(engine, players))
    pool = SandboxPool()
    pending = asyncio.create_task(pool.step_many(sandboxes))
    await asyncio.wait_for(entered.wait(), 1)
    if cancel:
        pending.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await pending

    assert cleaned_up.is_set()
    assert pool._active == set()
    assert all(sandbox.revision == 0 for sandbox in sandboxes)
    for sandbox in sandboxes:
        sandbox.register_player(FirstLegalPlayer(Color.RED))
    await pool.step_many(sandboxes)
    assert all(sandbox.revision == 1 for sandbox in sandboxes)


@pytest.mark.asyncio
async def test_pool_cancellation_releases_semaphore_waiters_and_reservations():
    entered = asyncio.Event()
    never = asyncio.Event()
    calls = []

    class PendingPlayer(FirstLegalPlayer):
        async def choose(self, context, feedback=None):
            calls.append(context.context_id)
            entered.set()
            await never.wait()

    sandboxes = []
    for _ in range(2):
        engine = GameEngine(COLORS, seed=7, shuffle_players=False)
        players = {color: FirstLegalPlayer(color) for color in COLORS}
        players[Color.RED] = PendingPlayer(Color.RED)
        sandboxes.append(CatanSandbox(engine, players))
    pool = SandboxPool(max_concurrent_steps=1)
    pending = asyncio.create_task(pool.step_many(sandboxes))
    await asyncio.wait_for(entered.wait(), 1)
    with pytest.raises(RuntimeError, match="in flight"):
        await pool.step_many((sandboxes[1],))
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert len(calls) == 1
    assert pool._active == set()
    for sandbox in sandboxes:
        sandbox.register_player(FirstLegalPlayer(Color.RED))
    await pool.step_many(sandboxes)
    assert all(sandbox.revision == 1 for sandbox in sandboxes)
