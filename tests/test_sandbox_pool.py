import asyncio

import pytest

from cle.players.baseline import FirstLegalPlayer
from cle.sandbox import CatanSandbox, SandboxPool
from game_engine.game import GameEngine
from game_engine.models.player import Color


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
