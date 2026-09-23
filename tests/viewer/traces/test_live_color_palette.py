from typing import Any

from cle.game_engine.game import GameEngine
from cle.game_engine.models.coordinate_system import Coordinate
from cle.game_engine.models.enums import FastResource
from cle.game_engine.models.player import Color
from cle.harness.models import ModelRequest
from cle.players.agent import AgentPlayer
from cle.sandbox.factory import LiveSandboxConfig, create_live_sandbox
from cle.sandbox.palette import (
    ALL_COLORS,
    CANONICAL_FOUR,
    balanced_datagen_colors,
    select_game_colors,
)


class UnusedTransport:
    async def complete(self, request: ModelRequest) -> None:
        raise AssertionError(f"transport should not be called: {request}")


def board_fingerprint(
    engine: GameEngine,
) -> tuple[tuple[Coordinate, FastResource | None, int | None], ...]:
    return tuple(
        sorted(
            (coordinate, tile.resource, tile.number)
            for coordinate, tile in engine.state.board.map.land_tiles.items()
        )
    )


def test_seeded_random_palette_is_distinct_deterministic_and_rng_isolated() -> None:
    first = select_game_colors("random_all", seed=123)
    second = select_game_colors("random_all", seed=123)

    assert first == second
    assert len(first) == len(set(first)) == 4
    assert set(first) <= set(ALL_COLORS)

    before_palette = GameEngine(first, seed=123, shuffle_players=False)
    select_game_colors("random_all", seed=999)
    after_palette = GameEngine(first, seed=123, shuffle_players=False)
    assert board_fingerprint(before_palette) == board_fingerprint(after_palette)
    assert before_palette.state.rng.getstate() == after_palette.state.rng.getstate()


def test_canonical_palette_and_balanced_datagen_cover_all_colors() -> None:
    assert select_game_colors("canonical_four", seed=1) == CANONICAL_FOUR

    selected = {
        color
        for index in range(len(ALL_COLORS))
        for color in balanced_datagen_colors(index, seed=77)
    }
    assert selected == set(ALL_COLORS)


def test_snapshot_restore_uses_exact_realized_colors() -> None:
    original = create_live_sandbox(
        LiveSandboxConfig(seed=88, shuffle_players=False, palette="random_all")
    )
    snapshot = original.snapshot()

    restored = create_live_sandbox(
        LiveSandboxConfig(seed=999, shuffle_players=False, palette="canonical_four"),
        snapshot=snapshot,
    )

    assert restored.game_engine.state.colors == original.game_engine.state.colors
    assert set(restored.players) == set(original.players)


def test_llm_vs_random_owns_first_realized_color_when_red_is_absent() -> None:
    seed: Any = next(
        candidate
        for candidate in range(1000)
        if Color.RED not in select_game_colors("random_all", seed=candidate)
    )
    sandbox = create_live_sandbox(
        LiveSandboxConfig(
            mode="llm_vs_random",
            seed=seed,
            shuffle_players=False,
            palette="random_all",
        ),
        transport=UnusedTransport(),
    )

    first_color = sandbox.game_engine.state.colors[0]
    assert first_color != Color.RED
    assert isinstance(sandbox.players[first_color], AgentPlayer)
    assert all(
        player.status()["kind"] == ("agent" if color == first_color else "first_legal")
        for color, player in sandbox.players.items()
    )
