"""Setup-phase prompt routing and snake order."""
from dataclasses import replace
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness import (
    ContextAssembler,
    PlayerSession,
    default_suite_path,
    load_context_suite,
)
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox import CatanSandbox

from .support import COLORS, NoCommunicationPolicy, _sandbox_and_player, allows_deprecated_suite


@pytest.mark.asyncio
async def test_setup_prompt_routing_distinguishes_both_settlements_and_roads() -> None:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    sandbox = CatanSandbox(
        engine,
        {color: FirstLegalPlayer(color) for color in COLORS},
        communication_policy=NoCommunicationPolicy(),
    )
    prompt_keys = {color: [] for color in COLORS}

    for _ in range(16):
        context = sandbox.decision_context()
        prompt_keys[context.actor].append(context.prompt_key)
        await sandbox.step()

    assert all(
        keys
        == [
            "initial_settlement_1",
            "initial_road_1",
            "initial_settlement_2",
            "initial_road_2",
        ]
        for keys in prompt_keys.values()
    )


@allows_deprecated_suite
@pytest.mark.asyncio
async def test_setup_order_uses_realized_snake_order_and_stays_setup_only() -> None:
    engine: Any = GameEngine(COLORS, seed=0, shuffle_players=True)
    sandbox = CatanSandbox(
        engine,
        {color: FirstLegalPlayer(color) for color in engine.state.colors},
        communication_policy=NoCommunicationPolicy(),
    )
    suite = load_context_suite()
    assembler = ContextAssembler(suite)
    realized_order = tuple(engine.state.colors)

    assert realized_order == (
        Color.ORANGE,
        Color.BLUE,
        Color.RED,
        Color.WHITE,
    )

    first_context = sandbox.decision_context()
    assert first_context.observation.turn_order == realized_order
    first_components = assembler.render_components(
        first_context,
        PlayerSession(first_context.actor, "setup-order:first"),
    )
    first_phase = next(
        component.value
        for component in first_components
        if component.id == "environment.phase_info"
    )
    assert "Round 1 (first settlement + road): ORANGE -> BLUE -> RED -> WHITE" in first_phase
    assert "Round 2 (second settlement + road): WHITE -> RED -> BLUE -> ORANGE" in first_phase
    assert "Your positions: round 1 = 1/4; round 2 = 4/4." in first_phase

    for _ in range(8):
        await sandbox.step()

    second_context = sandbox.decision_context()
    assert second_context.prompt_key == "initial_settlement_2"
    assert second_context.actor == Color.WHITE
    second_components = assembler.render_components(
        second_context,
        PlayerSession(second_context.actor, "setup-order:second"),
    )
    second_phase = next(
        component.value
        for component in second_components
        if component.id == "environment.phase_info"
    )
    assert "Round 1 (first settlement + road): ORANGE -> BLUE -> RED -> WHITE" in second_phase
    assert "Round 2 (second settlement + road): WHITE -> RED -> BLUE -> ORANGE" in second_phase
    assert "Your positions: round 1 = 4/4; round 2 = 1/4." in second_phase

    historical_suite = load_context_suite(
        default_suite_path().with_name("catan_v7.yaml")
    )
    historical_components = ContextAssembler(historical_suite).render_components(
        second_context,
        PlayerSession(second_context.actor, "setup-order:historical"),
    )
    historical_phase = next(
        component.value
        for component in historical_components
        if component.id == "environment.phase_info"
    )
    assert historical_suite.context.initial_placement_order == "omit"
    assert "Initial placement order" not in historical_phase

    for _ in range(8):
        await sandbox.step()

    main_context = sandbox.decision_context()
    main_components = assembler.render_components(
        main_context,
        PlayerSession(main_context.actor, "setup-order:main"),
    )
    main_phase = next(
        component.value
        for component in main_components
        if component.id == "environment.phase_info"
    )
    assert main_context.observation.current_phase == "main_game"
    assert "Initial placement order" not in main_phase


@allows_deprecated_suite
def test_legacy_shared_road_guidance_supports_current_routing() -> None:
    sandbox, player, _ = _sandbox_and_player([])
    context = sandbox.decision_context()
    suite = load_context_suite(default_suite_path().with_name("catan_v5.yaml"))
    assembler = ContextAssembler(suite)

    for prompt_key in ("initial_road_1", "initial_road_2"):
        components = assembler.render_components(
            replace(context, prompt_key=prompt_key),
            player.session,
        )
        component = next(
            item
            for item in components
            if item.id == "environment.phase_guidance"
        )
        assert component.value == suite.phase_guidance["initial_road"]
