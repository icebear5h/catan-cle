"""Phase guidance, template rendering, and social sections."""
import json
from dataclasses import replace
from typing import Any

import pytest

from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import player_freqdeck_add
from cle.harness import (
    ContextAssembler,
    ModelResponse,
    PlayerResponseParser,
    PlayerSession,
    default_suite_path,
    load_context_suite,
)
from cle.sandbox import CatanSandbox

from .support import _response, _sandbox_and_player, allows_deprecated_suite


def test_every_phase_key_renders_concise_player_facing_guidance() -> None:
    sandbox, player, _ = _sandbox_and_player([])
    context = sandbox.decision_context()
    suite = load_context_suite()
    assembler = ContextAssembler(suite)

    rendered_guidance = {}
    for prompt_key, guidance in suite.phase_guidance.items():
        components = assembler.render_components(
            replace(context, prompt_key=prompt_key),
            player.session,
        )
        component = next(
            item
            for item in components
            if item.id == "environment.phase_guidance"
        )
        assert component.value == guidance
        assert component.rendered == f"DECISION FACTS:\n{guidance}"
        rendered_guidance[prompt_key] = component.rendered

    assert "Maximizing raw pip count is not the objective" in (
        rendered_guidance["initial_settlement_1"]
    )
    assert "one coupled portfolio" in (
        rendered_guidance["initial_settlement_1"]
    )
    assert "opening archetype" in (
        rendered_guidance["initial_settlement_1"]
    )
    assert "at least two reachable expansion targets" in (
        rendered_guidance["initial_settlement_1"]
    )
    assert "placed independently of this road" in (
        rendered_guidance["initial_settlement_1"]
    )
    assert "one starting card from each adjacent non-desert tile" in (
        rendered_guidance["initial_settlement_2"]
    )
    assert "second settlement remains an independent choice" in (
        rendered_guidance["initial_road_1"]
    )
    assert "eventually join your two starting networks" in (
        rendered_guidance["initial_road_2"]
    )
    assert rendered_guidance["discarding"].startswith(
        "DECISION FACTS:\n"
        "Resolve the required discard. Choose exactly the stated number of cards "
        "from your holdings using named resource counts."
    )
    assert "movement and victim selection may be separate choices" in (
        rendered_guidance["robber"]
    )
    assert "the stolen card is random" in rendered_guidance["robber"]
    assert "next action as part of a sequence" in (
        rendered_guidance["main_game"]
    )
    assert "Trade acceptance is non-binding until the turn player confirms" in (
        rendered_guidance["main_game"]
    )


def test_template_rendering_preserves_template_syntax_in_plan_data() -> None:
    sandbox, player, _ = _sandbox_and_player([])
    suite = load_context_suite()
    context = sandbox.decision_context()
    plan = "Save {{ wood }} and {{ value }} for a road"
    choice: Any = PlayerResponseParser(suite).parse(context, ModelResponse(content=json.dumps({
        "game_plan": plan,
        "tool": "build_settlement",
        "arguments": {"node": f"<N{context.legal_actions[0].value:02d}>"},
    })))
    player.session.strategic_memory = choice.game_plan

    components = ContextAssembler(suite).render_components(context, player.session)

    memory = next(item for item in components if item.id == "environment.strategic_memory")
    assert memory.rendered == f"YOUR CURRENT GAME PLAN:\n{plan}"
    with pytest.raises(ValueError, match="Template variables have no values"):
        ContextAssembler._render_template("{{ missing }}", {"value": plan})


@allows_deprecated_suite
def test_v10_year_of_plenty_menu_names_singleton_resources(trade_sandbox: CatanSandbox) -> None:
    engine: Any = trade_sandbox.game_engine
    engine.state.development_listdeck.remove("YEAR_OF_PLENTY")
    engine.state.player_state["P0_YEAR_OF_PLENTY_IN_HAND"] = 1
    engine.state.player_state["P0_YEAR_OF_PLENTY_OWNED_AT_START"] = True
    bank = [1, 1, 0, 0, 0]
    player_freqdeck_add(
        engine.state,
        Color.BLUE,
        [held - remaining for held, remaining in zip(engine.state.resource_freqdeck, bank)],
    )
    engine.state.resource_freqdeck[:] = bank
    engine.state.playable_actions = generate_playable_actions(engine.state)
    context: Any = trade_sandbox.decision_context()
    suite: Any = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))
    components = ContextAssembler(suite).render_components(
        context, PlayerSession(context.actor, "year-of-plenty")
    )
    menu: Any = next(item.value for item in components if item.id == "environment.legal_actions")
    expected: Any = {
        ("WOOD",): "Year of Plenty: take WOOD",
        ("BRICK",): "Year of Plenty: take BRICK",
        ("WOOD", "BRICK"): "Year of Plenty: take WOOD and BRICK",
    }
    seen: Any = set()

    for index, action in enumerate(context.legal_actions):
        if action.action_type != ActionType.PLAY_YEAR_OF_PLENTY:
            continue
        assert f"{index}. {expected[action.value]}" in menu
        choice: Any = PlayerResponseParser(suite).parse(context, _response(action=index))
        assert context.action_at(choice.action_index) == action
        assert engine.is_action_valid(action)
        seen.add(action.value)
    assert seen == set(expected)


@allows_deprecated_suite
def test_v10_social_sections_use_only_supplied_perspective_and_leave_game_history_complete() -> None:
    sandbox, player, _ = _sandbox_and_player([])
    engine = sandbox.game_engine
    engine.step(engine.state.playable_actions[0])
    for index in range(15):
        engine.append_message(
            speaker=Color.BLUE,
            text=f"visible-offer-{index}",
            audience=(Color.RED,),
            causation_id=f"offer:{index}",
            commitment=("RED avoids BLUE", "BLUE offers ORE", 5) if index == 0 else None,
        )
    engine.append_message(
        speaker=Color.WHITE,
        text="private-white-orange",
        audience=(Color.ORANGE,),
        causation_id="private",
        commitment=("private condition", "private promise", 5),
    )
    context = replace(
        sandbox.decision_context(),
        recent_messages=engine.project_messages(Color.RED),
        active_commitments=engine.active_commitments(Color.RED),
    )
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))
    request = ContextAssembler(suite).assemble(context, player.session)
    components = {item.id: item for item in request.components}
    talk = components["environment.recent_table_talk"].value
    assert "visible-offer-3" in talk
    assert "visible-offer-14" in talk
    assert "visible-offer-0" not in talk
    assert "BLUE to RED: RED avoids BLUE -> BLUE offers ORE (expires turn 5)" in (
        components["environment.commitments"].value
    )
    assert "private-white-orange" not in request.messages[-1].content
    assert "private promise" not in request.messages[-1].content
    assert components["environment.visible_events"].value == ContextAssembler._format_events(
        context.events
    )
    assert "0. RED: BUILD_SETTLEMENT" in components["environment.visible_events"].value

    empty_context = replace(context, recent_messages=(), active_commitments=())
    empty = ContextAssembler(suite).assemble(empty_context, player.session)
    assert "visible-offer-14" not in empty.messages[-1].content
    assert "BLUE offers ORE" not in empty.messages[-1].content

    historical = load_context_suite(default_suite_path().with_name("catan_v9.yaml"))
    assert historical.context.social_context is False
    legacy_components = ContextAssembler(historical).render_components(context, player.session)
    assert all(item.id not in {
        "environment.recent_table_talk", "environment.commitments"
    } for item in legacy_components)
    assert "visible-offer-14" not in "\n".join(item.rendered for item in legacy_components)
