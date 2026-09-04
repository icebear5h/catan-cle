from dataclasses import dataclass, field, replace
import json

import pytest

from cle.harness import (
    ContextAssembler,
    ContextSuite,
    ModelResponse,
    PlayerSession,
    default_suite_path,
    load_context_suite,
)
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox import CatanSandbox
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _response(action=0, plan="expand toward wheat"):
    return ModelResponse(
        content=(
            f"<game_plan>{plan}</game_plan>"
            f"<action>{action}</action>"
        ),
        model="test/model",
        native_reasoning="native analysis",
        native_reasoning_details=({"type": "reasoning.text"},),
        reasoning_request=(("effort", "xhigh"), ("exclude", False)),
        provider_response_id="gen-context-test",
        provider_request_id="req-context-test",
        provider_native_finish_reason="stop",
    )


@dataclass
class FixedTransport:
    responses: list[ModelResponse]
    requests: list = field(default_factory=list)

    async def complete(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


class NoCommunicationPolicy:
    def pre_action(self, engine):
        return ()

    def after_events(self, engine, events, *, round_number):
        return ()


def _sandbox_and_player(responses):
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    transport = FixedTransport(list(responses))
    red = AgentPlayer(Color.RED, transport, session_id=f"{engine.id}:RED")
    players = {Color.RED: red}
    players.update({color: FirstLegalPlayer(color) for color in COLORS[1:]})
    return (
        CatanSandbox(engine, players, communication_policy=NoCommunicationPolicy()),
        red,
        transport,
    )


def test_default_yaml_suite_is_strict_and_separately_loadable():
    first = load_context_suite()
    second = load_context_suite()

    assert first == second
    assert first is not second
    assert first.id == "catan-agent"
    assert first.version == "9.0.0"
    assert first.context.mode == "components"
    assert first.context.initial_placement_order == "both_rounds"
    assert first.context.order == (
        "trajectory",
        "strategic_memory",
        "visible_events",
        "phase_info",
        "board_state",
        "resources",
        "opponents",
        "trade_window",
        "phase_guidance",
        "legal_actions",
        "decision_request",
        "response_schema",
    )
    assert first.context.trajectory.max_messages is None

    guidance_word_counts = {
        key: len(value.split())
        for key, value in first.phase_guidance.items()
    }
    assert set(guidance_word_counts) == {
        "initial_settlement_1",
        "initial_settlement_2",
        "initial_road_1",
        "initial_road_2",
        "initial_placement",
        "discarding",
        "robber",
        "main_game",
    }
    assert guidance_word_counts["initial_settlement_1"] == 256
    assert all(
        10 <= count <= 110
        for key, count in guidance_word_counts.items()
        if key != "initial_settlement_1"
    )
    assert 600 <= sum(guidance_word_counts.values()) <= 700

    strategy_variant = json.loads(
        default_suite_path()
        .parents[3]
        .joinpath(
            "evals",
            "suites",
            "catan_initial_settlement_strategy_v2.json",
        )
        .read_text(encoding="utf-8")
    )
    assert first.phase_guidance["initial_settlement_1"] == (
        strategy_variant["guidance"]
    )

    historical_v8 = load_context_suite(
        default_suite_path().with_name("catan_v8.yaml")
    )
    assert historical_v8.version == "8.0.0"
    assert historical_v8.context.initial_placement_order == "both_rounds"
    assert historical_v8.phase_guidance["initial_settlement_1"] != (
        strategy_variant["guidance"]
    )
    assert "Maximizing raw pip count is not the objective" not in (
        historical_v8.phase_guidance["initial_settlement_1"]
    )

    historical_v7 = load_context_suite(
        default_suite_path().with_name("catan_v7.yaml")
    )
    v8_without_order = historical_v8.model_dump(mode="python")
    v8_without_order["version"] = "7.0.0"
    v8_without_order["context"]["initial_placement_order"] = "omit"
    assert ContextSuite.model_validate(v8_without_order) == historical_v7

    guidance_text = "\n".join(first.phase_guidance.values()).lower()
    for implementation_phrase in (
        "game engine",
        "implementation",
        "legal menu",
        "live menu",
        "prompt",
        "rationale",
        "random stream",
        "runtime",
        "seeded",
    ):
        assert implementation_phrase not in guidance_text

    stable_contract = "\n".join(
        (first.system.template, first.response.instruction)
    ).lower()
    assert "rationale" not in stable_contract
    for prescriptive_phrase in (
        "expert",
        "prioritize",
        "prefer",
        "think in sequences",
        "cities > settlements",
        "block the leading opponent",
    ):
        assert prescriptive_phrase not in stable_contract


def test_legacy_decision_suite_remains_loadable():
    suite = load_context_suite(default_suite_path().with_name("catan_v5.yaml"))

    assert suite.version == "5.0.0"
    assert suite.context.mode == "legacy"
    assert suite.sections["observation"].template is None


@pytest.mark.asyncio
async def test_agent_player_keeps_full_conversation_and_full_visible_events():
    sandbox, player, transport = _sandbox_and_player(
        [_response(), _response(plan="connect the opening road"), _response()]
    )

    first = await sandbox.step()
    second = await sandbox.step()

    assert first.transitions[0].resolved_action == first.contexts[0].legal_actions[0]
    assert second.transitions[0].resolved_action == second.contexts[0].legal_actions[0]
    assert first.attempts[0].choice.native_reasoning == "native analysis"
    assert first.attempts[0].choice.native_reasoning_details == (
        {"type": "reasoning.text"},
    )
    assert dict(first.attempts[0].choice.reasoning_request) == {
        "effort": "xhigh",
        "exclude": False,
    }
    assert first.attempts[0].choice.provider_response_id == "gen-context-test"
    assert first.attempts[0].choice.provider_request_id == "req-context-test"
    assert first.attempts[0].choice.provider_native_finish_reason == "stop"
    first_system = transport.requests[0].messages[0].content
    first_user = transport.requests[0].messages[-1].content
    assert first_system == "You are playing a game of Catan. You are playing as RED."
    assert "expert" not in first_system.lower()
    assert "prioritize" not in first_system.lower()
    assert len(first_system.split()) < 150
    first_request = transport.requests[0]
    first_board = first_request.board_presentation
    assert first_board.kind == "text"
    assert first_board.format == "indexed_tile_rows/v3"
    assert first_board.provenance.source_id == first_request.decision_id
    assert first_board.provenance.perspective == Color.RED
    assert first_board.provenance.identity_space == "canonical_engine_ids"
    assert "CATAN FULL PUBLIC GRAPH V1" in first_board.content
    assert first_board.byte_length > 10_000
    first_components = first_request.components
    assert [component.id for component in first_components] == [
        "system.identity",
        "environment.strategic_memory",
        "environment.visible_events",
        "environment.phase_info",
        "environment.board_state",
        "environment.resources",
        "environment.opponents",
        "environment.trade_window",
        "environment.phase_guidance",
        "environment.legal_actions",
        "environment.decision_request",
        "environment.response_schema",
    ]
    assert first_user == "\n\n".join(
        component.rendered for component in first_components[1:]
    )
    assert "DECISION FACTS:" in first_user
    assert "Round 1 (first settlement + road): RED -> BLUE -> WHITE -> ORANGE" in first_user
    assert "Round 2 (second settlement + road): ORANGE -> WHITE -> BLUE -> RED" in first_user
    assert "Your positions: round 1 = 1/4; round 2 = 4/4." in first_user
    assert "first settlement gives no starting cards" in first_user
    assert "second settlement is independent of the first road" in first_user
    assert "nominal diversity without buildable combinations can be weak" in first_user
    assert next(
        component.value
        for component in first_components
        if component.id == "environment.board_state"
    ).startswith("YOUR BUILDINGS:")
    assert next(
        component.value
        for component in first_components
        if component.id == "environment.resources"
    ).startswith("YOUR RESOURCES:")
    assert first_user.count("VALID ACTIONS:") == 1
    assert "<valid_actions>" not in first_user
    assert "dice=" in first_user
    assert "pip" in first_user.lower()
    assert player.session.strategic_memory == "connect the opening road"
    assert all(
        "CATAN FULL PUBLIC GRAPH" not in message.content
        for message in player.session.messages
    )
    assert [message.role for message in player.session.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message.role for message in transport.requests[1].messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    second_user = transport.requests[1].messages[-1].content
    assert "0. RED: BUILD_SETTLEMENT" in second_user
    assert "more than one viable expansion location" in second_user
    assert "remains an independent choice" in second_user
    assert "does not need to connect to this road" in second_user

    while sandbox.current_actor() != Color.RED:
        await sandbox.step()
    await sandbox.step()

    third_user = transport.requests[2].messages[-1].content
    assert "0. RED: BUILD_SETTLEMENT" in third_user
    assert "1. RED: BUILD_ROAD" in third_user


def test_every_phase_key_renders_concise_player_facing_guidance():
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
    assert "nominal diversity without buildable combinations can be weak" in (
        rendered_guidance["initial_settlement_1"]
    )
    assert "at least two reachable expansion targets" in (
        rendered_guidance["initial_settlement_1"]
    )
    assert "second settlement is independent of the first road" in (
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
    assert rendered_guidance["discarding"] == (
        "DECISION FACTS:\n"
        "Resolve the required discard. Reassess your plan after the robber "
        "sequence changes production or resources."
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


@pytest.mark.asyncio
async def test_setup_prompt_routing_distinguishes_both_settlements_and_roads():
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


@pytest.mark.asyncio
async def test_setup_order_uses_realized_snake_order_and_stays_setup_only():
    engine = GameEngine(COLORS, seed=0, shuffle_players=True)
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


def test_legacy_shared_road_guidance_supports_current_routing():
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


def test_component_suite_rejects_unknown_missing_duplicate_and_oversized_strings():
    suite = load_context_suite()

    unknown = suite.model_dump(mode="python")
    unknown["sections"]["board_state"]["template"] = "{{ hidden_hand }}"
    with pytest.raises(ValueError, match="unknown variables"):
        ContextSuite.model_validate(unknown)

    missing = suite.model_dump(mode="python")
    del missing["sections"]["resources"]
    with pytest.raises(ValueError, match="unknown sections"):
        ContextSuite.model_validate(missing)

    duplicate = suite.model_dump(mode="python")
    duplicate["context"]["order"] = (
        *duplicate["context"]["order"],
        "resources",
    )
    with pytest.raises(ValueError, match="duplicate sections"):
        ContextSuite.model_validate(duplicate)

    oversized = suite.model_dump(mode="python")
    oversized["sections"]["resources"]["template"] = "x" * 12_001
    with pytest.raises(ValueError, match="at most 12000 characters"):
        ContextSuite.model_validate(oversized)


@pytest.mark.asyncio
async def test_sandbox_snapshot_restores_private_player_conversation():
    sandbox, player, _ = _sandbox_and_player([_response(), _response()])
    await sandbox.step()
    snapshot = sandbox.snapshot()

    await sandbox.step()
    assert len(player.session.messages) == 4

    sandbox.restore(snapshot)

    assert len(player.session.messages) == 2
    assert player.session.strategic_memory == "expand toward wheat"
    assert len(player.session.receipts) == 1


@pytest.mark.asyncio
async def test_sandbox_retries_invalid_player_output_with_feedback():
    sandbox, _, transport = _sandbox_and_player(
        [ModelResponse(content="<action>999</action>"), _response()]
    )

    result = await sandbox.step()

    assert result.after_revision == 1
    assert len(transport.requests) == 2
    assert "CORRECTION FROM THE SANDBOX" in transport.requests[1].messages[-1].content
    assert len(sandbox.decision_trace) == 1
