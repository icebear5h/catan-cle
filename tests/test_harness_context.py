from dataclasses import dataclass, field, replace
import json

import pytest

from cle.harness import (
    ContextAssembler,
    ContextSuite,
    ModelResponse,
    PlayerResponseParseError,
    PlayerResponseParser,
    PlayerSession,
    default_suite_path,
    load_context_suite,
)
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer, HumanPlayer, ScriptedPlayer
from cle.players.contracts import PlayerChoice
from cle.players.validation import action_from_choice
from cle.harness.response_xml import parse_response_fields
from cle.sandbox import CatanSandbox
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions, trade_response_actions
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import player_freqdeck_add


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


@pytest.fixture
def trade_sandbox():
    sandbox, _, _ = _sandbox_and_player([])
    engine = sandbox.game_engine
    for _ in range(16):
        engine.step(engine.state.playable_actions[0])
    engine.step(Action(Color.RED, ActionType.ROLL, (1, 1)), force=True)
    for color, bundle in (
        (Color.RED, (2, 0, 0, 0, 0)),
        (Color.BLUE, (0, 0, 0, 0, 2)),
    ):
        player_freqdeck_add(engine.state, color, bundle)
        for index, count in enumerate(bundle):
            engine.state.resource_freqdeck[index] -= count
    engine.state.playable_actions = generate_playable_actions(engine.state)
    return sandbox


def test_default_yaml_suite_is_strict_and_separately_loadable():
    first = load_context_suite()
    second = load_context_suite()

    assert first == second
    assert first is not second
    assert first.id == "catan-agent"
    assert first.version == "10.0.0"
    assert first.context.mode == "components"
    assert first.context.social_context is True
    assert first.context.initial_placement_order == "both_rounds"
    assert first.context.order == (
        "trajectory",
        "strategic_memory",
        "visible_events",
        "recent_table_talk",
        "commitments",
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
    response_fields, _ = parse_response_fields(first.response.instruction)
    assert set(response_fields) == set(first.response.tags)
    assert all(len(values) == 1 for values in response_fields.values())

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
        "environment.recent_table_talk",
        "environment.commitments",
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
        "Resolve the required discard. Choose exactly the stated number of cards "
        "from your holdings using named resource counts. Reassess your plan after the robber "
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


@pytest.mark.parametrize(
    "text, index, fallback",
    [
        ("<action>0</action>", 0, False),
        ("<AcTiOn>\n 7 \n</AcTiOn>", 7, False),
        ("<action>7</action><action>007</action>", 7, False),
        ("<action>7</action>action_index: 7", 7, False),
        ("7", 7, True),
        (" \n7\n ", 7, True),
        ("ACTION: 7", 7, True),
        ("action_index = 7", 7, True),
        ("action_index:\n7", 7, True),
        ("move: 7", 7, True),
        ("MOVE_INDEX=7", 7, True),
        ("action: 7\nmove_index: 007", 7, True),
        ("<game_plan>Need 2 roads</game_plan>action_index: 7", 7, True),
        ("<game_plan>Need 2 roads</game_plan>7", 7, True),
        ("<action>zero-based index from VALID ACTIONS</action>7", 7, True),
        (
            "<game_plan>Consider <action>1</action> or action_index: 2</game_plan>"
            "<action>7</action>",
            7,
            False,
        ),
    ],
)
def test_action_parser_preserves_exact_indices_and_explicit_fallbacks(text, index, fallback):
    sandbox, _, _ = _sandbox_and_player([])
    context = sandbox.decision_context()

    choice = PlayerResponseParser(load_context_suite()).parse(
        context, ModelResponse(content=text)
    )

    assert choice.action_index == index
    assert (choice.parse_warning is not None) == fallback
    assert context.action_at(choice.action_index) == context.legal_actions[index]
    assert sandbox.game_engine.is_action_valid(context.action_at(choice.action_index))


@pytest.mark.parametrize("suite_name", ["catan_v10.yaml", "catan_v9.yaml", "catan_v4.yaml"])
@pytest.mark.parametrize("selection", ["<action>7</action>", "action_index: 7"])
def test_action_parser_ignores_the_suites_echoed_schema(suite_name, selection):
    sandbox, _, _ = _sandbox_and_player([])
    suite = load_context_suite(default_suite_path().with_name(suite_name))
    response = ModelResponse(
        content=f"{suite.response.instruction}\n<game_plan>Need 2 roads</game_plan>\n{selection}"
    )

    choice = PlayerResponseParser(suite).parse(sandbox.decision_context(), response)

    assert choice.action_index == 7
    assert choice.raw_response == response.content


@pytest.mark.parametrize("suite_name", ["catan_v4.yaml", "catan_v9.yaml"])
@pytest.mark.parametrize("selection", ["<action>7</action>", "action_index: 7", "7"])
@pytest.mark.parametrize(
    "rationale",
    [
        "<rationale>Best move: settle at node 7.</rationale>",
        "<RATIONALE><action>1</action>\nmove_index: 2</RATIONALE>",
    ],
)
def test_action_parser_treats_historical_rationale_as_inert_data(suite_name, selection, rationale):
    sandbox, _, _ = _sandbox_and_player([])
    context = sandbox.decision_context()
    suite = load_context_suite(default_suite_path().with_name(suite_name))
    text = f"<game_plan>Expand...</game_plan>{rationale}{selection}"

    choice = PlayerResponseParser(suite).parse(context, ModelResponse(content=text))

    assert choice.action_index == 7
    assert context.action_at(choice.action_index) == context.legal_actions[7]
    assert choice.game_plan == "Expand..."
    assert choice.rationale == ""
    assert choice.native_reasoning == ""
    assert choice.raw_response == text


@pytest.mark.parametrize(
    "instruction",
    [
        "<game_plan>plan</game_plan><action>0</action>",
        "Return a zero-based action index.",
    ],
)
def test_action_parser_keeps_real_selections_when_schema_has_no_placeholder(instruction):
    sandbox, _, _ = _sandbox_and_player([])
    suite_data = load_context_suite().model_dump(mode="python")
    suite_data["response"]["instruction"] = instruction
    parser = PlayerResponseParser(ContextSuite.model_validate(suite_data))
    context = sandbox.decision_context()

    assert parser.parse(context, _response(action=0)).action_index == 0
    with pytest.raises(PlayerResponseParseError, match="conflicting"):
        parser.parse(context, ModelResponse(content="<action>0</action><action>1</action>"))
    with pytest.raises(PlayerResponseParseError, match="whole non-negative integers"):
        parser.parse(context, ModelResponse(content="<action></action>action_index: 7"))


@pytest.mark.parametrize("template", ["<action>{}</action>", "action_index: {}", "{}"])
@pytest.mark.parametrize(
    "value",
    ["-1", "+1", "1.9", "0/1", "1e1", "1 2", "1,2", "0x1", "", "999"],
)
def test_action_parser_rejects_non_integer_or_out_of_range_indices(template, value):
    sandbox, _, _ = _sandbox_and_player([])

    with pytest.raises(PlayerResponseParseError):
        PlayerResponseParser(load_context_suite()).parse(
            sandbox.decision_context(), ModelResponse(content=template.format(value))
        )


@pytest.mark.parametrize(
    "text",
    [
        "<game_plan>Need 2 roads</game_plan>",
        "<game_plan>action_index: 7</game_plan>",
        "<game_plan><action>7</action></game_plan>",
        "<rationale><action>7</action></rationale>",
        "<rationale>move_index: 7</rationale>",
        '<trade_offer>{"give":{"WOOD":1},"receive":{"ORE":2}}</trade_offer>',
        '<trade_offer>{"note":"action_index: 7"}</trade_offer>',
        "I need 2 roads and will decide later.",
        "I need 2 roads.\n7",
        "<game_plan>Need 2 roads</game_plan><action>not sure</action>",
        "<action>-1</action><action>2</action>",
    ],
)
def test_action_parser_does_not_infer_an_index_from_plan_trade_or_prose(text):
    sandbox, _, _ = _sandbox_and_player([])

    with pytest.raises(PlayerResponseParseError):
        PlayerResponseParser(load_context_suite()).parse(
            sandbox.decision_context(), ModelResponse(content=text)
        )


@pytest.mark.parametrize(
    "text",
    [
        "<action>1</action><action>2</action>",
        "action: 1\nmove_index: 2",
        "<action>1</action>action_index: 2",
        "<action>1</action>2",
    ],
)
def test_action_parser_rejects_conflicting_selections(text):
    sandbox, _, _ = _sandbox_and_player([])

    with pytest.raises(PlayerResponseParseError, match="conflicting"):
        PlayerResponseParser(load_context_suite()).parse(
            sandbox.decision_context(), ModelResponse(content=text)
        )


def test_counteroffer_parser_preserves_engine_generated_parent_id(trade_sandbox):
    engine = trade_sandbox.game_engine
    parser = PlayerResponseParser(load_context_suite())
    context = trade_sandbox.decision_context()
    index = next(
        i
        for i, action in enumerate(context.legal_actions)
        if action.action_type == ActionType.OFFER_TRADE
    )
    root_choice = parser.parse(
        context,
        ModelResponse(
            content=(
                f"<action>{index}</action>"
                '<trade_offer>{"give":{"WOOD":1},"receive":{"ORE":1}}</trade_offer>'
            )
        ),
    )
    selected = context.action_at(root_choice.action_index)
    root = engine.step(
        Action(selected.color, selected.action_type, root_choice.trade_offer)
    ).resolved_action.value
    context = replace(
        context,
        context_id=f"{engine.id}:{engine.revision}:BLUE",
        actor=Color.BLUE,
        observation=engine.observe(Color.BLUE),
        events=engine.project_events(Color.BLUE),
        legal_actions=tuple(trade_response_actions(engine.state, Color.BLUE)),
    )
    index = next(
        i
        for i, action in enumerate(context.legal_actions)
        if action.action_type == ActionType.COUNTER_OFFER
    )

    choice = parser.parse(
        context,
        ModelResponse(
            content=(
                f"<action>{index}</action>"
                '<trade_offer>{"give":{"ORE":1},"receive":{"WOOD":2}}</trade_offer>'
            )
        ),
    )

    assert ":o1" in root.id
    assert choice.trade_offer.parent_offer_id == root.id
    assert choice.trade_offer.audience == frozenset({Color.RED})
    selected = context.action_at(choice.action_index)
    action = Action(selected.color, selected.action_type, choice.trade_offer)
    assert engine.is_action_valid(action)
    assert engine.step(action).resolved_action.value.parent_offer_id == root.id


@pytest.mark.parametrize(
    "payload",
    [
        '{"give":{"WOOD":2,"WOOD":1},"receive":{"ORE":1}}',
        '{"give":{"WOOD":1},"receive":{"ORE":2,"ORE":1}}',
        '{"give":{"WOOD":2},"give":{"WOOD":1},"receive":{"ORE":1}}',
        '{"give":{"WOOD":1},"receive":{"ORE":1},"receive":{"ORE":2}}',
        '{"give":{"WOOD":1},"receive":{"ORE":1},"give_any":0,"give_any":1}',
        '{"give":{"WOOD":1},"receive":{"ORE":1},"receive_any":0,"receive_any":1}',
        '{"give":{"WOOD":2,"wood":1},"receive":{"ORE":1}}',
    ],
)
def test_trade_parser_rejects_duplicate_json_keys(trade_sandbox, payload):
    context = trade_sandbox.decision_context()
    index = next(
        i
        for i, action in enumerate(context.legal_actions)
        if action.action_type == ActionType.OFFER_TRADE
    )

    with pytest.raises(PlayerResponseParseError, match="Duplicate"):
        PlayerResponseParser(load_context_suite()).parse(
            context,
            ModelResponse(content=f"<action>{index}</action><trade_offer>{payload}</trade_offer>"),
        )


def test_template_rendering_preserves_template_syntax_in_plan_data():
    sandbox, player, _ = _sandbox_and_player([])
    suite = load_context_suite()
    context = sandbox.decision_context()
    plan = "Save {{ wood }} and {{ value }} for a road"
    choice = PlayerResponseParser(suite).parse(context, _response(plan=plan))
    player.session.strategic_memory = choice.game_plan

    components = ContextAssembler(suite).render_components(context, player.session)

    memory = next(item for item in components if item.id == "environment.strategic_memory")
    assert memory.rendered == f"YOUR CURRENT GAME PLAN:\n{plan}"
    with pytest.raises(ValueError, match="Template variables have no values"):
        ContextAssembler._render_template("{{ missing }}", {"value": plan})


def test_year_of_plenty_menu_names_singleton_resources(trade_sandbox):
    engine = trade_sandbox.game_engine
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
    context = trade_sandbox.decision_context()
    suite = load_context_suite()
    components = ContextAssembler(suite).render_components(
        context, PlayerSession(context.actor, "year-of-plenty")
    )
    menu = next(item.value for item in components if item.id == "environment.legal_actions")
    expected = {
        ("WOOD",): "Year of Plenty: take WOOD",
        ("BRICK",): "Year of Plenty: take BRICK",
        ("WOOD", "BRICK"): "Year of Plenty: take WOOD and BRICK",
    }
    seen = set()

    for index, action in enumerate(context.legal_actions):
        if action.action_type != ActionType.PLAY_YEAR_OF_PLENTY:
            continue
        assert f"{index}. {expected[action.value]}" in menu
        choice = PlayerResponseParser(suite).parse(context, _response(action=index))
        assert context.action_at(choice.action_index) == action
        assert engine.is_action_valid(action)
        seen.add(action.value)
    assert seen == set(expected)


def test_v10_social_sections_use_only_supplied_perspective_and_leave_game_history_complete():
    sandbox, player, _ = _sandbox_and_player([])
    engine = sandbox.game_engine
    engine.step(engine.state.playable_actions[0])
    for index in range(15):
        engine.append_message(
            speaker=Color.BLUE,
            text=f"visible-offer-{index}",
            audience=(Color.RED,),
            intent="TRADE",
            causation_id=f"offer:{index}",
            commitment=("RED avoids BLUE", "BLUE offers ORE", 5) if index == 0 else None,
        )
    engine.append_message(
        speaker=Color.WHITE,
        text="private-white-orange",
        audience=(Color.ORANGE,),
        intent="TRADE",
        causation_id="private",
        commitment=("private condition", "private promise", 5),
    )
    context = replace(
        sandbox.decision_context(),
        recent_messages=engine.project_messages(Color.RED),
        active_commitments=engine.active_commitments(Color.RED),
    )
    suite = load_context_suite()
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


@pytest.mark.parametrize("suite_name", ["catan_v9.yaml", "catan_v10.yaml"])
def test_component_order_requires_explicit_social_policy(suite_name):
    suite = load_context_suite(default_suite_path().with_name(suite_name))
    data = suite.model_dump(mode="python")
    data["context"]["social_context"] = not suite.context.social_context
    with pytest.raises(ValueError, match="fixed component order"):
        ContextSuite.model_validate(data)


@pytest.fixture
def discard_sandbox():
    sandbox, _, _ = _sandbox_and_player([])
    engine = sandbox.game_engine
    for _ in range(16):
        engine.step(engine.state.playable_actions[0])
    player_freqdeck_add(engine.state, Color.RED, (4, 0, 0, 0, 4))
    engine.state.resource_freqdeck[0] -= 4
    engine.state.resource_freqdeck[4] -= 4
    engine.step(Action(Color.RED, ActionType.ROLL, (3, 4)), force=True)
    context = sandbox.decision_context()
    assert context.legal_actions[0].action_type == ActionType.DISCARD
    return sandbox


@pytest.fixture
def discard_context(discard_sandbox):
    context = discard_sandbox.decision_context()
    # Supply the engine's exact required count independently of context assembly.
    return replace(context, discard_count=sum(context.observation.my_resources.values()) // 2)


def test_v10_discard_parser_and_menu_use_exact_named_bundle(discard_context):
    context = discard_context
    count = context.discard_count
    payload = f'{{"ore":{count - 2},"wood":2}}'
    text = f"<game_plan>Keep building cards</game_plan><action>0</action><discard>{payload}</discard>"
    suite = load_context_suite()
    choice = PlayerResponseParser(suite).parse(context, ModelResponse(content=text))

    assert choice.discard_cards == ("WOOD", "WOOD") + ("ORE",) * (count - 2)
    assert choice.raw_response == text
    assert action_from_choice(context, choice) == Action(
        context.actor, ActionType.DISCARD, choice.discard_cards
    )
    components = ContextAssembler(suite).render_components(
        context, PlayerSession(context.actor, "discard:RED")
    )
    menu = next(item.value for item in components if item.id == "environment.legal_actions")
    assert f"Discard exactly {count} resource cards" in menu
    assert "<discard>" in menu


def test_parsed_discard_bundle_is_the_exact_engine_action(discard_sandbox, discard_context):
    context = discard_context
    engine = discard_sandbox.game_engine
    before = dict(engine.observe(context.actor).my_resources)
    choice = PlayerResponseParser(load_context_suite()).parse(
        context,
        ModelResponse(content=(
            '<action>0</action><discard>{"WOOD":2,"ORE":'
            f'{context.discard_count - 2}'
            '}</discard>'
        )),
    )
    transition = engine.step(action_from_choice(context, choice))
    after = engine.observe(context.actor).my_resources
    assert transition.resolved_action.value == choice.discard_cards
    assert after == {
        resource: count - choice.discard_cards.count(resource)
        for resource, count in before.items()
    }


@pytest.mark.parametrize("payload", [
    "", "null", "[]", '["WOOD"]', '"WOOD"', "{}", '{"WOOD":true}',
    '{"WOOD":-1}', '{"WOOD":0}', '{"WOOD":1.5}', '{"WOOD":1e1}',
    '{"WOOD":"4"}', '{"WOOD":NaN}', '{"WOOD":{}}', '{"WOOD":[]}',
    '{"ANY":4}', '{"wood":2,"WOOD":2}', '{"WOOD":2,"WOOD":2}',
    '{"WOOD":999}', '{"WOOD":1}', '{"ORE":100000000000000000000}',
])
def test_discard_parser_rejects_invalid_json_resources_counts_and_holdings(discard_context, payload):
    with pytest.raises(PlayerResponseParseError):
        PlayerResponseParser(load_context_suite()).parse(
            discard_context, ModelResponse(content=f"<action>0</action><discard>{payload}</discard>")
        )


def test_discard_requirement_is_suite_opt_in_and_old_menu_is_unchanged(discard_context):
    context = discard_context
    parser = PlayerResponseParser(load_context_suite())
    with pytest.raises(PlayerResponseParseError, match="requires <discard>"):
        parser.parse(context, _response())
    historical = load_context_suite(default_suite_path().with_name("catan_v9.yaml"))
    legacy_choice = PlayerResponseParser(historical).parse(context, _response())
    assert legacy_choice.discard_cards is None
    assert action_from_choice(context, legacy_choice) == context.legal_actions[0]
    components = ContextAssembler(historical).render_components(
        context, PlayerSession(context.actor, "historical:RED")
    )
    assert next(item.value for item in components if item.id == "environment.legal_actions") == (
        "0. Discard resources"
    )
    data = load_context_suite().model_dump(mode="python")
    data["response"]["tags"] = tuple(tag for tag in data["response"]["tags"] if tag != "discard")
    assert PlayerResponseParser(ContextSuite.model_validate(data)).parse(
        context, _response()
    ).discard_cards is None


def test_legacy_suite_keeps_automatic_discard_execution(discard_sandbox, discard_context):
    suite = load_context_suite(default_suite_path().with_name("catan_v9.yaml"))
    choice = PlayerResponseParser(suite).parse(discard_context, _response())
    transition = discard_sandbox.game_engine.step(action_from_choice(discard_context, choice))
    assert choice.discard_cards is None
    assert len(transition.resolved_action.value) == discard_context.discard_count


@pytest.mark.parametrize("parameter", [
    '<discard>{"WOOD":4}</discard>',
    '<trade_offer>{"give":{"WOOD":1},"receive":{"ORE":1}}</trade_offer>',
])
def test_parser_rejects_parameters_on_wrong_action(parameter):
    sandbox, _, _ = _sandbox_and_player([])
    with pytest.raises(PlayerResponseParseError, match="only valid"):
        PlayerResponseParser(load_context_suite()).parse(
            sandbox.decision_context(), ModelResponse(content=f"<action>0</action>{parameter}")
        )


@pytest.mark.parametrize("text", [
    "<!-- <action>0</action> -->",
    "<!-- action_index: 0 -->",
    "<action index='0'>0</action>",
    "<action>0</action><action/>",
    "<action>0</action><action>1",
    "<action><action>0</action></action>",
    "<game_plan>one</game_plan><game_plan>two</game_plan><action>0</action>",
    '<action>0</action><discard>{"WOOD":4}</discard><discard>{"WOOD":4}</discard>',
])
def test_decision_parser_rejects_malformed_repeated_and_comment_only_controls(text):
    sandbox, _, _ = _sandbox_and_player([])
    with pytest.raises(PlayerResponseParseError):
        PlayerResponseParser(load_context_suite()).parse(
            sandbox.decision_context(), ModelResponse(content=text)
        )


def test_decision_parser_keeps_xml_escaped_prose_and_raw_output_separate():
    sandbox, _, _ = _sandbox_and_player([])
    text = (
        "<!-- <action>2</action> action_index: 3 -->"
        "<game_plan>ORE &gt; WOOD &amp; wheat &lt; sheep</game_plan><action>0</action>"
    )
    choice = PlayerResponseParser(load_context_suite()).parse(
        sandbox.decision_context(), ModelResponse(content=text)
    )
    assert choice.action_index == 0
    assert choice.game_plan == "ORE > WOOD & wheat < sheep"
    assert choice.raw_response == text


@pytest.mark.asyncio
async def test_first_legal_discards_deterministically_and_scripted_preserves_typed_choice(discard_context):
    context = discard_context
    player = FirstLegalPlayer(context.actor)
    first = await player.choose(context)
    second = await player.choose(context)
    assert first == second
    assert len(first.choice.discard_cards) == context.discard_count
    assert first.choice.discard_cards[:4] == ("WOOD",) * 4
    assert action_from_choice(context, first.choice).value == first.choice.discard_cards
    exact_cards = ("ORE",) * (context.discard_count - 2) + ("WOOD",) * 2
    concrete = replace(context, legal_actions=(Action(context.actor, ActionType.DISCARD, exact_cards),))
    assert (await player.choose(concrete)).choice.discard_cards == exact_cards

    scripted = ScriptedPlayer(context.actor, [first.choice, 0])
    saved = scripted.snapshot()
    assert (await scripted.choose(context)).choice is first.choice
    assert (await scripted.choose(context)).choice == PlayerChoice(0)
    scripted.restore(saved)
    assert (await scripted.choose(context)).choice == first.choice
    scripted.restore((3, 4, (0,)))
    assert scripted.event_cursor == 3
    assert scripted.accepted_choices == 4
    assert (await scripted.choose(context)).choice == PlayerChoice(0)


@pytest.mark.asyncio
async def test_human_discard_asks_for_exact_cards_and_retries_invalid_bundles(discard_context):
    context = discard_context
    valid_payload = f'{{"WOOD":2,"ORE":{context.discard_count - 2}}}'
    answers = iter([
        "0", "not JSON", "[]", '{"WOOD":true}', '{"WOOD":100}',
        '{"WOOD":2,"WOOD":2}', '{"WOOD":1}', valid_payload,
    ])
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return next(answers)

    choice = (await HumanPlayer(context.actor, input_fn=answer).choose(context)).choice
    assert choice.discard_cards == ("WOOD",) * 2 + ("ORE",) * (context.discard_count - 2)
    assert len(prompts) == 8
    assert all(f"exactly {context.discard_count}" in prompt for prompt in prompts[1:])
    assert action_from_choice(context, choice).value == choice.discard_cards
