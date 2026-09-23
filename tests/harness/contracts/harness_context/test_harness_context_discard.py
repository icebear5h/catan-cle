"""Discard menus, parsers, and player behaviour."""
from dataclasses import replace
from typing import Any

import pytest

from cle.game_engine.models.enums import Action, ActionType
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
from cle.players.baseline import FirstLegalPlayer, HumanPlayer, ScriptedPlayer
from cle.players.contracts import PlayerChoice, PlayerContext
from cle.players.validation import action_from_choice
from cle.sandbox import CatanSandbox

from .support import _response, _sandbox_and_player


def test_v10_discard_parser_and_menu_use_exact_named_bundle(discard_context: PlayerContext) -> None:
    context = discard_context
    count: Any = context.discard_count
    payload = f'{{"ore":{count - 2},"wood":2}}'
    text: Any = f"<game_plan>Keep building cards</game_plan><action>0</action><discard>{payload}</discard>"
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))
    choice: Any = PlayerResponseParser(suite).parse(context, ModelResponse(content=text))

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


def test_v10_parsed_discard_bundle_is_the_exact_engine_action(discard_sandbox: CatanSandbox, discard_context: PlayerContext) -> None:
    context: Any = discard_context
    engine: Any = discard_sandbox.game_engine
    before: Any = dict(engine.observe(context.actor).my_resources)
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))
    choice: Any = PlayerResponseParser(suite).parse(
        context,
        ModelResponse(content=(
            '<action>0</action><discard>{"WOOD":2,"ORE":'
            f'{context.discard_count - 2}'
            '}</discard>'
        )),
    )
    transition: Any = engine.step(action_from_choice(context, choice))
    after: Any = engine.observe(context.actor).my_resources
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
def test_v10_discard_parser_rejects_invalid_json_resources_counts_and_holdings(discard_context: PlayerContext, payload: str) -> None:
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))
    with pytest.raises(PlayerResponseParseError):
        PlayerResponseParser(suite).parse(
            discard_context, ModelResponse(content=f"<action>0</action><discard>{payload}</discard>")
        )


def test_discard_requirement_is_suite_opt_in_and_old_menu_is_unchanged(discard_context: PlayerContext) -> None:
    context: Any = discard_context
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))
    parser = PlayerResponseParser(suite)
    with pytest.raises(PlayerResponseParseError, match="requires <discard>"):
        parser.parse(context, _response())
    historical = load_context_suite(default_suite_path().with_name("catan_v9.yaml"))
    legacy_choice: Any = PlayerResponseParser(historical).parse(context, _response())
    assert legacy_choice.discard_cards is None
    assert action_from_choice(context, legacy_choice) == context.legal_actions[0]
    components = ContextAssembler(historical).render_components(
        context, PlayerSession(context.actor, "historical:RED")
    )
    assert next(item.value for item in components if item.id == "environment.legal_actions") == (
        "0. Discard resources"
    )
    data: Any = suite.model_dump(mode="python")
    data["response"]["tags"] = tuple(tag for tag in data["response"]["tags"] if tag != "discard")
    assert PlayerResponseParser(ContextSuite.model_validate(data)).parse(
        context, _response()
    ).discard_cards is None


def test_legacy_suite_keeps_automatic_discard_execution(discard_sandbox: CatanSandbox, discard_context: PlayerContext) -> None:
    suite = load_context_suite(default_suite_path().with_name("catan_v9.yaml"))
    choice: Any = PlayerResponseParser(suite).parse(discard_context, _response())
    transition = discard_sandbox.game_engine.step(action_from_choice(discard_context, choice))
    assert choice.discard_cards is None
    assert len(transition.resolved_action.value) == discard_context.discard_count


@pytest.mark.parametrize("parameter", [
    '<discard>{"WOOD":4}</discard>',
    '<trade_offer>{"give":{"WOOD":1},"receive":{"ORE":1}}</trade_offer>',
])
def test_v10_parser_rejects_parameters_on_wrong_action(parameter: str) -> None:
    sandbox, _, _ = _sandbox_and_player([])
    suite = load_context_suite(default_suite_path().with_name("catan_v10.yaml"))
    with pytest.raises(PlayerResponseParseError, match="only valid"):
        PlayerResponseParser(suite).parse(
            sandbox.decision_context(), ModelResponse(content=f"<action>0</action>{parameter}")
        )


@pytest.mark.asyncio
async def test_first_legal_discards_deterministically_and_scripted_preserves_typed_choice(discard_context: PlayerContext) -> None:
    context = discard_context
    player: Any = FirstLegalPlayer(context.actor)
    first: Any = await player.choose(context)
    second = await player.choose(context)
    assert first == second
    assert len(first.choice.discard_cards) == context.discard_count
    assert first.choice.discard_cards[:4] == ("WOOD",) * 4
    assert action_from_choice(context, first.choice).value == first.choice.discard_cards
    exact_cards: Any = ("ORE",) * (context.discard_count - 2) + ("WOOD",) * 2
    concrete: Any = replace(context, legal_actions=(Action(context.actor, ActionType.DISCARD, exact_cards),))
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
async def test_human_discard_asks_for_exact_cards_and_retries_invalid_bundles(discard_context: PlayerContext) -> None:
    context: Any = discard_context
    valid_payload = f'{{"WOOD":2,"ORE":{context.discard_count - 2}}}'
    answers = iter([
        "0", "not JSON", "[]", '{"WOOD":true}', '{"WOOD":100}',
        '{"WOOD":2,"WOOD":2}', '{"WOOD":1}', valid_payload,
    ])
    prompts: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    choice: Any = (await HumanPlayer(context.actor, input_fn=answer).choose(context)).choice
    assert choice.discard_cards == ("WOOD",) * 2 + ("ORE",) * (context.discard_count - 2)
    assert len(prompts) == 8
    assert all(f"exactly {context.discard_count}" in prompt for prompt in prompts[1:])
    assert action_from_choice(context, choice).value == choice.discard_cards
