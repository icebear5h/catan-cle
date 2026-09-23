"""Legacy package paths must remain live helper lookup sites after extraction."""

from collections.abc import Callable
from dataclasses import replace
from types import ModuleType
from typing import ParamSpec, TypeVar

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer, TradeWindow
from cle.harness import action_tools, board_surface
from cle.harness.catan_board_surface import ImageBoardPresenter, IndexedTileRowsBoardPresenter
from cle.players.contracts import PlayerContext
from cle.players.data import JsonValue

_P = ParamSpec("_P")
_R = TypeVar("_R")
COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _watch(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, original: Callable[_P, _R],
) -> list[None]:
    calls: list[None] = []

    def tracing_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(f"{module.__name__}.{original.__name__}", tracing_wrapper)
    return calls


def _context(actor: Color = Color.RED) -> PlayerContext:
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    return PlayerContext(
        context_id="helper-patch-sites", actor=actor,
        turn_number=engine.state.num_turns, phase="main_game",
        observation=engine.observe(actor), events=(),
        legal_actions=tuple(engine.state.playable_actions), prompt_key="main_game",
    )


def test_settlement_parser_and_renderers_use_original_helper_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    arguments: dict[str, JsonValue] = {"node": action_tools.node_token(context.legal_actions[0].value)}
    expected = action_tools.parse_tool_choice(context, "build_settlement", arguments)
    rendered = action_tools.render_action_tools(context)
    shared = action_tools.render_shared_legal_actions(context)
    spatial = _watch(monkeypatch, action_tools, action_tools._spatial_values)
    required = _watch(monkeypatch, action_tools, action_tools._require_arguments)
    admitted = _watch(monkeypatch, action_tools, action_tools.action_from_choice)
    tokens = _watch(monkeypatch, action_tools, action_tools.node_token)
    names = _watch(monkeypatch, action_tools, action_tools.legal_tool_names)
    note = _watch(monkeypatch, action_tools, action_tools.trade_responder_note)

    assert action_tools.parse_tool_choice(context, "build_settlement", arguments) == expected
    assert len(spatial) == len(required) == len(admitted) == 1
    assert tokens
    assert action_tools.render_action_tools(context) == rendered
    assert len(spatial) == 2
    assert action_tools.render_shared_legal_actions(context) == shared
    assert len(names) == len(note) == 1


def test_shared_counteroffer_uses_original_nested_trade_helper_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(Color.BLUE)
    window = TradeWindow("helper-window", Color.RED, COLORS)
    root = window.create_offer(TradeOffer(
        Color.RED, frozenset({Color.BLUE}), (1, 0, 0, 0, 0), (0, 0, 0, 0, 1),
    ))
    context.observation.trade_window = window
    context.observation.my_resources["ORE"] = 2
    context = replace(context, legal_actions=(Action(
        Color.BLUE, ActionType.COUNTER_OFFER,
        f"COUNTER_OFFER:{root.id}: supply a named trade_offer",
    ),))
    arguments: dict[str, JsonValue] = {
        "player": "RED", "original": {"give": {"ORE": 1}, "receive": {"WOOD": 1}},
        "proposed": {"give": {"ORE": 2}, "receive": {"WOOD": 1}},
    }
    expected = action_tools.parse_tool_choice(context, "counter_offer", arguments, shared=True)
    rendered = action_tools.render_action_tools(context)
    shared = _watch(monkeypatch, action_tools, action_tools._shared_trade_arguments)
    semantic = _watch(monkeypatch, action_tools, action_tools._semantic_offer)
    parent = _watch(monkeypatch, action_tools, action_tools._counter_parent)
    bundle = _watch(monkeypatch, action_tools, action_tools._bundle)
    resource = _watch(monkeypatch, action_tools, action_tools._resource)
    color = _watch(monkeypatch, action_tools, action_tools._color)
    offer_id = _watch(monkeypatch, action_tools, action_tools._offer_id)

    assert action_tools.parse_tool_choice(context, "counter_offer", arguments, shared=True) == expected
    assert len(shared) == len(semantic) == len(parent) == len(color) == len(offer_id) == 1
    assert len(bundle) == len(resource) == 4
    assert action_tools.render_action_tools(context) == rendered
    assert len(parent) == 2


@pytest.mark.parametrize(
    ("tool", "action", "arguments"),
    [
        ("play_year_of_plenty", Action(Color.RED, ActionType.PLAY_YEAR_OF_PLENTY, ("WOOD", "ORE")),
         {"take": {"WOOD": 1, "ORE": 1}}),
        ("play_monopoly", Action(Color.RED, ActionType.PLAY_MONOPOLY, "ORE"), {"resource": "ore"}),
        ("steal_from", Action(Color.RED, ActionType.STEAL, (Color.BLUE, None)), {"player": "blue"}),
    ],
)
def test_resource_and_victim_parser_helpers_are_late_bound(
    monkeypatch: pytest.MonkeyPatch, tool: str, action: Action, arguments: dict[str, JsonValue],
) -> None:
    context = replace(_context(), legal_actions=(action,))
    expected = action_tools.parse_tool_choice(context, tool, arguments)
    cards = _watch(monkeypatch, action_tools, action_tools._card_counts)
    resource = _watch(monkeypatch, action_tools, action_tools._resource)
    color = _watch(monkeypatch, action_tools, action_tools._color)
    assert action_tools.parse_tool_choice(context, tool, arguments) == expected
    if tool == "play_year_of_plenty":
        assert len(cards) == 1 and len(resource) == 2
    elif tool == "play_monopoly":
        assert len(resource) == 1
    else:
        assert len(color) == 1


def test_board_factories_and_validation_use_original_digest_helper_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    text = IndexedTileRowsBoardPresenter().present(context)
    image = ImageBoardPresenter(image_size=512).present(context)
    hashes = _watch(monkeypatch, board_surface, board_surface._sha256_bytes)
    validations = _watch(monkeypatch, board_surface, board_surface._require_sha256)
    assert replace(text.provenance) == text.provenance
    assert board_surface.TextBoardPresentation.create(
        provenance=text.provenance, format=text.format, content=text.content,
    ) == text
    assert board_surface.ImageBoardPresentation.create(
        provenance=image.provenance, format=image.format, media_type=image.media_type,
        data=image.data, width=image.width, height=image.height,
        contains_entity_labels=image.contains_entity_labels,
    ) == image
    assert len(hashes) == 4 and len(validations) == 3


def test_recursive_redaction_uses_original_sanitizer_path(monkeypatch: pytest.MonkeyPatch) -> None:
    image = ImageBoardPresenter(image_size=512).present(_context())
    payload = {"nested": (["data:image/png;base64,a"], b"unchanged")}
    expected = board_surface.sanitize_provider_payload(payload, image)
    calls = _watch(monkeypatch, board_surface, board_surface.sanitize_provider_payload)
    assert board_surface.sanitize_provider_payload(payload, image) == expected
    assert len(calls) == 5  # dict, tuple, list, image URI, untouched bytes.
