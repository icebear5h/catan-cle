"""Production, trade identity, and discard evidence."""
import random
from typing import Any

import pytest

from cle.game_engine.models.decks import (
    CITY_COST_FREQDECK,
    RESOURCE_CARDS_PER_TYPE,
)
from cle.game_engine.models.enums import (
    CITY,
    RESOURCES,
    Action,
    ActionPrompt,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state import (
    validate_discard,
)
from cle.game_engine.state_functions import (
    get_player_freqdeck,
)
from cle.game_engine.trading import TradeCandidate, TradeOffer
from cle.sandbox import CatanSandbox

from .players import _AuditPlayer
from .support import COLORS, _assert_inventory, _expect_rule, _fund, _position, _turn


def test_sole_producer_receives_remaining_bank_card() -> None:
    game = _position(((Color.RED, 0),))
    _fund(game, Color.RED, CITY_COST_FREQDECK)
    game.step(Action(Color.RED, ActionType.BUILD_CITY, 0))
    tile: Any = next(
        tile
        for coordinate, tile in game.state.board.map.land_tiles.items()
        if 0 in tile.nodes.values()
        and tile.resource is not None
        and coordinate != game.state.board.robber_coordinate
    )
    index = RESOURCES.index(tile.resource)
    cards = [0] * len(RESOURCES)
    cards[index] = RESOURCE_CARDS_PER_TYPE - 1
    _fund(game, Color.BLUE, cards)
    _turn(game, Color.RED, rolled=False)
    _assert_inventory(game)
    assert game.state.board.buildings == {0: (Color.RED, CITY)}
    assert game.state.resource_freqdeck[index] == 1
    assert get_player_freqdeck(game.state, Color.RED)[index] == 0
    assert tile.number is not None
    first_die = max(1, tile.number - 6)
    game.step(Action(Color.RED, ActionType.ROLL, (first_die, tile.number - first_die)), force=True)
    _assert_inventory(game)
    observed = {
        "received": get_player_freqdeck(game.state, Color.RED)[index],
        "bank": game.state.resource_freqdeck[index],
    }
    _expect_rule(observed, {"received": 1, "bank": 0})


def test_sequential_completed_trades_have_distinct_offer_ids() -> None:
    game = _position()
    _fund(game, Color.RED, [1, 1, 0, 0, 0])
    _fund(game, Color.BLUE, [0, 0, 0, 0, 2])
    ids = []
    for give in ((1, 0, 0, 0, 0), (0, 1, 0, 0, 0)):
        offer = TradeOffer(Color.RED, frozenset({Color.BLUE}), give, (0, 0, 0, 0, 1))
        created = game.step(Action(Color.RED, ActionType.OFFER_TRADE, offer)).resolved_action.value
        assert isinstance(created.id, str) and created.id
        ids.append(created.id)
        game.step(Action(Color.BLUE, ActionType.ACCEPT_TRADE, created.id))
        game.step(
            Action(
                Color.RED,
                ActionType.CONFIRM_TRADE,
                TradeCandidate(created.id, Color.RED, Color.BLUE),
            )
        )
        _assert_inventory(game)
    assert get_player_freqdeck(game.state, Color.RED) == [0, 0, 0, 0, 2]
    assert get_player_freqdeck(game.state, Color.BLUE) == [1, 1, 0, 0, 0]
    assert game.state.num_turns == 0
    _expect_rule(len(set(ids)), 2)


@pytest.mark.parametrize(
    "limit,hands,next_color,next_prompt",
    [
        (5, (6, 6), Color.BLUE, ActionPrompt.DISCARD),
        (7, (8, 8), Color.BLUE, ActionPrompt.DISCARD),
        (10, (11, 8), Color.RED, ActionPrompt.MOVE_ROBBER),
    ],
)
def test_discard_threshold_applies_to_every_seat(limit: int, hands: tuple[int, int], next_color: Color, next_prompt: ActionPrompt) -> None:
    game = _position(discard_limit=limit)
    _fund(game, Color.RED, [hands[0], 0, 0, 0, 0])
    _fund(game, Color.BLUE, [0, hands[1], 0, 0, 0])
    _turn(game, Color.RED, rolled=False)
    game.step(Action(Color.RED, ActionType.ROLL, (3, 4)), force=True)
    assert game.state.current_color() == Color.RED
    assert game.state.current_prompt == ActionPrompt.DISCARD
    game.step(game.state.playable_actions[0])
    _assert_inventory(game)
    assert sum(get_player_freqdeck(game.state, Color.RED)) == hands[0] - hands[0] // 2
    assert sum(get_player_freqdeck(game.state, Color.BLUE)) == hands[1]
    observed = (game.state.current_color(), game.state.current_prompt)
    _expect_rule(observed, (next_color, next_prompt))


def test_player_can_choose_which_held_cards_to_discard() -> None:
    game = _position()
    _fund(game, Color.RED, [4, 4, 0, 0, 0])
    _turn(game, Color.RED, rolled=False)
    game.step(Action(Color.RED, ActionType.ROLL, (3, 4)), force=True)
    assert game.state.current_color() == Color.RED
    assert game.state.current_prompt == ActionPrompt.DISCARD
    assert get_player_freqdeck(game.state, Color.RED) == [4, 4, 0, 0, 0]
    _assert_inventory(game)
    choice = Action(Color.RED, ActionType.DISCARD, ["WOOD"] * 4)
    valid = game.is_action_valid(choice)
    revision = game.revision
    rejected = False
    try:
        game.step(choice)
    except ValueError:
        rejected = True
    _assert_inventory(game)
    observed = {
        "is_valid": valid,
        "rejected": rejected,
        "revision_delta": game.revision - revision,
        "hand": get_player_freqdeck(game.state, Color.RED),
    }
    _expect_rule(
        observed,
        {
            "is_valid": True,
            "rejected": False,
            "revision_delta": 1,
            "hand": [0, 4, 0, 0, 0],
        },
    )


@pytest.mark.filterwarnings("ignore:Prompt suite .* is deprecated:DeprecationWarning")
@pytest.mark.asyncio
async def test_local_audit_player_supplies_exact_discard_parameters() -> None:
    game = _position()
    _fund(game, Color.RED, [4, 4, 0, 0, 0])
    _turn(game, Color.RED, rolled=False)
    game.step(Action(Color.RED, ActionType.ROLL, (3, 4)), force=True)
    players = {color: _AuditPlayer(color, 1, random.Random(1)) for color in COLORS}
    sandbox = CatanSandbox(game, players)
    context = sandbox.decision_context()
    revision = game.revision
    attempt: Any = await players[Color.RED].choose(context)
    cards = attempt.choice.discard_cards
    assert isinstance(cards, tuple) and len(cards) == 4
    assert "<discard>" in attempt.choice.raw_response
    assert validate_discard(game.state, Action(Color.RED, ActionType.DISCARD, cards)) == cards
    assert game.revision == revision
    assert not players[Color.RED].session.receipts
    _assert_inventory(game)
