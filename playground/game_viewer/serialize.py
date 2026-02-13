"""Pure serialization utilities for game state — no Flask dependency."""

import json

from engine.json import GameEncoder
from engine.game import Game
from .live.game_logging import get_player_resources, get_player_dev_cards


def serialize_game_for_inject(game: Game) -> dict:
    """Serialize a Game object into the payload expected by /api/inject-state.

    Returns a dict with 'game', 'all_player_resources', etc. ready to POST.
    """
    game_json = json.loads(json.dumps(game, cls=GameEncoder))

    all_resources = get_player_resources(game.state)
    all_dev_cards = get_player_dev_cards(game.state)

    player_types = {}
    for player in game.state.players:
        color_str = player.color.name if hasattr(player.color, 'name') else str(player.color)
        player_types[color_str] = "Benchmark"

    return {
        "game": game_json,
        "all_player_resources": all_resources,
        "all_player_dev_cards": all_dev_cards,
        "player_types": player_types,
    }
