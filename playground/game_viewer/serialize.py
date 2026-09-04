"""Pure serialization utilities for game state — no Flask dependency."""

import json

from cle.game_engine.json import GameEncoder
from cle.game_engine.game import GameEngine
from .live.game_logging import get_player_resources, get_player_dev_cards


def serialize_game_for_inject(game: GameEngine) -> dict:
    """Serialize a GameEngine object into the payload expected by /api/inject-state.

    Returns a dict with 'game', 'all_player_resources', etc. ready to POST.
    """
    game_json = json.loads(json.dumps(game, cls=GameEncoder))

    all_resources = get_player_resources(game.state)
    all_dev_cards = get_player_dev_cards(game.state)

    player_types = {
        color.name: "Benchmark"
        for color in game.state.colors
    }

    return {
        "game": game_json,
        "all_player_resources": all_resources,
        "all_player_dev_cards": all_dev_cards,
        "player_types": player_types,
    }
