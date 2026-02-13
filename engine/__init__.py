"""
This is to allow an API like:

from catanatron import Game, Player, Color
"""

from engine.game import Game
from engine.models.player import Player, Color, SimplePlayer
from engine.models.enums import (
    Action,
    ActionType,
    WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE,
    RESOURCES,
)
