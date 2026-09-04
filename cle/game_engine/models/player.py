"""Participant identity types used by the rules engine."""

from enum import Enum


class Color(Enum):
    """A participant color and stable game identity."""

    RED = "RED"
    BLUE = "BLUE"
    ORANGE = "ORANGE"
    WHITE = "WHITE"
    BLACK = "BLACK"
    GREEN = "GREEN"
    BRONZE = "BRONZE"
    SILVER = "SILVER"
    GOLD = "GOLD"
    PINK = "PINK"
    MYSTIC_BLUE = "MYSTIC_BLUE"

    def __repr__(self) -> str:
        return f"C.{self.name}"
