"""The board's fixed colour vocabulary: tiles, players, tokens, and pips."""

from cle.game_engine.models.player import Color

__all__ = [
    "OCEAN_COLOR",
    "PIPS",
    "PLAYER_COLORS",
    "PLAYER_OUTLINE",
    "RESOURCE_COLORS",
    "RESOURCE_LABELS",
    "ROBBER_COLOR",
    "TOKEN_BG",
    "TOKEN_BORDER",
]

RESOURCE_COLORS: dict[str | None, str] = {
    "WOOD": "#228b22",
    "BRICK": "#c45a2c",
    "SHEEP": "#90ee90",
    "WHEAT": "#f4d03f",
    "ORE": "#708090",
    None: "#d2b48c",  # desert
}

RESOURCE_LABELS: dict[str | None, str] = {
    "WOOD": "W",
    "BRICK": "B",
    "SHEEP": "S",
    "WHEAT": "Wh",
    "ORE": "O",
    None: "",
}

PLAYER_COLORS: dict[Color, str] = {
    Color.RED: "#dc2626",
    Color.BLUE: "#2563eb",
    Color.WHITE: "#e5e7eb",
    Color.ORANGE: "#ea580c",
}

PLAYER_OUTLINE: dict[Color, str] = {
    Color.RED: "#991b1b",
    Color.BLUE: "#1d4ed8",
    Color.WHITE: "#9ca3af",
    Color.ORANGE: "#c2410c",
}

OCEAN_COLOR: str = "#1e3a5f"
TOKEN_BG: str = "#fef3c7"
TOKEN_BORDER: str = "#92400e"
ROBBER_COLOR: str = "#1f2937"

# Number -> pip count (probability dots)
PIPS: dict[int, int] = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
