"""Deterministic participant-color selection shared by live games and datagen."""

from __future__ import annotations

import hashlib
import random
from typing import Literal, cast

from cle.game_engine.models.player import Color


PaletteMode = Literal["random_all", "canonical_four"]
CANONICAL_FOUR = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
ALL_COLORS = tuple(Color)
PALETTE_MODES = ("random_all", "canonical_four")


def validate_palette_mode(mode: str) -> PaletteMode:
    if mode not in PALETTE_MODES:
        raise ValueError(f"Unknown color palette {mode!r}; choose random_all or canonical_four")
    return cast(PaletteMode, mode)


def palette_seed(game_seed: int) -> int:
    digest = hashlib.sha256(f"catan-color-palette/v1:{game_seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def select_game_colors(
    mode: PaletteMode = "random_all",
    *,
    seed: int | None = None,
) -> tuple[Color, Color, Color, Color]:
    """Select four colors without consuming the engine's own RNG stream."""

    validate_palette_mode(mode)
    if mode == "canonical_four":
        return CANONICAL_FOUR
    chooser: random.Random | random.SystemRandom
    chooser = random.Random(palette_seed(seed)) if seed is not None else random.SystemRandom()
    selected = chooser.sample(ALL_COLORS, 4)
    return cast(tuple[Color, Color, Color, Color], tuple(selected))


def balanced_datagen_colors(index: int, *, seed: int) -> tuple[Color, Color, Color, Color]:
    """Return a seeded four-color window with near-uniform corpus coverage."""

    if index < 0:
        raise ValueError("datagen palette index must be non-negative")
    order = list(ALL_COLORS)
    random.Random(palette_seed(seed)).shuffle(order)
    start = (index * 4) % len(order)
    selected = tuple(order[(start + offset) % len(order)] for offset in range(4))
    return cast(tuple[Color, Color, Color, Color], selected)
