"""Shared feature-vector vocabulary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, TypeAlias

from cle.game_engine.models.player import Color

if TYPE_CHECKING:
    from cle.game_engine.game import GameEngine

FeatureValue: TypeAlias = bool | int | float
FeatureExtractor: TypeAlias = Callable[["GameEngine", Color], Mapping[str, FeatureValue]]
EdgePath: TypeAlias = list[tuple[int, int]]
LevelNodes: TypeAlias = tuple[int, set[int], dict[int, EdgePath]]
