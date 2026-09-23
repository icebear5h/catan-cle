"""Typed records shared by the VLM player's prompt, decision, and trace paths."""

from __future__ import annotations

from typing import TypedDict

__all__ = ["TurnTrace"]


class TurnTrace(TypedDict):
    """One action already taken this turn, replayed back to the model."""

    action_desc: str
    turn_plan: str
    action_idx: int
