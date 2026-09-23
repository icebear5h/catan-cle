"""Validation for caller-supplied model and strategic-memory inputs."""

from __future__ import annotations

import re

MAX_GAME_PLAN_CHARS = 4_000
MAX_MODEL_ID_CHARS = 200
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+~-]*$")


class HarnessValidationError(ValueError):
    """Raised when a model request cannot form a valid player decision."""


def validate_model_id(model: object) -> str:
    """Return a normalized provider/model identifier."""
    if not isinstance(model, str) or not model.strip():
        raise HarnessValidationError("model is required")
    normalized = model.strip()
    if len(normalized) > MAX_MODEL_ID_CHARS:
        raise HarnessValidationError(
            f"model must be at most {MAX_MODEL_ID_CHARS} characters"
        )
    if not _MODEL_ID_PATTERN.fullmatch(normalized):
        raise HarnessValidationError(
            "model contains unsupported characters; use a provider/model ID"
        )
    return normalized


def validate_game_plan(game_plan: object) -> str:
    """Return bounded caller-held strategic memory for one player session."""
    if game_plan is None:
        return ""
    if not isinstance(game_plan, str):
        raise HarnessValidationError("game_plan must be a string")
    if len(game_plan) > MAX_GAME_PLAN_CHARS:
        raise HarnessValidationError(
            f"game_plan must be at most {MAX_GAME_PLAN_CHARS} characters"
        )
    return game_plan.strip()
