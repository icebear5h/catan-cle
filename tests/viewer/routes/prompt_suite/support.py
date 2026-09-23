"""Prompt-suite edit payload builders and a ready live sandbox."""
from typing import Any

from cle.game_engine.game import GameEngine
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox import CatanSandbox

from .conftest import COLORS


def _edits(payload: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "decision": {
            "system_identity": payload["decision"]["system_identity"],
            "components": payload["decision"]["components"],
            "phase_guidance": payload["decision"]["phase_guidance"],
            "response_instruction": payload["decision"]["response_instruction"],
        },
        "communication": {
            "system_identity": payload["communication"]["system_identity"],
            "components": payload["communication"]["components"],
        },
    }


def _save_payload(
    payload: dict[str, dict[str, Any]],
    edits: dict[str, Any],
) -> dict[str, Any]:
    return {
        "expected": {
            "decision": payload["decision"]["sha256"],
            "communication": payload["communication"]["sha256"],
        },
        **edits,
    }


def _sandbox() -> CatanSandbox:
    engine = GameEngine(COLORS, seed=5, shuffle_players=False)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    return CatanSandbox(engine, players)
