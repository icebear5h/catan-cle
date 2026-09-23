"""Leakage ledger loading and public-board contract gates."""

from __future__ import annotations

import json
from pathlib import Path

from data_pipeline.board_recognition.source_lock._config import (
    DEFAULT_LEAKAGE_LEDGER,
    ReplaySourceAuditError,
)
from data_pipeline.json_coerce import as_dict, as_list
from data_pipeline.json_types import JsonDict


def load_leakage_ledger(path: Path = DEFAULT_LEAKAGE_LEDGER) -> tuple[JsonDict, set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required benchmark leakage ledger is missing: {path}")
    payload = as_dict(json.loads(path.read_text()))
    game_ids = payload.get("benchmark_game_ids")
    if not isinstance(game_ids, list) or not game_ids:
        raise ReplaySourceAuditError("benchmark leakage ledger has no game IDs")
    return payload, {str(game_id) for game_id in game_ids}


def visible_board_facts(contract: JsonDict) -> JsonDict:
    """Return only facts that can alter the rendered public board image."""

    return {
        "tiles": [
            {
                "id": row["id"],
                "resource": row["resource"],
                "number": row["number"],
                "robber": bool(row["has_robber"]),
            }
            for row in map(as_dict, as_list(contract["tiles"]))
        ],
        "nodes": [
            {
                "id": row["id"],
                "color": row["color"],
                "building": row["building"],
            }
            for row in map(as_dict, as_list(contract["nodes"]))
        ],
        "edges": [
            {
                "id": row["id"],
                "owner": row["road_color"],
            }
            for row in map(as_dict, as_list(contract["edges"]))
        ],
        "ports": [
            {
                "id": row["id"],
                "kind": row["kind"],
                "resource": row["resource"],
            }
            for row in map(as_dict, as_list(contract["ports"]))
        ],
    }


def validate_public_board_contract(contract: JsonDict) -> None:
    expected_counts = {"tiles": 19, "nodes": 54, "edges": 72, "ports": 9}
    actual_counts = {name: len(as_list(contract.get(name, []))) for name in expected_counts}
    if actual_counts != expected_counts:
        raise ReplaySourceAuditError(
            f"public board topology mismatch: expected={expected_counts} actual={actual_counts}"
        )
    tiles = [as_dict(tile) for tile in as_list(contract["tiles"])]
    if sum(bool(tile.get("has_robber")) for tile in tiles) != 1:
        raise ReplaySourceAuditError("public board contract must contain exactly one robber")
    players = [as_dict(player) for player in as_list(contract.get("players", []))]
    colors = [player.get("color") for player in players]
    if len(colors) != 4 or len(set(colors)) != 4:
        raise ReplaySourceAuditError(f"expected four distinct replay colors, received {colors}")


__all__ = [
    "load_leakage_ledger",
    "validate_public_board_contract",
    "visible_board_facts",
]
