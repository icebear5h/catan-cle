"""Canonical identifier and digest helpers shared by public-board snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

PUBLIC_BOARD_CONTRACT_SCHEMA = "catan_public_board_contract/v0"
PUBLIC_BOARD_FACTS_SCHEMA = "catan_full_public_graph/v1"
CANONICAL_BOARD_IDENTITY = "canonical_engine_ids"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_edge(edge: Sequence[int]) -> tuple[int, int]:
    first, second = edge
    return (first, second) if first < second else (second, first)


def tile_id(tile_id: int) -> str:
    return f"T{tile_id:02d}"


def node_id(node_id: int) -> str:
    return f"N{node_id:02d}"


def edge_id(edge: Sequence[int]) -> str:
    first, second = canonical_edge(edge)
    return f"E{first:02d}_{second:02d}"


def port_id(port_id: int) -> str:
    return f"P{port_id:02d}"
