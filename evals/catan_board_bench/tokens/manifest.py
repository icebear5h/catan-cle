"""Token specs, the exported manifest, and tokenizer registration."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Protocol

from cle.game_engine.board_tokens import (
    edge_token,
    node_token,
    port_token,
    tile_token,
)
from cle.game_engine.models.enums import CITY, RESOURCES, ROAD, SETTLEMENT, ActionType
from cle.game_engine.models.map import NUM_EDGES, NUM_NODES, NUM_TILES
from cle.game_engine.models.player import Color
from evals.catan_board_bench.tokens.atlas import atlas_metadata_json, base_edges
from evals.catan_board_bench.tokens.vocabulary import (
    BOARD_OBJECTS,
    RECOGNITION_CLASS_VOCABULARIES,
    RECOGNITION_HEADS,
    CatanTokenSpec,
    action_token,
    building_token,
    color_token,
    object_token,
    recognition_answer_token,
    recognition_answer_tokens,
    recognition_query_token,
    recognition_query_tokens,
    resource_token,
)
from evals.json_types import JsonDict


def token_specs() -> List[CatanTokenSpec]:
    """Return all Catan added-token specs in stable order."""

    specs: List[CatanTokenSpec] = []

    specs.extend(
        CatanTokenSpec(node_token(node_id), "node", node_id) for node_id in range(NUM_NODES)
    )
    specs.extend(CatanTokenSpec(edge_token(edge), "edge", list(edge)) for edge in base_edges())
    specs.extend(
        CatanTokenSpec(tile_token(tile_id), "tile", tile_id) for tile_id in range(NUM_TILES)
    )
    specs.extend(CatanTokenSpec(port_token(port_id), "port", port_id) for port_id in range(9))
    specs.extend(
        CatanTokenSpec(resource_token(resource), "resource", resource)
        for resource in [*RESOURCES, None]
    )
    specs.extend(
        CatanTokenSpec(object_token(object_name), "object", object_name)
        for object_name in BOARD_OBJECTS
    )
    specs.extend(CatanTokenSpec(color_token(color), "color", color.value) for color in Color)
    specs.extend(
        CatanTokenSpec(building_token(building), "building", building)
        for building in [SETTLEMENT, CITY, ROAD]
    )
    specs.extend(
        CatanTokenSpec(action_token(action_type), "action", action_type.value)
        for action_type in ActionType
    )
    specs.extend(
        CatanTokenSpec(recognition_query_token(head), "recognition_query", head)
        for head in RECOGNITION_HEADS
    )
    specs.extend(
        CatanTokenSpec(
            recognition_answer_token(head, class_name),
            "recognition_answer",
            {"head": head, "class_name": class_name},
        )
        for head in RECOGNITION_HEADS
        for class_name in RECOGNITION_CLASS_VOCABULARIES[head]
    )

    tokens = [spec.token for spec in specs]
    if len(tokens) != len(set(tokens)):
        duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
        raise RuntimeError(f"duplicate Catan tokens: {duplicates}")

    return specs


def added_tokens() -> List[str]:
    """Return just the regular added vocabulary tokens."""

    return [spec.token for spec in token_specs()]


def atlas_tokens() -> List[str]:
    """Return the 154 stable board-location tokens in canonical order."""

    categories = {"node", "edge", "tile", "port"}
    return [spec.token for spec in token_specs() if spec.category in categories]


def recognition_trainable_tokens() -> List[str]:
    """Return the historical atlas/query/answer rows used by Qwen SFT."""

    return [*atlas_tokens(), *recognition_query_tokens(), *recognition_answer_tokens()]


def recognition_token_inventory() -> JsonDict:
    tokens = recognition_trainable_tokens()
    return {
        "schema": "catan_board_recognition_token_inventory/v1",
        "atlas_tokens": list(atlas_tokens()),
        "query_tokens": list(recognition_query_tokens()),
        "answer_tokens": list(recognition_answer_tokens()),
        "tokens": list(tokens),
        "counts": {
            "atlas": 154,
            "query": len(recognition_query_tokens()),
            "answer": len(recognition_answer_tokens()),
            "total": len(tokens),
        },
    }


def semantic_recognition_token_inventory() -> JsonDict:
    """Return the semantic projection's bidirectional atlas-row inventory."""

    tokens = atlas_tokens()
    return {
        "schema": "catan_board_recognition_token_inventory/v3",
        "token_type": "regular_added_tokens",
        "trainable_side": "input_and_output_rows",
        "atlas_tokens": list(tokens),
        "tokens": list(tokens),
        "counts": {
            "node": NUM_NODES,
            "edge": NUM_EDGES,
            "tile": NUM_TILES,
            "port": 9,
            "atlas": len(tokens),
            "total": len(tokens),
        },
    }


def token_manifest() -> JsonDict:
    specs = token_specs()
    categories: dict[str, List[str]] = {}
    for spec in specs:
        categories.setdefault(spec.category, []).append(spec.token)

    return {
        "token_type": "regular_added_tokens",
        "note": "Use tokenizer.add_tokens(...), not tokenizer.add_special_tokens(...).",
        "counts": {category: len(tokens) for category, tokens in categories.items()},
        "total": len(specs),
        "tokens": [spec.token for spec in specs],
        "specs": [asdict(spec) for spec in specs],
        "atlas": atlas_metadata_json(),
    }


def write_token_manifest(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(token_manifest(), indent=2) + "\n")
    return output_path


class TokenAdder(Protocol):
    """The slice of a Hugging Face tokenizer that registers regular tokens."""

    def add_tokens(self, new_tokens: List[str], /) -> int: ...


def add_tokens_to_tokenizer(tokenizer: TokenAdder) -> int:
    """Add Catan tokens to a Hugging Face tokenizer as regular tokens."""

    return tokenizer.add_tokens(added_tokens())

