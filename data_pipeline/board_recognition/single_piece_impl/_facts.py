"""Tile descriptions and the per-image tile question rows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TypedDict

from data_pipeline.board_recognition.single_piece_impl._rows import _row
from data_pipeline.board_recognition.spatial_localization import (
    _spatial_target,
    _stable_rank,
)
from data_pipeline.json_coerce import as_dict, as_list, as_str
from data_pipeline.json_types import JsonDict


class TileFact(TypedDict):
    """One tile's token, resource word, number word, and unique description."""

    token: str
    resource: str
    number: str
    description: str | None


def tile_facts(contract: JsonDict) -> list[TileFact]:
    """Tile token, resource word, number word, and a unique description if any.

    The description mirrors the inverse corpus ("Where is the 8 wheat tile?")
    and is only emitted when that number/resource pair is unique on the board.
    """

    tiles: list[TileFact] = []
    for raw_tile in as_list(contract["tiles"]):
        tile_row = as_dict(raw_tile)
        resource = "desert" if tile_row.get("resource") is None else str(tile_row["resource"]).lower()
        number = "none" if tile_row.get("number") is None else str(tile_row["number"])
        tiles.append({
            "token": as_str(tile_row["token"]),
            "resource": resource,
            "number": number,
            "description": None,
        })
    pairs = Counter((tile["resource"], tile["number"]) for tile in tiles)
    for tile in tiles:
        if tile["resource"] == "desert":
            tile["description"] = "the desert tile" if pairs[("desert", "none")] == 1 else None
        elif pairs[(tile["resource"], tile["number"])] == 1:
            tile["description"] = f"the {tile['number']} {tile['resource']} tile"
        else:
            tile["description"] = None
    return tiles


def tile_rows_for_image(
    *,
    state: JsonDict,
    tiles: Sequence[TileFact],
    regions: dict[str, JsonDict],
    controls: dict[str, JsonDict],
    image_name: str,
    salt: str,
) -> list[JsonDict]:
    """Resource, number, and inverse localization rows for one sampled tile.

    Tiles are printed on every board, so these rows cost no extra rendering and
    give the token-to-position direction a large, unambiguous target.
    """

    unique = [tile for tile in tiles if tile["description"]]
    pool = unique if unique else list(tiles)
    tile = pool[_stable_rank(state["sample_id"], salt, "tile") % len(pool)]
    token = tile["token"]
    metadata = {
        "split": state["split"],
        "state_id": state["sample_id"],
        "entity_type": "tile",
        "target_token": token,
        "piece": "TILE",
        "color": "none",
        "color_heldout": False,
    }
    target = _spatial_target(regions[token], controls[token])
    stem = f"{state['sample_id']}_{token[1:-1]}_{salt}"
    rows = [
        _row(
            row_id=f"{stem}_tile_resource",
            image_name=image_name,
            prompt=f"{token} resource?",
            answer=tile["resource"],
            task_type="tile_resource",
            category="tile.resource",
            polarity="positive",
            metadata=metadata,
            spatial_target=target,
        ),
        _row(
            row_id=f"{stem}_tile_number",
            image_name=image_name,
            prompt=f"{token} number?",
            answer=tile["number"],
            task_type="tile_number",
            category="tile.number",
            polarity="positive",
            metadata=metadata,
            spatial_target=target,
        ),
    ]
    if tile["description"]:
        rows.append(
            _row(
                row_id=f"{stem}_tile_to_token",
                image_name=image_name,
                prompt=f"Where is {tile['description']}?",
                answer=token,
                task_type="tile_to_token",
                category="localization",
                polarity="token_return",
                metadata=metadata,
                spatial_target=target,
            )
        )
    return rows
