"""Build the single-piece localization curriculum (``spatial_localization_v2``).

Each image is a validated empty replay board with exactly one real piece added
through the production renderer: a settlement or city on one node, or a road
on one edge, in one player color. Four rows accompany every image:

- ``piece_to_token``: "Which node has the settlement?" -> ``<N17>``
- ``colored_piece_to_token``: "Which node has the red settlement?" -> ``<N17>``
- ``occupancy_positive``: "<N17> building?" -> "red settlement"
- ``occupancy_negative``: "<N20> building?" -> "empty" for another location

The forward rows use the exact production ``node.occupancy`` and
``edge.owner`` prompts, so this stage trains the heads that collapsed to
"empty" on dense boards, with a single piece and no clutter. One player color
is withheld from training and appears only in validation and test, which makes
the held-out color the generalization probe. Train rows are written in a
deterministic shuffled order; validation and test keep canonical order.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from data_pipeline.board_recognition.replay_dataset import (
    DEFAULT_STYLE_PATH,
    file_sha256,
    load_render_style,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _control_regions,
    _deterministic_shuffle,
    _spatial_target,
    _stable_rank,
    _summarize_rows,
    _write_json,
    _write_jsonl,
    atlas_regions,
)
from evals.catan_board_bench.render import render_contract_image


JsonDict = dict[str, Any]
EXPORT_SCHEMA = "catan_single_piece_localization/v1"
ROW_SCHEMA = "catan_single_piece_localization_row/v1"
DEFAULT_OUTPUT_NAME = "spatial_localization_v2"
COLORS = (
    "RED",
    "BLUE",
    "ORANGE",
    "WHITE",
    "BLACK",
    "GREEN",
    "BRONZE",
    "SILVER",
    "GOLD",
    "PINK",
    "MYSTIC_BLUE",
)
HELDOUT_COLOR = "PINK"
NODE_PIECES = ("SETTLEMENT", "CITY")
EDGE_PIECE = "ROAD"
TRAIN_IMAGES_PER_BOARD_PER_ENTITY = 70
EVAL_IMAGES_PER_BOARD_PER_ENTITY = 30
FORWARD_QUERY = {"node": "building?", "edge": "road?"}
LOCATION_NOUN = {"node": "node", "edge": "edge"}
TILE_ROWS_PER_IMAGE = 3


def tile_facts(contract: JsonDict) -> list[JsonDict]:
    """Tile token, resource word, number word, and a unique description if any.

    The description mirrors the inverse corpus ("Where is the 8 wheat tile?")
    and is only emitted when that number/resource pair is unique on the board.
    """

    tiles = []
    for tile in contract["tiles"]:
        resource = "desert" if tile.get("resource") is None else str(tile["resource"]).lower()
        number = "none" if tile.get("number") is None else str(tile["number"])
        tiles.append({"token": tile["token"], "resource": resource, "number": number})
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
    tiles: Sequence[JsonDict],
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


def color_words(color: str) -> str:
    return color.lower().replace("_", " ")


def piece_words(piece: str) -> str:
    return piece.lower()


def forward_answer(color: str, piece: str) -> str:
    return f"{color_words(color)} {piece_words(piece)}"


def piece_combinations(entity_type: str, colors: Sequence[str]) -> list[tuple[str, str]]:
    """All (piece, color) pairs for one entity type, in a stable order."""

    pieces = NODE_PIECES if entity_type == "node" else (EDGE_PIECE,)
    return [(piece, color) for piece in pieces for color in colors]


def sample_placements(
    *,
    sample_id: str,
    entity_type: str,
    tokens: Sequence[str],
    colors: Sequence[str],
    count: int,
) -> list[tuple[str, str, str]]:
    """Deterministically pick ``count`` (token, piece, color) placements.

    Every location, piece, and color is ranked by a stable hash of the board so
    each board sees a different spread while the export stays reproducible.
    """

    universe = [
        (token, piece, color)
        for token in tokens
        for piece, color in piece_combinations(entity_type, colors)
    ]
    if count > len(universe):
        raise SpatialLocalizationError(
            f"requested {count} {entity_type} placements from {len(universe)} combinations"
        )
    ranked = sorted(
        universe,
        key=lambda item: (_stable_rank(sample_id, entity_type, *item, "placement"), item),
    )
    return ranked[:count]


def place_piece(contract: JsonDict, token: str, piece: str, color: str) -> JsonDict:
    """Return a deep copy of ``contract`` with exactly one added piece."""

    updated = copy.deepcopy(contract)
    if piece in NODE_PIECES:
        node = next((entry for entry in updated["nodes"] if entry["token"] == token), None)
        if node is None:
            raise SpatialLocalizationError(f"node token not in contract: {token}")
        if node.get("building") is not None:
            raise SpatialLocalizationError(f"node already occupied: {token}")
        node["building"] = piece
        node["building_token"] = f"<{piece}>"
        node["color"] = color
        node["color_token"] = f"<{color}>"
        return updated
    if piece != EDGE_PIECE:
        raise SpatialLocalizationError(f"unknown piece: {piece}")
    edge = next((entry for entry in updated["edges"] if entry["token"] == token), None)
    if edge is None:
        raise SpatialLocalizationError(f"edge token not in contract: {token}")
    if edge.get("road_color") is not None:
        raise SpatialLocalizationError(f"edge already occupied: {token}")
    edge["road_color"] = color
    edge["road_color_token"] = f"<{color}>"
    return updated


def _row(
    *,
    row_id: str,
    image_name: str,
    prompt: str,
    answer: str,
    task_type: str,
    category: str,
    polarity: str,
    metadata: JsonDict,
    spatial_target: JsonDict | None,
) -> JsonDict:
    row = {
        "schema": ROW_SCHEMA,
        "row_id": row_id,
        "curriculum_stage": "spatial_grounding",
        "grounding_stage": "single_piece",
        "task_family": "single_piece_localization",
        "task_type": task_type,
        "category": category,
        "polarity": polarity,
        "images": [image_name],
        "messages": [
            {"role": "user", "content": f"<image>\n{prompt}"},
            {"role": "assistant", "content": answer},
        ],
        **metadata,
    }
    if spatial_target is not None:
        row["spatial_targets"] = [spatial_target]
    return row


def rows_for_placement(
    *,
    state: JsonDict,
    regions: dict[str, JsonDict],
    controls: dict[str, JsonDict],
    token: str,
    piece: str,
    color: str,
    image_name: str,
    empty_token: str,
) -> list[JsonDict]:
    entity_type = regions[token]["entity_type"]
    noun = LOCATION_NOUN[entity_type]
    heldout = color == HELDOUT_COLOR
    metadata = {
        "split": state["split"],
        "state_id": state["sample_id"],
        "entity_type": entity_type,
        "target_token": token,
        "piece": piece,
        "color": color_words(color),
        "color_heldout": heldout,
    }
    target = _spatial_target(regions[token], controls[token])
    stem = f"{state['sample_id']}_{token[1:-1]}_{piece}_{color}"
    category = "node.occupancy" if entity_type == "node" else "edge.owner"
    return [
        _row(
            row_id=f"{stem}_piece_to_token",
            image_name=image_name,
            prompt=f"Which {noun} has the {piece_words(piece)}?",
            answer=token,
            task_type="piece_to_token",
            category="localization",
            polarity="token_return",
            metadata=metadata,
            spatial_target=target,
        ),
        _row(
            row_id=f"{stem}_colored_piece_to_token",
            image_name=image_name,
            prompt=f"Which {noun} has the {forward_answer(color, piece)}?",
            answer=token,
            task_type="colored_piece_to_token",
            category="localization",
            polarity="token_return",
            metadata=metadata,
            spatial_target=target,
        ),
        _row(
            row_id=f"{stem}_occupancy_positive",
            image_name=image_name,
            prompt=f"{token} {FORWARD_QUERY[entity_type]}",
            answer=forward_answer(color, piece),
            task_type="occupancy_positive",
            category=category,
            polarity="positive",
            metadata=metadata,
            spatial_target=target,
        ),
        _row(
            row_id=f"{stem}_occupancy_negative",
            image_name=image_name,
            prompt=f"{empty_token} {FORWARD_QUERY[entity_type]}",
            answer="empty",
            task_type="occupancy_negative",
            category=category,
            polarity="hard_negative",
            metadata={**metadata, "queried_token": empty_token},
            spatial_target=None,
        ),
    ]


def render_placement(
    contract: JsonDict,
    token: str,
    piece: str,
    color: str,
    image_size: int,
    style: Any,
    destination: Path,
) -> str:
    """Render one single-piece board to ``destination`` (runs in a worker)."""

    rendered = render_contract_image(
        place_piece(contract, token, piece, color),
        image_size=image_size,
        style=style,
    )
    if rendered.size != (image_size, image_size):
        raise SpatialLocalizationError(f"renderer returned {rendered.size}")
    rendered.convert("RGB").save(destination)
    return destination.name


def build_board(
    *,
    state: JsonDict,
    contract: JsonDict,
    output_images: Path,
    style: Any,
    images_per_entity: int,
    colors: Sequence[str],
    pool: ProcessPoolExecutor | None = None,
    tile_rows: bool = False,
) -> list[JsonDict]:
    """Render every sampled single-piece board for one empty state."""

    image_size = int(state["image_size"][0])
    regions = atlas_regions(
        contract,
        image_size=image_size,
        view_padding_factor=style.view_padding_factor,
    )
    controls = _control_regions(regions)
    tiles = tile_facts(contract) if tile_rows else []
    rows: list[JsonDict] = []
    for entity_type in ("node", "edge"):
        tokens = [token for token, region in regions.items() if region["entity_type"] == entity_type]
        placements = sample_placements(
            sample_id=state["sample_id"],
            entity_type=entity_type,
            tokens=tokens,
            colors=colors,
            count=images_per_entity,
        )
        jobs = [
            (
                token,
                piece,
                color,
                output_images / f"{state['split']}_{state['sample_id']}_{token[1:-1]}_{piece}_{color}.png",
            )
            for token, piece, color in placements
        ]
        if pool is None:
            for token, piece, color, destination in jobs:
                render_placement(contract, token, piece, color, image_size, style, destination)
        else:
            futures = [
                pool.submit(render_placement, contract, token, piece, color, image_size, style, destination)
                for token, piece, color, destination in jobs
            ]
            for future in futures:
                future.result()
        for token, piece, color, destination in jobs:
            image_name = destination.name
            others = [candidate for candidate in tokens if candidate != token]
            empty_token = others[_stable_rank(state["sample_id"], token, piece, color, "empty") % len(others)]
            rows.extend(
                rows_for_placement(
                    state=state,
                    regions=regions,
                    controls=controls,
                    token=token,
                    piece=piece,
                    color=color,
                    image_name=image_name,
                    empty_token=empty_token,
                )
            )
            if tile_rows:
                rows.extend(
                    tile_rows_for_image(
                        state=state,
                        tiles=tiles,
                        regions=regions,
                        controls=controls,
                        image_name=image_name,
                        salt=f"{token[1:-1]}_{piece}_{color}",
                    )
                )
    return rows


def export_single_piece_curriculum(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    style_path: str | Path = DEFAULT_STYLE_PATH,
    overwrite: bool = False,
    train_images_per_board_per_entity: int = TRAIN_IMAGES_PER_BOARD_PER_ENTITY,
    eval_images_per_board_per_entity: int = EVAL_IMAGES_PER_BOARD_PER_ENTITY,
    workers: int | None = None,
    tile_rows: bool = False,
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    )
    validate_replay_v1_dataset(dataset_root, rerender=False)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        if output.parent != dataset_root:
            raise SpatialLocalizationError("refusing to overwrite output outside the replay dataset")
        shutil.rmtree(output)
    images_dir = output / "images"
    images_dir.mkdir(parents=True)

    style = load_render_style(Path(style_path))
    states = [row for row in read_jsonl(dataset_root / "manifest.jsonl") if row["density_bin"] == "empty"]
    states_by_split = {
        split: [row for row in states if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    if {split: len(rows) for split, rows in states_by_split.items()} != {
        "train": 43,
        "validation": 5,
        "test": 5,
    }:
        raise SpatialLocalizationError("replay_v1 empty-board split counts changed")
    train_colors = tuple(color for color in COLORS if color != HELDOUT_COLOR)

    files: dict[str, JsonDict] = {}
    worker_count = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        for split, split_states in states_by_split.items():
            rows: list[JsonDict] = []
            for state in split_states:
                contract_path = dataset_root / state["contract_path"]
                if file_sha256(contract_path) != state["sha256"]["contract"]:
                    raise SpatialLocalizationError(f"contract changed: {contract_path}")
                rows.extend(
                    build_board(
                        state=state,
                        contract=json.loads(contract_path.read_text()),
                        output_images=images_dir,
                        style=style,
                        images_per_entity=(
                            train_images_per_board_per_entity
                            if split == "train"
                            else eval_images_per_board_per_entity
                        ),
                        colors=train_colors if split == "train" else COLORS,
                        pool=pool,
                        tile_rows=tile_rows,
                    )
                )
            if split == "train":
                rows = _deterministic_shuffle(rows, "single_piece")
            path = output / "stage1" / f"{split}.jsonl"
            _write_jsonl(path, rows)
            summary = _summarize_rows(rows)
            summary["dimensions"]["color"] = dict(sorted(Counter(row["color"] for row in rows).items()))
            summary["dimensions"]["piece"] = dict(sorted(Counter(row["piece"] for row in rows).items()))
            summary["unique_images"] = len({row["images"][0] for row in rows})
            files[f"stage1/{split}.jsonl"] = {**summary, "sha256": file_sha256(path)}

    metadata = {
        "schema": EXPORT_SCHEMA,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": file_sha256(dataset_root / "manifest.jsonl"),
        "style_sha256": file_sha256(Path(style_path)),
        "colors": list(COLORS),
        "heldout_color": HELDOUT_COLOR,
        "node_pieces": list(NODE_PIECES),
        "edge_piece": EDGE_PIECE,
        "train_images_per_board_per_entity": train_images_per_board_per_entity,
        "eval_images_per_board_per_entity": eval_images_per_board_per_entity,
        "rows_per_image": 4 + (TILE_ROWS_PER_IMAGE if tile_rows else 0),
        "tile_rows": tile_rows,
        "train_row_order": "deterministic_shuffle",
        "files": files,
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--style-path", type=Path, default=DEFAULT_STYLE_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--tile-rows",
        action="store_true",
        help="Add tile resource, number, and inverse rows for one tile per image.",
    )
    parser.add_argument(
        "--train-images-per-board-per-entity",
        type=int,
        default=TRAIN_IMAGES_PER_BOARD_PER_ENTITY,
    )
    parser.add_argument(
        "--eval-images-per-board-per-entity",
        type=int,
        default=EVAL_IMAGES_PER_BOARD_PER_ENTITY,
    )
    args = parser.parse_args(argv)
    result = export_single_piece_curriculum(
        args.dataset_dir,
        output_dir=args.output_dir,
        style_path=args.style_path,
        overwrite=args.overwrite,
        train_images_per_board_per_entity=args.train_images_per_board_per_entity,
        eval_images_per_board_per_entity=args.eval_images_per_board_per_entity,
        workers=args.workers,
        tile_rows=args.tile_rows,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
