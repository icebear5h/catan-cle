"""Build the adjacent-pair localization curriculum (``spatial_localization_pairs_v1``).

Each image is a validated empty replay board with exactly two real pieces on
touching locations. The single-piece stage let the model answer
``<N17> building?`` by describing the only piece it could see; on real boards
that shortcut names the neighbour's piece for an empty spot and says "empty"
for a piece next to another one. Two touching pieces make the occupancy
prompt unanswerable without deciding which spot holds which piece.

Pair kinds per board: ``node_node`` (the two endpoints of an edge; the game's
distance rule is ignored on purpose because touching buildings are the
hardest perception case), ``edge_edge`` (two edges sharing a node), and
``node_edge`` (a building and a road that touch). Colours always differ for
``node_node`` and ``edge_edge``; ``node_edge`` pairs share a colour half the
time because a road touching its own settlement is the normal game case.

The three ``*_far`` kinds are the control: the same two pieces placed more
than ``NEAR_MAX_HOPS`` apart, so a checkpoint that reads far pairs but not
touching pairs fails on neighbour discrimination, while one that fails both
has a multi-piece prior problem. They default to zero training images and
are meant for validation; set them per kind with
``--eval-images-per-board-per-kind node_node_far=10,...``.

``single_node`` and ``single_edge`` put one piece on the board with the
single-piece stage's rows, so a mixed file keeps the lone-piece anchor that
pure pair training erodes (the pairs_v1 run pushed lone-piece positives from
0.94 to 0.83 on the single-piece validation set). They default to zero.

Rows per image: ``occupancy_positive`` and ``colored_piece_to_token`` for each
piece, ``piece_to_token`` only when the type alone identifies one piece,
``occupancy_negative_adjacent`` for a third location touching either piece,
``occupancy_negative_far`` for a location beyond ``NEAR_MAX_HOPS`` of both,
plus the optional tile rows shared with the single-piece stage. Validation
and test put the novel probe colour on one piece of a fifth of the pairs and
drop only the rows whose answer names that colour.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations as combinations
from pathlib import Path
from typing import Sequence

from data_pipeline.json_coerce import as_dict, as_list, as_str
from data_pipeline.json_types import JsonDict, JsonValue

from ..replay_dataset import (
    DEFAULT_STYLE_PATH,
    load_render_style,
    read_jsonl,
    validate_replay_v1_dataset,
)
from ..single_piece_localization import COLORS as COLORS
from ..single_piece_localization import DEFAULT_NEGATIVES as DEFAULT_NEGATIVES
from ..single_piece_localization import EDGE_PIECE as EDGE_PIECE
from ..single_piece_localization import FORWARD_QUERY as FORWARD_QUERY
from ..single_piece_localization import LOCATION_NOUN as LOCATION_NOUN
from ..single_piece_localization import NEAR_MAX_HOPS as NEAR_MAX_HOPS
from ..single_piece_localization import NODE_PIECES as NODE_PIECES
from ..single_piece_localization import NOVEL_COLOR_FRACTION as NOVEL_COLOR_FRACTION
from ..single_piece_localization import (
    TILE_ROWS_PER_IMAGE,
    novel_hue,
    parse_negatives,
    write_novel_sprites,
)
from ..single_piece_localization import _row as _row
from ..single_piece_localization import color_words as color_words
from ..single_piece_localization import forward_answer as forward_answer
from ..single_piece_localization import is_novel_color as is_novel_color
from ..single_piece_localization import neighbor_distances as neighbor_distances
from ..single_piece_localization import neighbor_tokens as neighbor_tokens
from ..single_piece_localization import piece_words as piece_words
from ..single_piece_localization import place_piece as place_piece
from ..single_piece_localization import render_contract as render_contract
from ..single_piece_localization import rows_for_placement as rows_for_placement
from ..single_piece_localization import sample_empty_tokens as sample_empty_tokens
from ..single_piece_localization import sample_placements as sample_placements
from ..single_piece_localization import tile_facts as tile_facts
from ..single_piece_localization import tile_rows_for_image as tile_rows_for_image
from ..sources import file_sha256
from ..spatial_localization import SpatialLocalizationError as SpatialLocalizationError
from ..spatial_localization import _control_regions as _control_regions
from ..spatial_localization import (
    _deterministic_shuffle,
    _summarize_rows,
    _write_json,
    _write_jsonl,
)
from ..spatial_localization import _spatial_target as _spatial_target
from ..spatial_localization import _stable_rank as _stable_rank
from ..spatial_localization import atlas_regions as atlas_regions
from .boards import build_pair_board as build_pair_board
from .locations import _pick as _pick
from .locations import base_kind as base_kind
from .locations import cross_touching as cross_touching
from .locations import entity_of as entity_of
from .locations import far_location_pairs as far_location_pairs
from .locations import is_far_kind as is_far_kind
from .locations import location_pairs as location_pairs
from .locations import parse_kind_counts as parse_kind_counts
from .locations import place_pair as place_pair
from .locations import sample_pair_empty_tokens as sample_pair_empty_tokens
from .locations import sample_pairs as sample_pairs
from .rows import _piece_metadata as _piece_metadata
from .rows import rows_for_pair as rows_for_pair

Placement = tuple[str, str, str]  # (token, piece, color)
EXPORT_SCHEMA = "catan_adjacent_pair_localization/v1"
ROW_SCHEMA = "catan_adjacent_pair_localization_row/v1"
GROUNDING_STAGE = "adjacent_pair"
TASK_FAMILY = "adjacent_pair_localization"
DEFAULT_OUTPUT_NAME = "spatial_localization_pairs_v1"
TOUCHING_KINDS = ("node_node", "edge_edge", "node_edge")
FAR_KINDS = ("node_node_far", "edge_edge_far", "node_edge_far")
SINGLE_KINDS = ("single_node", "single_edge")
PAIR_KINDS = TOUCHING_KINDS + FAR_KINDS + SINGLE_KINDS
TRAIN_IMAGES_PER_BOARD_PER_KIND = 40
EVAL_IMAGES_PER_BOARD_PER_KIND = 30
SAME_COLOR_FRACTION = 0.5
NAMED_ROWS_PER_IMAGE = {"node_node": 6, "edge_edge": 4, "node_edge": 6}


def export_adjacent_pair_curriculum(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    style_path: str | Path = DEFAULT_STYLE_PATH,
    overwrite: bool = False,
    train_images_per_board_per_kind: int | str | dict[str, int] = TRAIN_IMAGES_PER_BOARD_PER_KIND,
    eval_images_per_board_per_kind: int | str | dict[str, int] = EVAL_IMAGES_PER_BOARD_PER_KIND,
    workers: int | None = None,
    tile_rows: bool = False,
    negatives: dict[str, int] | None = None,
) -> JsonDict:
    negative_counts = dict(DEFAULT_NEGATIVES if negatives is None else negatives)
    train_counts = (
        dict(train_images_per_board_per_kind)
        if isinstance(train_images_per_board_per_kind, dict)
        else parse_kind_counts(train_images_per_board_per_kind, TRAIN_IMAGES_PER_BOARD_PER_KIND)
    )
    eval_counts = (
        dict(eval_images_per_board_per_kind)
        if isinstance(eval_images_per_board_per_kind, dict)
        else parse_kind_counts(eval_images_per_board_per_kind, EVAL_IMAGES_PER_BOARD_PER_KIND)
    )
    dataset_root = Path(dataset_dir).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
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
    states_by_split = {split: [row for row in states if row["split"] == split] for split in ("train", "validation", "test")}
    if {split: len(rows) for split, rows in states_by_split.items()} != {"train": 43, "validation": 5, "test": 5}:
        raise SpatialLocalizationError("replay_v1 empty-board split counts changed")
    asset_root = output / "assets"
    novel_colors: dict[str, JsonValue] = {}
    novel_names: dict[str, str] = {}
    for split in ("validation", "test"):
        hue = novel_hue(f"{split}:{file_sha256(dataset_root / 'manifest.jsonl')}")
        name = f"NOVEL_{split.upper()}_H{hue:03d}"
        write_novel_sprites(asset_root, name, hue)
        novel_colors[split] = {"name": name, "hue_degrees": hue}
        novel_names[split] = name

    files: dict[str, JsonValue] = {}
    worker_count = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        for split, split_states in states_by_split.items():
            rows: list[JsonDict] = []
            for state in split_states:
                contract_path = dataset_root / as_str(state["contract_path"])
                if file_sha256(contract_path) != as_dict(state["sha256"])["contract"]:
                    raise SpatialLocalizationError(f"contract changed: {contract_path}")
                rows.extend(
                    build_pair_board(
                        state=state,
                        contract=json.loads(contract_path.read_text()),
                        output_images=images_dir,
                        style=style,
                        images_per_kind=train_counts if split == "train" else eval_counts,
                        colors=COLORS,
                        pool=pool,
                        tile_rows=tile_rows,
                        novel_color=novel_names.get(split),
                        asset_root=asset_root,
                        negatives=negative_counts,
                    )
                )
            if split == "train":
                rows = _deterministic_shuffle(rows, "adjacent_pair")
            path = output / "stage1" / f"{split}.jsonl"
            _write_jsonl(path, rows)
            summary = _summarize_rows(rows)
            for key in ("color", "piece", "pair_kind", "negative_distance"):
                dimensions = as_dict(summary["dimensions"])
                dimensions[key] = dict(sorted(Counter(str(row.get(key, "unknown")) for row in rows).items()))
            summary["unique_images"] = len({as_str(as_list(row["images"])[0]) for row in rows})
            files[f"stage1/{split}.jsonl"] = {**summary, "sha256": file_sha256(path)}

    metadata: JsonDict = {
        "schema": EXPORT_SCHEMA,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": file_sha256(dataset_root / "manifest.jsonl"),
        "style_sha256": file_sha256(Path(style_path)),
        "colors": list(COLORS),
        "novel_colors": novel_colors,
        "novel_color_fraction": NOVEL_COLOR_FRACTION,
        "asset_root": str(asset_root),
        "pair_kinds": list(PAIR_KINDS),
        "same_color_fraction_node_edge": SAME_COLOR_FRACTION,
        "train_images_per_board_per_kind": {key: value for key, value in train_counts.items()},
        "eval_images_per_board_per_kind": {key: value for key, value in eval_counts.items()},
        "negatives_per_image": {key: value for key, value in negative_counts.items()},
        "named_rows_per_image": {kind: count for kind, count in NAMED_ROWS_PER_IMAGE.items()},
        "tile_rows_per_image": TILE_ROWS_PER_IMAGE if tile_rows else 0,
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
    parser.add_argument("--tile-rows", action="store_true", help="Add tile resource, number, and inverse rows per image.")
    parser.add_argument(
        "--negatives",
        default=",".join(f"{kind}={count}" for kind, count in DEFAULT_NEGATIVES.items()),
        help="Empty negatives per image by kind, e.g. adjacent=1,far=1.",
    )
    parser.add_argument(
        "--train-images-per-board-per-kind",
        default=str(TRAIN_IMAGES_PER_BOARD_PER_KIND),
        help="One count for every kind, or per kind such as node_node=30,edge_edge=80,node_edge=50.",
    )
    parser.add_argument("--eval-images-per-board-per-kind", default=str(EVAL_IMAGES_PER_BOARD_PER_KIND))
    args = parser.parse_args(argv)
    result = export_adjacent_pair_curriculum(
        args.dataset_dir,
        output_dir=args.output_dir,
        style_path=args.style_path,
        overwrite=args.overwrite,
        train_images_per_board_per_kind=args.train_images_per_board_per_kind,
        eval_images_per_board_per_kind=args.eval_images_per_board_per_kind,
        workers=args.workers,
        tile_rows=args.tile_rows,
        negatives=parse_negatives(args.negatives),
    )
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
