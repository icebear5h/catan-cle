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
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from data_pipeline.board_recognition.replay_dataset import (
    DEFAULT_STYLE_PATH,
    file_sha256,
    load_render_style,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.single_piece_localization import (
    COLORS,
    DEFAULT_NEGATIVES,
    EDGE_PIECE,
    FORWARD_QUERY,
    LOCATION_NOUN,
    NEAR_MAX_HOPS,
    NODE_PIECES,
    NOVEL_COLOR_FRACTION,
    TILE_ROWS_PER_IMAGE,
    _row,
    color_words,
    forward_answer,
    is_novel_color,
    neighbor_distances,
    neighbor_tokens,
    novel_hue,
    parse_negatives,
    piece_words,
    place_piece,
    render_contract,
    rows_for_placement,
    sample_empty_tokens,
    sample_placements,
    tile_facts,
    tile_rows_for_image,
    write_novel_sprites,
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


JsonDict = dict[str, Any]
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


def base_kind(pair_kind: str) -> str:
    return pair_kind[: -len("_far")] if pair_kind.endswith("_far") else pair_kind


def is_far_kind(pair_kind: str) -> bool:
    return pair_kind.endswith("_far")


def parse_kind_counts(spec: str | int, default: int) -> dict[str, int]:
    """Images per board per pair kind from ``40`` or ``node_node=30,edge_edge=80``.

    A bare number applies to the touching kinds and leaves the far control and
    single kinds at zero. A per-kind list is explicit: every kind it does not
    name is zero, so ``node_node_far=10`` alone builds a far-only set.
    """

    if isinstance(spec, int) or str(spec).isdigit():
        return {kind: (int(spec) if kind in TOUCHING_KINDS else 0) for kind in PAIR_KINDS}
    counts = {kind: 0 for kind in PAIR_KINDS}
    for part in filter(None, (piece.strip() for piece in str(spec).split(","))):
        kind, _, value = part.partition("=")
        if kind not in PAIR_KINDS or not value.isdigit():
            raise SpatialLocalizationError(f"bad pair-kind count {part!r}; kinds are {PAIR_KINDS}")
        counts[kind] = int(value)
    return counts


def entity_of(token: str) -> str:
    return "node" if token.startswith("<N") else "edge"


def cross_touching(contract: JsonDict) -> dict[str, list[str]]:
    """Map every node to the edges at it and every edge to its endpoint nodes."""

    touching = {node["token"]: sorted(node["adjacent_edge_tokens"]) for node in contract["nodes"]}
    touching.update({edge["token"]: sorted(edge["node_tokens"]) for edge in contract["edges"]})
    return touching


def far_location_pairs(contract: JsonDict, pair_kind: str) -> list[tuple[str, str]]:
    """Every pair of the kind's entity types more than ``NEAR_MAX_HOPS`` apart."""

    neighbors = neighbor_tokens(contract)
    touching = cross_touching(contract)
    nodes = [node["token"] for node in contract["nodes"]]
    edges = [edge["token"] for edge in contract["edges"]]
    if pair_kind == "node_node_far":
        candidates = combinations(nodes, 2)
    elif pair_kind == "edge_edge_far":
        candidates = combinations(edges, 2)
    elif pair_kind == "node_edge_far":
        candidates = ((node, edge) for node in nodes for edge in edges)
    else:
        raise SpatialLocalizationError(f"unknown pair kind: {pair_kind}")
    pairs = []
    for first, second in candidates:
        if entity_of(first) == entity_of(second):
            if second in neighbor_distances(neighbors, first, NEAR_MAX_HOPS):
                continue
        else:
            near = neighbor_distances(neighbors, first, NEAR_MAX_HOPS)
            if any(endpoint in near for endpoint in touching[second]):
                continue
        pairs.append((first, second))
    return sorted(pairs)


def location_pairs(contract: JsonDict, pair_kind: str) -> list[tuple[str, str]]:
    """Every location pair of one kind, in a stable order."""

    if is_far_kind(pair_kind):
        return far_location_pairs(contract, pair_kind)
    if pair_kind == "node_node":
        pairs = {tuple(sorted(edge["node_tokens"])) for edge in contract["edges"]}
    elif pair_kind == "edge_edge":
        pairs = {
            tuple(sorted(pair))
            for node in contract["nodes"]
            for pair in combinations(node["adjacent_edge_tokens"], 2)
        }
    elif pair_kind == "node_edge":
        pairs = {(node["token"], edge) for node in contract["nodes"] for edge in node["adjacent_edge_tokens"]}
    else:
        raise SpatialLocalizationError(f"unknown pair kind: {pair_kind}")
    return sorted(pairs)


def _pick(sample_id: str, salt: str, options: Sequence[str]) -> str:
    return options[_stable_rank(sample_id, salt) % len(options)]


def sample_pairs(
    *,
    sample_id: str,
    pair_kind: str,
    pairs: Sequence[tuple[str, str]],
    colors: Sequence[str],
    count: int,
    novel_color: str | None = None,
) -> list[tuple[Placement, Placement]]:
    """Deterministically pick ``count`` location pairs and dress them with pieces.

    With ``novel_color`` exactly one piece of every pair wears the probe colour
    and the pair never shares a colour.
    """

    if count > len(pairs):
        raise SpatialLocalizationError(f"requested {count} {pair_kind} pairs from {len(pairs)} locations")
    ranked = sorted(pairs, key=lambda pair: (_stable_rank(sample_id, pair_kind, *pair, "pair"), pair))
    placements: list[tuple[Placement, Placement]] = []
    for first, second in ranked[:count]:
        salt = f"{pair_kind}:{first}:{second}"
        piece_a = _pick(sample_id, salt + ":piece_a", NODE_PIECES) if entity_of(first) == "node" else EDGE_PIECE
        piece_b = _pick(sample_id, salt + ":piece_b", NODE_PIECES) if entity_of(second) == "node" else EDGE_PIECE
        color_a = _pick(sample_id, salt + ":color_a", colors)
        others = [color for color in colors if color != color_a]
        same = (
            base_kind(pair_kind) == "node_edge"
            and novel_color is None
            and _stable_rank(sample_id, salt, "same_color") % 100 < SAME_COLOR_FRACTION * 100
        )
        color_b = color_a if same else _pick(sample_id, salt + ":color_b", others)
        if novel_color is not None:
            if _stable_rank(sample_id, salt, "novel_side") % 2:
                color_a = novel_color
            else:
                color_b = novel_color
        placements.append(((first, piece_a, color_a), (second, piece_b, color_b)))
    return placements


def place_pair(contract: JsonDict, first: Placement, second: Placement) -> JsonDict:
    return place_piece(place_piece(contract, *first), *second)


def sample_pair_empty_tokens(
    *,
    sample_id: str,
    first: Placement,
    second: Placement,
    tokens: Sequence[str],
    neighbors: dict[str, list[str]],
    touching: dict[str, list[str]],
    counts: dict[str, int],
) -> list[JsonDict]:
    """Pick the empty locations queried for one pair.

    ``adjacent`` empties touch either piece: a same-type hop-1 neighbour or the
    other type's touching location. ``far`` empties are beyond ``NEAR_MAX_HOPS``
    of the same-type piece and do not touch the other piece. Each entry records
    the anchor piece it was measured against.
    """

    anchors = {first[0]: first, second[0]: second}
    adjacent: dict[str, JsonDict] = {}
    for anchor in anchors:
        for token in neighbors[anchor] + touching[anchor]:
            if token not in anchors and token not in adjacent:
                adjacent[token] = {"token": token, "anchor": anchor, "negative_distance": 1}
    far: dict[str, JsonDict] = {}
    for token in tokens:
        if token in anchors or token in adjacent:
            continue
        same_type = [anchor for anchor in anchors if entity_of(anchor) == entity_of(token)]
        if same_type and any(token in neighbor_distances(neighbors, anchor, NEAR_MAX_HOPS) for anchor in same_type):
            continue
        if any(token in touching[anchor] for anchor in anchors):
            continue
        far[token] = {"token": token, "anchor": same_type[0] if same_type else first[0], "negative_distance": "far"}
    chosen: list[JsonDict] = []
    for kind, pool in (("adjacent", adjacent), ("far", far)):
        wanted = counts.get(kind, 0)
        if wanted and not pool:
            raise SpatialLocalizationError(f"{first[0]}/{second[0]} has no {kind} empty candidates")
        ranked = sorted(pool, key=lambda token: (_stable_rank(sample_id, first[0], second[0], f"empty_{kind}", token), token))
        chosen.extend({**pool[token], "kind": kind} for token in ranked[:wanted])
    return chosen


def _piece_metadata(state: JsonDict, piece: Placement, partner: Placement, pair_kind: str) -> JsonDict:
    token, kind, color = piece
    partner_token, partner_kind, partner_color = partner
    return {
        "split": state["split"],
        "state_id": state["sample_id"],
        "entity_type": entity_of(token),
        "target_token": token,
        "piece": kind,
        "color": color_words(color),
        "color_heldout": is_novel_color(color),
        "pair_kind": pair_kind,
        "partner_token": partner_token,
        "partner_piece": partner_kind,
        "partner_color": color_words(partner_color),
        "partner_distance": "far" if is_far_kind(pair_kind) else 1,
        "same_color": color == partner_color,
    }


def rows_for_pair(
    *,
    state: JsonDict,
    regions: dict[str, JsonDict],
    controls: dict[str, JsonDict],
    first: Placement,
    second: Placement,
    pair_kind: str,
    image_name: str,
    empties: Sequence[JsonDict],
) -> list[JsonDict]:
    stem = f"{state['sample_id']}_{pair_kind}_{first[0][1:-1]}_{first[1]}_{first[2]}_{second[0][1:-1]}_{second[1]}_{second[2]}"
    rows: list[JsonDict] = []
    for piece, partner in ((first, second), (second, first)):
        token, kind, color = piece
        entity_type = entity_of(token)
        noun = LOCATION_NOUN[entity_type]
        category = "node.occupancy" if entity_type == "node" else "edge.owner"
        metadata = _piece_metadata(state, piece, partner, pair_kind)
        target = _spatial_target(regions[token], controls[token])
        piece_stem = f"{stem}_{token[1:-1]}"
        common = {
            "image_name": image_name,
            "metadata": metadata,
            "spatial_target": target,
            "schema": ROW_SCHEMA,
            "grounding_stage": GROUNDING_STAGE,
            "task_family": TASK_FAMILY,
        }
        if not is_novel_color(color):
            rows.append(
                _row(
                    row_id=f"{piece_stem}_occupancy_positive",
                    prompt=f"{token} {FORWARD_QUERY[entity_type]}",
                    answer=forward_answer(color, kind),
                    task_type="occupancy_positive",
                    category=category,
                    polarity="positive",
                    **common,
                )
            )
            rows.append(
                _row(
                    row_id=f"{piece_stem}_colored_piece_to_token",
                    prompt=f"Which {noun} has the {forward_answer(color, kind)}?",
                    answer=token,
                    task_type="colored_piece_to_token",
                    category="localization",
                    polarity="token_return",
                    **common,
                )
            )
        if kind != partner[1]:
            rows.append(
                _row(
                    row_id=f"{piece_stem}_piece_to_token",
                    prompt=f"Which {noun} has the {piece_words(kind)}?",
                    answer=token,
                    task_type="piece_to_token",
                    category="localization",
                    polarity="token_return",
                    **common,
                )
            )
    for index, empty in enumerate(empties):
        token = empty["token"]
        entity_type = entity_of(token)
        anchor = first if empty["anchor"] == first[0] else second
        partner = second if anchor is first else first
        rows.append(
            _row(
                row_id=f"{stem}_occupancy_negative_{empty['kind']}_{index}",
                image_name=image_name,
                prompt=f"{token} {FORWARD_QUERY[entity_type]}",
                answer="empty",
                task_type=f"occupancy_negative_{empty['kind']}",
                category="node.occupancy" if entity_type == "node" else "edge.owner",
                polarity="hard_negative",
                metadata={
                    **_piece_metadata(state, anchor, partner, pair_kind),
                    "queried_token": token,
                    "negative_kind": empty["kind"],
                    "negative_distance": empty["negative_distance"],
                },
                spatial_target=None,
                schema=ROW_SCHEMA,
                grounding_stage=GROUNDING_STAGE,
                task_family=TASK_FAMILY,
            )
        )
    return rows


def build_pair_board(
    *,
    state: JsonDict,
    contract: JsonDict,
    output_images: Path,
    style: Any,
    images_per_kind: int | dict[str, int],
    colors: Sequence[str],
    pool: ProcessPoolExecutor | None = None,
    tile_rows: bool = False,
    novel_color: str | None = None,
    asset_root: Path | None = None,
    negatives: dict[str, int] | None = None,
) -> list[JsonDict]:
    """Render every sampled two-piece board for one empty state."""

    image_size = int(state["image_size"][0])
    regions = atlas_regions(contract, image_size=image_size, view_padding_factor=style.view_padding_factor)
    controls = _control_regions(regions)
    neighbors = neighbor_tokens(contract)
    touching = cross_touching(contract)
    tokens = [token for token, region in regions.items() if region["entity_type"] in ("node", "edge")]
    counts = dict(DEFAULT_NEGATIVES if negatives is None else negatives)
    kind_counts = {kind: 0 for kind in PAIR_KINDS}
    kind_counts.update(images_per_kind if isinstance(images_per_kind, dict) else parse_kind_counts(images_per_kind, 0))
    tiles = tile_facts(contract) if tile_rows else []
    rows: list[JsonDict] = []
    for single_kind in SINGLE_KINDS:
        wanted = kind_counts[single_kind]
        if wanted <= 0:
            continue
        entity_type = single_kind.split("_")[1]
        entity_tokens = [token for token in tokens if entity_of(token) == entity_type]
        novel_count = round(wanted * NOVEL_COLOR_FRACTION) if novel_color else 0
        singles = sample_placements(
            sample_id=state["sample_id"], entity_type=entity_type, tokens=entity_tokens, colors=colors, count=wanted - novel_count
        )
        if novel_count:
            singles += sample_placements(
                sample_id=state["sample_id"] + "_novel", entity_type=entity_type, tokens=entity_tokens, colors=(novel_color,), count=novel_count
            )
        single_jobs = [
            (token, piece, color, output_images / f"{state['split']}_{state['sample_id']}_{single_kind}_{token[1:-1]}_{piece}_{color}.png")
            for token, piece, color in singles
        ]
        if pool is None:
            for token, piece, color, destination in single_jobs:
                render_contract(place_piece(contract, token, piece, color), image_size, style, destination, asset_root)
        else:
            futures = [
                pool.submit(render_contract, place_piece(contract, token, piece, color), image_size, style, destination, asset_root)
                for token, piece, color, destination in single_jobs
            ]
            for future in futures:
                future.result()
        for token, piece, color, destination in single_jobs:
            single_rows = rows_for_placement(
                state=state,
                regions=regions,
                controls=controls,
                token=token,
                piece=piece,
                color=color,
                image_name=destination.name,
                empty_tokens=sample_empty_tokens(
                    sample_id=state["sample_id"], token=token, piece=piece, color=color, tokens=entity_tokens, neighbors=neighbors, counts=counts
                ),
                distances=neighbor_distances(neighbors, token),
            )
            for row in single_rows:
                row["pair_kind"] = single_kind
                row["partner_distance"] = "none"
            rows.extend(single_rows)
            if tile_rows:
                rows.extend(
                    tile_rows_for_image(state=state, tiles=tiles, regions=regions, controls=controls, image_name=destination.name, salt=destination.stem)
                )
    for pair_kind in TOUCHING_KINDS + FAR_KINDS:
        wanted = kind_counts[pair_kind]
        if wanted <= 0:
            continue
        pairs = location_pairs(contract, pair_kind)
        novel_count = round(wanted * NOVEL_COLOR_FRACTION) if novel_color else 0
        placements = sample_pairs(
            sample_id=state["sample_id"],
            pair_kind=pair_kind,
            pairs=pairs,
            colors=colors,
            count=wanted - novel_count,
        )
        if novel_count:
            placements += sample_pairs(
                sample_id=state["sample_id"] + "_novel",
                pair_kind=pair_kind,
                pairs=pairs,
                colors=colors,
                count=novel_count,
                novel_color=novel_color,
            )
        jobs = []
        for first, second in placements:
            name = (
                f"{state['split']}_{state['sample_id']}_{pair_kind}_"
                f"{first[0][1:-1]}_{first[1]}_{first[2]}_{second[0][1:-1]}_{second[1]}_{second[2]}.png"
            )
            jobs.append((first, second, output_images / name))
        if pool is None:
            for first, second, destination in jobs:
                render_contract(place_pair(contract, first, second), image_size, style, destination, asset_root)
        else:
            futures = [
                pool.submit(render_contract, place_pair(contract, first, second), image_size, style, destination, asset_root)
                for first, second, destination in jobs
            ]
            for future in futures:
                future.result()
        for first, second, destination in jobs:
            empties = sample_pair_empty_tokens(
                sample_id=state["sample_id"],
                first=first,
                second=second,
                tokens=tokens,
                neighbors=neighbors,
                touching=touching,
                counts=counts,
            )
            rows.extend(
                rows_for_pair(
                    state=state,
                    regions=regions,
                    controls=controls,
                    first=first,
                    second=second,
                    pair_kind=pair_kind,
                    image_name=destination.name,
                    empties=empties,
                )
            )
            if tile_rows:
                rows.extend(
                    tile_rows_for_image(
                        state=state,
                        tiles=tiles,
                        regions=regions,
                        controls=controls,
                        image_name=destination.name,
                        salt=destination.stem,
                    )
                )
    return rows


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
    novel_colors = {}
    for split in ("validation", "test"):
        hue = novel_hue(f"{split}:{file_sha256(dataset_root / 'manifest.jsonl')}")
        name = f"NOVEL_{split.upper()}_H{hue:03d}"
        write_novel_sprites(asset_root, name, hue)
        novel_colors[split] = {"name": name, "hue_degrees": hue}

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
                    build_pair_board(
                        state=state,
                        contract=json.loads(contract_path.read_text()),
                        output_images=images_dir,
                        style=style,
                        images_per_kind=train_counts if split == "train" else eval_counts,
                        colors=COLORS,
                        pool=pool,
                        tile_rows=tile_rows,
                        novel_color=novel_colors[split]["name"] if split in novel_colors else None,
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
                summary["dimensions"][key] = dict(sorted(Counter(str(row.get(key, "unknown")) for row in rows).items()))
            summary["unique_images"] = len({row["images"][0] for row in rows})
            files[f"stage1/{split}.jsonl"] = {**summary, "sha256": file_sha256(path)}

    metadata = {
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
        "train_images_per_board_per_kind": train_counts,
        "eval_images_per_board_per_kind": eval_counts,
        "negatives_per_image": negative_counts,
        "named_rows_per_image": NAMED_ROWS_PER_IMAGE,
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
