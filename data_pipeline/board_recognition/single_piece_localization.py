"""Build the single-piece localization curriculum (``spatial_localization_v2``).

Each image is a validated empty replay board with exactly one real piece added
through the production renderer: a settlement or city on one node, or a road
on one edge, in one player color. Three rows name the piece and a
configurable set of "empty" negatives accompanies every image:

- ``piece_to_token``: "Which node has the settlement?" -> ``<N17>``
- ``colored_piece_to_token``: "Which node has the red settlement?" -> ``<N17>``
- ``occupancy_positive``: "<N17> building?" -> "red settlement"
- ``occupancy_negative_adjacent``: "<N16> building?" -> "empty" for a node
  within ``NEAR_MAX_HOPS`` edges of the piece (or an edge within that many
  nodes of the road); touching locations rank first, then each further ring
- ``occupancy_negative_far``: "<N40> building?" -> "empty" for a location
  beyond ``NEAR_MAX_HOPS`` from the piece

``DEFAULT_NEGATIVES`` sets how many of each kind an image carries; the
default 1 and 1 beside the six named and tile rows keeps "empty" at 25% of
the file. The adjacent negatives exist because a model that localizes
the piece to the right neighborhood but blames the wrong entity answers them
with the real piece; uniform empties almost never sample that case. Every
negative row records ``negative_distance`` (1, 2, or "far").

The forward rows use the exact production ``node.occupancy`` and
``edge.owner`` prompts, so this stage trains the heads that collapsed to
"empty" on dense boards, with a single piece and no clutter. All eleven player
colors are trained. Validation and test add a novel probe color that exists in
no sprite set: a seeded hue from the gaps between the real colors, applied by
hue-rotating the red sprites. Novel-color images carry only the rows whose
answer does not name the color (type-only localization, the empty negative,
and tile rows). Train rows are written in a deterministic shuffled order;
validation and test keep canonical order.
"""

from __future__ import annotations

import argparse
import colorsys
import copy
import json
import os
import re
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
from evals.catan_board_bench import render as board_render
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
NOVEL_COLOR_FRACTION = 0.2
# Hue intervals (degrees) with at least 25 degrees of clearance from every
# saturated stop in the shipped piece sprites.
NOVEL_HUE_INTERVALS = ((70, 105), (145, 190), (250, 275), (295, 325))
NOVEL_SPRITE_BASE = "red"
SPRITE_PIECES = ("settlement", "city", "road")
HEX_COLOR_RE = re.compile(r"#([0-9a-fA-F]{6})")
NODE_PIECES = ("SETTLEMENT", "CITY")
EDGE_PIECE = "ROAD"
TRAIN_IMAGES_PER_BOARD_PER_ENTITY = 70
EVAL_IMAGES_PER_BOARD_PER_ENTITY = 30
FORWARD_QUERY = {"node": "building?", "edge": "road?"}
LOCATION_NOUN = {"node": "node", "edge": "edge"}
TILE_ROWS_PER_IMAGE = 3
NAMED_ROWS_PER_IMAGE = 3
NEGATIVE_KINDS = ("adjacent", "far")
DEFAULT_NEGATIVES = {"adjacent": 1, "far": 1}
# Coastal nodes have only five same-type locations within two hops; three
# hops gives every placement at least seven near candidates when a heavier
# mix is requested. The default single near negative always touches the piece.
NEAR_MAX_HOPS = 3


def parse_negatives(spec: str) -> dict[str, int]:
    """Parse ``adjacent=1,far=1`` into per-kind counts."""

    counts = dict(DEFAULT_NEGATIVES)
    for part in filter(None, (piece.strip() for piece in spec.split(","))):
        kind, _, value = part.partition("=")
        if kind not in NEGATIVE_KINDS or not value.isdigit():
            raise SpatialLocalizationError(f"bad negative spec {part!r}; kinds are {NEGATIVE_KINDS}")
        counts[kind] = int(value)
    return counts


def neighbor_tokens(contract: JsonDict) -> dict[str, list[str]]:
    """Map every node and edge token to its touching same-type locations.

    Nodes neighbor the nodes they share an edge with; edges neighbor the edges
    they share a node with. Both come straight from the contract graph.
    """

    nodes_of_edge = {edge["token"]: list(edge["node_tokens"]) for edge in contract["edges"]}
    edges_of_node = {node["token"]: list(node["adjacent_edge_tokens"]) for node in contract["nodes"]}
    neighbors: dict[str, list[str]] = {}
    for node_token, edge_tokens in edges_of_node.items():
        neighbors[node_token] = sorted(
            {other for edge_token in edge_tokens for other in nodes_of_edge[edge_token] if other != node_token}
        )
    for edge_token, endpoints in nodes_of_edge.items():
        neighbors[edge_token] = sorted(
            {other for node_token in endpoints for other in edges_of_node[node_token] if other != edge_token}
        )
    return neighbors


def neighbor_distances(neighbors: dict[str, list[str]], token: str, max_hops: int = NEAR_MAX_HOPS) -> dict[str, int]:
    """Hop distance from ``token`` to every same-type location within ``max_hops``."""

    distances = {token: 0}
    frontier = [token]
    for hops in range(1, max_hops + 1):
        frontier = [other for current in frontier for other in neighbors[current] if other not in distances]
        for other in frontier:
            distances.setdefault(other, hops)
    return distances


def sample_empty_tokens(
    *,
    sample_id: str,
    token: str,
    piece: str,
    color: str,
    tokens: Sequence[str],
    neighbors: dict[str, list[str]],
    counts: dict[str, int],
) -> dict[str, list[str]]:
    """Pick the empty locations queried for one placement, per negative kind.

    ``adjacent`` draws touching locations first, then each further ring out
    to ``NEAR_MAX_HOPS``, each in a stable hashed order; ``far`` draws from
    everything beyond that.
    Nothing repeats within an image, and a count larger than its pool is
    capped by the pool.
    """

    distances = neighbor_distances(neighbors, token)

    def ranked(kind: str, candidates: Sequence[str]) -> list[str]:
        return sorted(
            candidates,
            key=lambda item: (
                distances.get(item, 0),
                _stable_rank(sample_id, token, piece, color, f"empty_{kind}", item),
                item,
            ),
        )

    pools = {
        "adjacent": ranked("adjacent", [candidate for candidate in tokens if distances.get(candidate, 0) > 0]),
        "far": ranked("far", [candidate for candidate in tokens if candidate not in distances]),
    }
    chosen: dict[str, list[str]] = {}
    for kind in NEGATIVE_KINDS:
        if counts.get(kind, 0) and not pools[kind]:
            raise SpatialLocalizationError(f"{token} has no {kind} empty candidates")
        chosen[kind] = pools[kind][: counts.get(kind, 0)]
    return chosen


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


def is_novel_color(color: str) -> bool:
    return color.startswith("NOVEL_")


def novel_hue(seed: str) -> int:
    """Pick a probe hue deterministically from the gaps between real colors."""

    span = sum(high - low for low, high in NOVEL_HUE_INTERVALS)
    offset = _stable_rank(seed, "novel_hue") % span
    for low, high in NOVEL_HUE_INTERVALS:
        if offset < high - low:
            return low + offset
        offset -= high - low
    raise AssertionError("unreachable")


def recolor_svg(svg: str, hue_degrees: int, *, min_saturation: float = 0.25) -> str:
    """Rotate every saturated hex stop onto ``hue_degrees``, keeping S and V."""

    def swap(match: re.Match[str]) -> str:
        value = match.group(1)
        r, g, b = (int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))
        _, sat, val = colorsys.rgb_to_hsv(r, g, b)
        if sat < min_saturation:
            return match.group(0)
        nr, ng, nb = colorsys.hsv_to_rgb(hue_degrees / 360, sat, val)
        return "#%02X%02X%02X" % (round(nr * 255), round(ng * 255), round(nb * 255))

    return HEX_COLOR_RE.sub(swap, svg)


def write_novel_sprites(asset_root: Path, name: str, hue_degrees: int) -> list[Path]:
    """Create ``<piece>_<name>.svg`` sprites in a private copy of the asset tree."""

    if not asset_root.exists():
        shutil.copytree(board_render.ASSET_ROOT, asset_root)
    written = []
    for piece in SPRITE_PIECES:
        source = asset_root / "pieces" / f"{piece}_{NOVEL_SPRITE_BASE}.svg"
        target = asset_root / "pieces" / f"{piece}_{name.lower()}.svg"
        target.write_text(recolor_svg(source.read_text(), hue_degrees))
        written.append(target)
    return written


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
    empty_tokens: dict[str, list[str]],
    distances: dict[str, int] | None = None,
) -> list[JsonDict]:
    entity_type = regions[token]["entity_type"]
    distances = distances or {}
    noun = LOCATION_NOUN[entity_type]
    novel = is_novel_color(color)
    metadata = {
        "split": state["split"],
        "state_id": state["sample_id"],
        "entity_type": entity_type,
        "target_token": token,
        "piece": piece,
        "color": color_words(color),
        "color_heldout": novel,
    }
    target = _spatial_target(regions[token], controls[token])
    stem = f"{state['sample_id']}_{token[1:-1]}_{piece}_{color}"
    category = "node.occupancy" if entity_type == "node" else "edge.owner"
    named_rows = [] if novel else [
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
    ]
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
        *named_rows,
        *(
            _row(
                row_id=f"{stem}_occupancy_negative_{kind}_{index}",
                image_name=image_name,
                prompt=f"{empty_token} {FORWARD_QUERY[entity_type]}",
                answer="empty",
                task_type=f"occupancy_negative_{kind}",
                category=category,
                polarity="hard_negative",
                metadata={
                    **metadata,
                    "queried_token": empty_token,
                    "negative_kind": kind,
                    "negative_distance": distances.get(empty_token, "far"),
                },
                spatial_target=None,
            )
            for kind in NEGATIVE_KINDS
            for index, empty_token in enumerate(empty_tokens.get(kind, ()))
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
    asset_root: Path | None = None,
) -> str:
    """Render one single-piece board to ``destination`` (runs in a worker)."""

    if asset_root is not None:
        board_render.ASSET_ROOT = Path(asset_root)
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
    novel_color: str | None = None,
    asset_root: Path | None = None,
    negatives: dict[str, int] | None = None,
) -> list[JsonDict]:
    """Render every sampled single-piece board for one empty state.

    ``novel_color`` adds ``NOVEL_COLOR_FRACTION`` of the placements in a probe
    color whose sprites live under ``asset_root``.
    """

    image_size = int(state["image_size"][0])
    regions = atlas_regions(
        contract,
        image_size=image_size,
        view_padding_factor=style.view_padding_factor,
    )
    controls = _control_regions(regions)
    neighbors = neighbor_tokens(contract)
    negative_counts = dict(DEFAULT_NEGATIVES if negatives is None else negatives)
    tiles = tile_facts(contract) if tile_rows else []
    rows: list[JsonDict] = []
    for entity_type in ("node", "edge"):
        tokens = [token for token, region in regions.items() if region["entity_type"] == entity_type]
        novel_count = round(images_per_entity * NOVEL_COLOR_FRACTION) if novel_color else 0
        placements = sample_placements(
            sample_id=state["sample_id"],
            entity_type=entity_type,
            tokens=tokens,
            colors=colors,
            count=images_per_entity - novel_count,
        )
        if novel_count:
            placements += sample_placements(
                sample_id=state["sample_id"] + "_novel",
                entity_type=entity_type,
                tokens=tokens,
                colors=(novel_color,),
                count=novel_count,
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
                render_placement(contract, token, piece, color, image_size, style, destination, asset_root)
        else:
            futures = [
                pool.submit(
                    render_placement, contract, token, piece, color, image_size, style, destination, asset_root
                )
                for token, piece, color, destination in jobs
            ]
            for future in futures:
                future.result()
        for token, piece, color, destination in jobs:
            image_name = destination.name
            empty_tokens = sample_empty_tokens(
                sample_id=state["sample_id"],
                token=token,
                piece=piece,
                color=color,
                tokens=tokens,
                neighbors=neighbors,
                counts=negative_counts,
            )
            rows.extend(
                rows_for_placement(
                    state=state,
                    regions=regions,
                    controls=controls,
                    token=token,
                    piece=piece,
                    color=color,
                    image_name=image_name,
                    empty_tokens=empty_tokens,
                    distances=neighbor_distances(neighbors, token),
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
    negatives: dict[str, int] | None = None,
) -> JsonDict:
    negative_counts = dict(DEFAULT_NEGATIVES if negatives is None else negatives)
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
                        colors=COLORS,
                        pool=pool,
                        tile_rows=tile_rows,
                        novel_color=novel_colors[split]["name"] if split in novel_colors else None,
                        asset_root=asset_root,
                        negatives=negative_counts,
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
        "novel_colors": novel_colors,
        "novel_color_fraction": NOVEL_COLOR_FRACTION,
        "novel_sprite_base": NOVEL_SPRITE_BASE,
        "asset_root": str(asset_root),
        "node_pieces": list(NODE_PIECES),
        "edge_piece": EDGE_PIECE,
        "train_images_per_board_per_entity": train_images_per_board_per_entity,
        "eval_images_per_board_per_entity": eval_images_per_board_per_entity,
        "negatives_per_image": negative_counts,
        "rows_per_image": (
            NAMED_ROWS_PER_IMAGE + sum(negative_counts.values()) + (TILE_ROWS_PER_IMAGE if tile_rows else 0)
        ),
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
        "--negatives",
        default=",".join(f"{kind}={count}" for kind, count in DEFAULT_NEGATIVES.items()),
        help="Empty negatives per image by kind, e.g. adjacent=1,far=1.",
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
        negatives=parse_negatives(args.negatives),
    )
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
