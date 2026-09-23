"""Rendering one placement and assembling a whole board's images and rows."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from data_pipeline.board_recognition.single_piece_impl._config import (
    DEFAULT_NEGATIVES,
    NOVEL_COLOR_FRACTION,
)
from data_pipeline.board_recognition.single_piece_impl._facts import (
    tile_facts,
    tile_rows_for_image,
)
from data_pipeline.board_recognition.single_piece_impl._placements import (
    place_piece,
    sample_placements,
)
from data_pipeline.board_recognition.single_piece_impl._rows import rows_for_placement
from data_pipeline.board_recognition.single_piece_impl._topology import (
    neighbor_distances,
    neighbor_tokens,
    sample_empty_tokens,
)
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _control_regions,
    atlas_regions,
)
from data_pipeline.json_coerce import as_int, as_list, as_str
from data_pipeline.json_types import JsonDict
from evals.catan_board_bench import render as board_render
from evals.catan_board_bench.render import RenderStyle, render_contract_image


def render_contract(
    contract: JsonDict,
    image_size: int,
    style: RenderStyle,
    destination: Path,
    asset_root: Path | None = None,
) -> str:
    """Render an already-populated contract to ``destination`` (runs in a worker)."""

    if asset_root is not None:
        board_render.ASSET_ROOT = Path(asset_root)
    rendered = render_contract_image(contract, image_size=image_size, style=style)
    if rendered.size != (image_size, image_size):
        raise SpatialLocalizationError(f"renderer returned {rendered.size}")
    rendered.convert("RGB").save(destination)
    return destination.name


def render_placement(
    contract: JsonDict,
    token: str,
    piece: str,
    color: str,
    image_size: int,
    style: RenderStyle,
    destination: Path,
    asset_root: Path | None = None,
) -> str:
    """Render one single-piece board to ``destination`` (runs in a worker)."""

    return render_contract(place_piece(contract, token, piece, color), image_size, style, destination, asset_root)


def build_board(
    *,
    state: JsonDict,
    contract: JsonDict,
    output_images: Path,
    style: RenderStyle,
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

    sample_id = as_str(state["sample_id"])
    image_size = as_int(as_list(state["image_size"])[0])
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
            sample_id=sample_id,
            entity_type=entity_type,
            tokens=tokens,
            colors=colors,
            count=images_per_entity - novel_count,
        )
        if novel_count and novel_color is not None:
            placements += sample_placements(
                sample_id=sample_id + "_novel",
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
                sample_id=sample_id,
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
