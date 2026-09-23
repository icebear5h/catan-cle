"""Render sampled pair and single-piece boards, then assemble their rows."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence, cast

from data_pipeline.board_recognition import adjacent_pair_localization as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_int, as_list, as_str
from evals.catan_board_bench.render import RenderStyle


def build_pair_board(
    *,
    state: JsonDict,
    contract: JsonDict,
    output_images: Path,
    style: RenderStyle,
    images_per_kind: int | dict[str, int],
    colors: Sequence[str],
    pool: ProcessPoolExecutor | None = None,
    tile_rows: bool = False,
    novel_color: str | None = None,
    asset_root: Path | None = None,
    negatives: dict[str, int] | None = None,
) -> list[JsonDict]:
    """Render every sampled two-piece board for one empty state."""

    sample_id = as_str(state["sample_id"])
    image_size = as_int(as_list(state["image_size"])[0])
    regions = api.atlas_regions(contract, image_size=image_size, view_padding_factor=style.view_padding_factor)
    controls = api._control_regions(regions)
    neighbors = api.neighbor_tokens(contract)
    touching = api.cross_touching(contract)
    tokens = [token for token, region in regions.items() if region["entity_type"] in ("node", "edge")]
    counts = dict(api.DEFAULT_NEGATIVES if negatives is None else negatives)
    kind_counts = {kind: 0 for kind in api.PAIR_KINDS}
    kind_counts.update(images_per_kind if isinstance(images_per_kind, dict) else api.parse_kind_counts(images_per_kind, 0))
    tiles = api.tile_facts(contract) if tile_rows else []
    rows: list[JsonDict] = []
    for single_kind in api.SINGLE_KINDS:
        wanted = kind_counts[single_kind]
        if wanted <= 0:
            continue
        entity_type = single_kind.split("_")[1]
        entity_tokens = [token for token in tokens if api.entity_of(token) == entity_type]
        novel_count = round(wanted * api.NOVEL_COLOR_FRACTION) if novel_color else 0
        singles = api.sample_placements(
            sample_id=sample_id, entity_type=entity_type, tokens=entity_tokens, colors=colors, count=wanted - novel_count
        )
        if novel_count:
            singles += api.sample_placements(
                sample_id=sample_id + "_novel", entity_type=entity_type, tokens=entity_tokens, colors=(cast(str, novel_color),), count=novel_count
            )
        single_jobs = [
            (token, piece, color, output_images / f"{state['split']}_{state['sample_id']}_{single_kind}_{token[1:-1]}_{piece}_{color}.png")
            for token, piece, color in singles
        ]
        if pool is None:
            for token, piece, color, destination in single_jobs:
                api.render_contract(api.place_piece(contract, token, piece, color), image_size, style, destination, asset_root)
        else:
            futures = [
                pool.submit(api.render_contract, api.place_piece(contract, token, piece, color), image_size, style, destination, asset_root)
                for token, piece, color, destination in single_jobs
            ]
            for future in futures:
                future.result()
        for token, piece, color, destination in single_jobs:
            single_rows = api.rows_for_placement(
                state=state, regions=regions, controls=controls,
                token=token, piece=piece, color=color, image_name=destination.name,
                empty_tokens=api.sample_empty_tokens(
                    sample_id=sample_id, token=token, piece=piece, color=color, tokens=entity_tokens, neighbors=neighbors, counts=counts
                ),
                distances=api.neighbor_distances(neighbors, token),
            )
            for row in single_rows:
                row["pair_kind"] = single_kind
                row["partner_distance"] = "none"
            rows.extend(single_rows)
            if tile_rows:
                rows.extend(
                    api.tile_rows_for_image(state=state, tiles=tiles, regions=regions, controls=controls, image_name=destination.name, salt=destination.stem)
                )
    for pair_kind in api.TOUCHING_KINDS + api.FAR_KINDS:
        wanted = kind_counts[pair_kind]
        if wanted <= 0:
            continue
        pairs = api.location_pairs(contract, pair_kind)
        novel_count = round(wanted * api.NOVEL_COLOR_FRACTION) if novel_color else 0
        placements = api.sample_pairs(
            sample_id=sample_id, pair_kind=pair_kind,
            pairs=pairs, colors=colors, count=wanted - novel_count,
        )
        if novel_count:
            placements += api.sample_pairs(
                sample_id=sample_id + "_novel", pair_kind=pair_kind,
                pairs=pairs, colors=colors, count=novel_count, novel_color=novel_color,
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
                api.render_contract(api.place_pair(contract, first, second), image_size, style, destination, asset_root)
        else:
            futures = [
                pool.submit(api.render_contract, api.place_pair(contract, first, second), image_size, style, destination, asset_root)
                for first, second, destination in jobs
            ]
            for future in futures:
                future.result()
        for first, second, destination in jobs:
            empties = api.sample_pair_empty_tokens(
                sample_id=sample_id, first=first, second=second,
                tokens=tokens, neighbors=neighbors, touching=touching, counts=counts,
            )
            rows.extend(
                api.rows_for_pair(
                    state=state, regions=regions, controls=controls, first=first,
                    second=second, pair_kind=pair_kind, image_name=destination.name, empties=empties,
                )
            )
            if tile_rows:
                rows.extend(
                    api.tile_rows_for_image(
                        state=state, tiles=tiles, regions=regions, controls=controls,
                        image_name=destination.name, salt=destination.stem,
                    )
                )
    return rows
