"""Water/coast framing and land-tile layers of the board raster."""

from __future__ import annotations

from PIL import Image

from evals.catan_board_bench.annotations import (
    HEX_SIZE,
    NUMBER_TOKEN_SIZE,
    NUMBER_TOKEN_Y_OFFSET,
    TILE_HEIGHT,
    TILE_WIDTH,
    RenderState,
    _hex_to_pixel,
    _view_box,
)
from evals.catan_board_bench.render.raster import (
    _asset_path,
    _paste_asset,
    _paste_rotated_asset,
)
from evals.catan_board_bench.render.style import (
    COAST_CORNER_ASSET,
    COAST_EDGE_ASSET,
    COAST_PIECE_SIZE,
    CUBE_DIRS,
    RenderStyle,
)


def _view_box_with_padding(render_state: RenderState, padding_factor: float) -> dict[str, float]:
    positions = [_hex_to_pixel(placed["coordinate"]) for placed in render_state.get("tiles", [])]
    if not positions:
        return _view_box(render_state)
    padding = HEX_SIZE * max(0.0, padding_factor)
    min_x = min(point["x"] for point in positions) - padding
    max_x = max(point["x"] for point in positions) + padding
    min_y = min(point["y"] for point in positions) - padding
    max_y = max(point["y"] for point in positions) + padding
    return {"min_x": min_x, "min_y": min_y, "width": max_x - min_x, "height": max_y - min_y}


def _render_coast(
    canvas: Image.Image,
    render_state: RenderState,
    transform: dict[str, float],
    style: RenderStyle,
) -> None:
    land_coords = {
        tuple(placed["coordinate"])
        for placed in render_state.get("tiles", [])
        if placed["tile"]["type"] in {"RESOURCE_TILE", "DESERT"}
    }
    explicit_water_coords = {
        tuple(placed["coordinate"])
        for placed in render_state.get("tiles", [])
        if placed["tile"]["type"] not in {"RESOURCE_TILE", "DESERT"}
    }
    inferred_water_coords = {
        (cx + dx, cy + dy, cz + dz)
        for cx, cy, cz in land_coords
        for dx, dy, dz in CUBE_DIRS
        if (cx + dx, cy + dy, cz + dz) not in land_coords
    }

    for cx, cy, cz in sorted(explicit_water_coords | inferred_water_coords):
        center = _hex_to_pixel((cx, cy, cz))
        land_dirs = [
            index
            for index, (dx, dy, dz) in enumerate(CUBE_DIRS)
            if (cx + dx, cy + dy, cz + dz) in land_coords
        ]
        if not land_dirs:
            continue

        if len(land_dirs) == 2:
            d1, d2 = land_dirs
            diff = ((d2 - d1) + 6) % 6
            if diff == 1:
                coast_size = COAST_PIECE_SIZE * style.coast_size_factor
                _paste_rotated_asset(
                    canvas,
                    _asset_path(COAST_CORNER_ASSET),
                    center_svg=(center["x"], center["y"]),
                    width_svg=coast_size,
                    height_svg=coast_size,
                    angle_deg=d1 * 60,
                    transform=transform,
                )
                continue
            if diff == 5:
                coast_size = COAST_PIECE_SIZE * style.coast_size_factor
                _paste_rotated_asset(
                    canvas,
                    _asset_path(COAST_CORNER_ASSET),
                    center_svg=(center["x"], center["y"]),
                    width_svg=coast_size,
                    height_svg=coast_size,
                    angle_deg=d2 * 60,
                    transform=transform,
                )
                continue

        for direction in land_dirs:
            coast_size = COAST_PIECE_SIZE * style.coast_size_factor
            _paste_rotated_asset(
                canvas,
                _asset_path(COAST_EDGE_ASSET),
                center_svg=(center["x"], center["y"]),
                width_svg=coast_size,
                height_svg=coast_size,
                angle_deg=direction * 60,
                transform=transform,
            )


def _render_land_tiles(
    canvas: Image.Image,
    render_state: RenderState,
    transform: dict[str, float],
    style: RenderStyle,
) -> None:
    for placed in render_state.get("tiles", []):
        tile = placed["tile"]
        if tile["type"] == "PORT":
            continue

        center = _hex_to_pixel(placed["coordinate"])
        is_desert = tile["type"] == "DESERT"
        is_resource = tile["type"] == "RESOURCE_TILE"
        if not is_desert and not is_resource:
            continue

        if is_desert:
            asset = _asset_path("/assets/tiles/desert.svg")
        else:
            resource = tile.get("resource")
            if resource is None:
                raise ValueError(f"resource tile {tile['id']} has no resource")
            asset = _asset_path(f"/assets/tiles/{resource.lower()}.svg")
        _paste_asset(
            canvas,
            asset,
            x_svg=center["x"] - TILE_WIDTH / 2,
            y_svg=center["y"] - TILE_HEIGHT / 2,
            width_svg=TILE_WIDTH,
            height_svg=TILE_HEIGHT,
            transform=transform,
        )

        number = tile.get("number") if is_resource else None
        if number:
            _paste_asset(
                canvas,
                _asset_path(f"/assets/numbers/{number}.svg"),
                x_svg=center["x"] - NUMBER_TOKEN_SIZE / 2,
                y_svg=center["y"] - NUMBER_TOKEN_SIZE / 2 + NUMBER_TOKEN_Y_OFFSET,
                width_svg=NUMBER_TOKEN_SIZE,
                height_svg=NUMBER_TOKEN_SIZE,
                transform=transform,
            )

