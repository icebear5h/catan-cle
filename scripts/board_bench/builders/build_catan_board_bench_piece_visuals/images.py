"""Compose primitive board images from the frontend assets."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from evals.catan_board_bench.render import WATER_RGB, _asset_path, _raster_asset

__all__ = [
    "base_canvas",
    "crop_square",
    "paste_asset",
    "render_node_image",
    "render_port_image",
    "render_road_image",
    "render_robber_image",
    "render_tile_image",
    "variant_offset",
]


def render_tile_image(
    resource: str | None,
    number: int | None,
    image_size: int,
    variant: int,
) -> tuple[Image.Image, list[int], list[int] | None]:
    canvas = base_canvas(image_size)
    tile_size = int(image_size * 0.72)
    tile_x = int((image_size - tile_size) / 2 + variant_offset(variant)[0])
    tile_y = int((image_size - tile_size) / 2 + variant_offset(variant)[1])
    asset_name = "desert" if resource is None else resource.lower()
    tile_bbox = paste_asset(
        canvas, _asset_path(f"/assets/tiles/{asset_name}.svg"), tile_x, tile_y, tile_size, tile_size
    )
    number_bbox = None
    if number is not None:
        number_size = int(image_size * 0.23)
        number_x = int(image_size / 2 - number_size / 2)
        number_y = int(image_size / 2 - number_size / 2 + image_size * 0.06)
        number_bbox = paste_asset(
            canvas,
            _asset_path(f"/assets/numbers/{number}.svg"),
            number_x,
            number_y,
            number_size,
            number_size,
        )
    return canvas.convert("RGB"), tile_bbox, number_bbox


def render_road_image(
    color: str | None, image_size: int, variant: int
) -> tuple[Image.Image, list[int] | None, int]:
    canvas = base_canvas(image_size)
    draw = ImageDraw.Draw(canvas, "RGBA")
    angle = 120 if variant % 2 == 0 else 60
    if color is None:
        draw.line(
            [(image_size * 0.25, image_size * 0.55), (image_size * 0.75, image_size * 0.45)],
            fill=(230, 230, 220, 85),
            width=max(4, image_size // 28),
        )
        return canvas.convert("RGB"), None, angle

    width = int(image_size * 0.28)
    height = int(image_size * 0.62)
    asset = _raster_asset(_asset_path(f"/assets/pieces/road_{color.lower()}.svg"), width, height)
    rotated = asset.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)
    x = int(image_size / 2 - rotated.width / 2)
    y = int(image_size / 2 - rotated.height / 2)
    canvas.alpha_composite(rotated, (x, y))
    return canvas.convert("RGB"), [x, y, x + rotated.width, y + rotated.height], angle


def render_node_image(
    color: str | None,
    building: str | None,
    image_size: int,
    variant: int,
) -> tuple[Image.Image, list[int]]:
    canvas = base_canvas(image_size)
    draw = ImageDraw.Draw(canvas, "RGBA")
    center = (
        image_size // 2 + variant_offset(variant)[0],
        image_size // 2 + variant_offset(variant)[1],
    )
    marker_radius = int(image_size * 0.12)
    marker_bbox = [
        center[0] - marker_radius,
        center[1] - marker_radius,
        center[0] + marker_radius,
        center[1] + marker_radius,
    ]
    draw.ellipse(marker_bbox, fill=(240, 224, 115, 95), outline=(92, 86, 30, 180), width=3)
    if not color or not building:
        return canvas.convert("RGB"), marker_bbox

    size = int(image_size * (0.36 if building == "SETTLEMENT" else 0.40))
    width = int(size * (1.12 if building == "CITY" else 1.0))
    x = int(center[0] - width / 2)
    y = int(center[1] - size * 0.72)
    bbox = paste_asset(
        canvas,
        _asset_path(f"/assets/pieces/{building.lower()}_{color.lower()}.svg"),
        x,
        y,
        width,
        size,
    )
    return canvas.convert("RGB"), bbox


def render_port_image(
    resource: str | None, image_size: int, variant: int
) -> tuple[Image.Image, list[int]]:
    canvas = base_canvas(image_size)
    slug = "generic" if resource is None else resource.lower()
    size = int(image_size * 0.52)
    x = int(image_size / 2 - size / 2 + variant_offset(variant)[0])
    y = int(image_size / 2 - size / 2 + variant_offset(variant)[1])
    bbox = paste_asset(canvas, _asset_path(f"/assets/tiles/port_{slug}.svg"), x, y, size, size)
    return canvas.convert("RGB"), bbox


def render_robber_image(
    resource: str | None,
    robber: bool,
    image_size: int,
    variant: int,
) -> tuple[Image.Image, list[int], list[int] | None]:
    canvas, tile_bbox, _ = render_tile_image(
        resource, None if resource is None else 8, image_size, variant
    )
    canvas = canvas.convert("RGBA")
    robber_bbox = None
    if robber:
        size = int(image_size * 0.28)
        x = int(image_size * 0.52 + variant_offset(variant)[0])
        y = int(image_size * 0.52 + variant_offset(variant)[1])
        robber_bbox = paste_asset(
            canvas, _asset_path("/assets/pieces/robber.svg"), x, y, size, size
        )
    return canvas.convert("RGB"), tile_bbox, robber_bbox


def base_canvas(image_size: int) -> Image.Image:
    canvas = Image.new("RGBA", (image_size, image_size), (*WATER_RGB, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    step = max(16, image_size // 8)
    for pos in range(0, image_size + 1, step):
        draw.line([(pos, 0), (pos, image_size)], fill=(255, 255, 255, 26), width=1)
        draw.line([(0, pos), (image_size, pos)], fill=(255, 255, 255, 26), width=1)
    return canvas


def paste_asset(
    canvas: Image.Image, path: Path, x: int, y: int, width: int, height: int
) -> list[int]:
    asset = _raster_asset(path, width, height)
    canvas.alpha_composite(asset, (x, y))
    return [x, y, x + width, y + height]


def crop_square(
    image: Image.Image, center_x: int, center_y: int, crop_size: int, output_size: int
) -> Image.Image:
    half = crop_size // 2
    left = center_x - half
    top = center_y - half
    right = left + crop_size
    bottom = top + crop_size
    crop = Image.new("RGB", (crop_size, crop_size), WATER_RGB)
    source_left = max(left, 0)
    source_top = max(top, 0)
    source_right = min(right, image.width)
    source_bottom = min(bottom, image.height)
    paste_x = source_left - left
    paste_y = source_top - top
    crop.paste(
        image.crop((source_left, source_top, source_right, source_bottom)), (paste_x, paste_y)
    )
    return crop.resize((output_size, output_size), Image.Resampling.LANCZOS)


def variant_offset(variant: int) -> tuple[int, int]:
    offsets = [(-2, 3), (3, -2), (0, 0), (-4, -1)]
    return offsets[variant % len(offsets)]
