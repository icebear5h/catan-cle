"""Canvas drawing for one probe layout: hexes, labels, and font loading."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from cle.game_engine.models.coordinate_system import UNIT_VECTORS, Direction
from evals.catan_board_bench.hex_direction_probe.layout import (
    _OFFSETS,
    _PALETTES,
    ANCHOR_LABEL,
    DIRECTION_ORDER,
    cube_to_pixel,
)


def render_layout(
    assignment: dict[Direction, str],
    *,
    layout_index: int,
    image_size: int,
) -> tuple[Image.Image, dict[str, list[int]]]:
    canvas = Image.new("RGB", (image_size, image_size), (8, 31, 52))
    draw = ImageDraw.Draw(canvas)
    offset = _OFFSETS[layout_index % len(_OFFSETS)]
    center = (image_size / 2 + offset[0], image_size / 2 + offset[1])
    radius = image_size * (0.142 + 0.003 * (layout_index % 3))
    label_font = load_font(max(28, int(radius * 0.58)))
    caption_font = load_font(max(15, int(radius * 0.25)))
    palette = _PALETTES[layout_index % len(_PALETTES)]
    centers: dict[str, list[int]] = {"anchor": [round(center[0]), round(center[1])]}

    draw.text(
        (image_size / 2, 18),
        "ONE-HOP HEX NEIGHBORS",
        font=caption_font,
        fill=(196, 225, 243),
        anchor="ma",
    )

    for direction_index, direction in enumerate(DIRECTION_ORDER):
        pixel_center = cube_to_pixel(
            UNIT_VECTORS[direction],
            center=center,
            radius=radius,
        )
        centers[direction.value] = [round(pixel_center[0]), round(pixel_center[1])]
        fill = palette[(direction_index * 5 + layout_index * 2) % len(palette)]
        draw_hex(
            draw,
            center=pixel_center,
            radius=radius,
            fill=fill,
            outline=(230, 244, 251),
            width=5,
        )
        draw_label(
            draw,
            pixel_center,
            assignment[direction],
            font=label_font,
            fill=(255, 255, 255),
        )

    draw_hex(
        draw,
        center=center,
        radius=radius,
        fill=(13, 148, 136),
        outline=(255, 255, 255),
        width=7,
    )
    draw_label(
        draw,
        center,
        ANCHOR_LABEL,
        font=label_font,
        fill=(255, 255, 255),
    )
    return canvas, centers


def draw_hex(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    radius: float,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    width: int,
) -> None:
    points = [
        (
            center[0] + radius * math.cos(math.radians(-90 + 60 * vertex)),
            center[1] + radius * math.sin(math.radians(-90 + 60 * vertex)),
        )
        for vertex in range(6)
    ]
    draw.polygon(points, fill=fill, outline=outline, width=width)


def draw_label(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    label: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bbox = draw.textbbox(center, label, font=font, anchor="mm", stroke_width=2)
    pad = 7
    draw.rounded_rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        radius=8,
        fill=(4, 18, 31, 215),
        outline=(255, 255, 255, 190),
        width=2,
    )
    draw.text(
        center,
        label,
        font=font,
        fill=fill,
        anchor="mm",
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()

