"""Create a visual pixel inspection sheet for Catan board screenshots.

This is intentionally image-only. It does not use engine state or a board
manifest. The goal is to make resolution loss visible to a human before we
decide what a VLM should be expected to read.

Usage:
    uv run python -m scripts.probes.inspect_board_pixels \
        --image playground/screenshots/turn_1_RED.png \
        --output /tmp/catan_pixel_inspection.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RESOLUTIONS = (1024, 768, 512, 384, 256)
ZOOM_RESOLUTIONS = (768, 512, 384, 256)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _fit_square(img: Image.Image, size: int) -> Image.Image:
    """Pad to square, then resize to size x size."""
    img = img.convert("RGB")
    side = max(img.size)
    square = Image.new("RGB", (side, side), (8, 90, 140))
    x = (side - img.width) // 2
    y = (side - img.height) // 2
    square.paste(img, (x, y))
    return square.resize((size, size), Image.Resampling.LANCZOS)


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    font = _font(18)
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 6
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0),
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def _center_crop(img: Image.Image, box_size: int) -> Image.Image:
    w, h = img.size
    side = min(box_size, w, h)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return img.crop((x0, y0, x0 + side, y0 + side))


def build_sheet(image_path: Path) -> Image.Image:
    src = Image.open(image_path).convert("RGB")

    full_tiles: list[tuple[str, Image.Image]] = []
    for resolution in RESOLUTIONS:
        down = _fit_square(src, resolution)
        display = down.resize((256, 256), Image.Resampling.NEAREST)
        full_tiles.append((f"full board {resolution}px -> zoomed", display))

    zoom_tiles: list[tuple[str, Image.Image]] = []
    square = _fit_square(src, 1024)

    # These are deliberately generic crops: center number cluster, lower robber/number
    # area, and right-side port/road area. A future manifest-based version should crop
    # exact tile/node/edge slots.
    crop_specs = [
        ("center symbols", (352, 352, 672, 672)),
        ("lower symbols", (336, 500, 688, 852)),
        ("right port/edge", (560, 300, 912, 652)),
    ]

    for crop_name, box in crop_specs:
        crop = square.crop(box)
        for resolution in ZOOM_RESOLUTIONS:
            scale = resolution / 1024
            resized = crop.resize(
                (max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
                Image.Resampling.LANCZOS,
            )
            display = resized.resize((220, 220), Image.Resampling.NEAREST)
            zoom_tiles.append((f"{crop_name} @ {resolution}px", display))

    margin = 24
    gap = 18
    label_h = 34
    tile_w = 256
    tile_h = 256 + label_h

    rows = []
    rows.append(full_tiles)
    for i in range(0, len(zoom_tiles), 4):
        rows.append(zoom_tiles[i : i + 4])

    sheet_w = margin * 2 + max(len(row) for row in rows) * tile_w + (max(len(row) for row in rows) - 1) * gap
    sheet_h = margin * 2 + len(rows) * tile_h + (len(rows) - 1) * gap

    sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 246, 248))
    draw = ImageDraw.Draw(sheet)
    title_font = _font(22)
    draw.text(
        (margin, 6),
        f"Pixel inspection: {image_path.name}",
        fill=(20, 24, 32),
        font=title_font,
    )

    y = margin + 16
    for row in rows:
        x = margin
        for label, tile in row:
            sheet.paste(tile, (x, y + label_h))
            _draw_label(draw, (x + 8, y + 8), label)
            x += tile_w + gap
        y += tile_h + gap

    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Catan board pixels at VLM resolutions.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sheet = build_sheet(args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
