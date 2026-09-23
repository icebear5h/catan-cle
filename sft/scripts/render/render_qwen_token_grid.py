"""Render Qwen3-VL effective 32x32 visual-token grid overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from sft.paths import SFT_DIAGNOSTICS_ROOT


def load_font(size: int = 14) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_overlay(
    image_path: Path,
    output_path: Path,
    *,
    target_size: int,
    token_px: int,
    label: bool,
) -> None:
    image = Image.open(image_path).convert("RGB").resize((target_size, target_size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(max(10, target_size // 64))

    cols = target_size // token_px
    rows = target_size // token_px

    for x in range(0, target_size + 1, token_px):
        draw.line([(x, 0), (x, target_size)], fill=(0, 255, 255, 160), width=1)
    for y in range(0, target_size + 1, token_px):
        draw.line([(0, y), (target_size, y)], fill=(0, 255, 255, 160), width=1)

    if label:
        for row in range(rows):
            for col in range(cols):
                text = f"{row},{col}"
                tx = col * token_px + 2
                ty = row * token_px + 2
                draw.rectangle(
                    [(tx - 1, ty - 1), (tx + len(text) * 8, ty + 12)],
                    fill=(0, 0, 0, 120),
                )
                draw.text((tx, ty), text, fill=(255, 255, 255, 230), font=font)

    caption = (
        f"{target_size}x{target_size}; effective Qwen visual grid "
        f"{rows}x{cols} = {rows * cols} tokens; {token_px}px/token"
    )
    text_bbox = draw.textbbox((0, 0), caption, font=font)
    draw.rectangle(
        [(8, target_size - 28), (text_bbox[2] + 18, target_size - 6)],
        fill=(0, 0, 0, 170),
    )
    draw.text((12, target_size - 25), caption, fill=(255, 255, 255, 240), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        default=SFT_DIAGNOSTICS_ROOT / "qwen_token_grid",
        type=Path,
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 768, 1024])
    parser.add_argument("--token-px", type=int, default=32)
    parser.add_argument("--label", action="store_true")
    args = parser.parse_args()

    for size in args.sizes:
        output = args.output_dir / f"{args.image.stem}_qwen_grid_{size}.png"
        render_overlay(
            args.image,
            output,
            target_size=size,
            token_px=args.token_px,
            label=args.label,
        )
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
