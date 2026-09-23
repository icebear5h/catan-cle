"""Asset resolution, SVG rasterization, and canvas compositing primitives."""

from __future__ import annotations

import io
import math
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw

from evals.catan_board_bench.annotations import HEX_SIZE
from evals.catan_board_bench.render.style import ASSET_ROOT, RESOURCE_COLORS


def _asset_path(web_path: str) -> Path:
    if not web_path.startswith("/assets/"):
        raise ValueError(f"expected frontend asset path, got {web_path}")
    return ASSET_ROOT / web_path.removeprefix("/assets/")


def _paste_asset(
    canvas: Image.Image,
    path: Path,
    *,
    x_svg: float,
    y_svg: float,
    width_svg: float,
    height_svg: float,
    transform: dict[str, float],
) -> list[int]:
    x, y = _svg_point_to_px(x_svg, y_svg, transform)
    width = max(1, int(round(width_svg * transform["scale"])))
    height = max(1, int(round(height_svg * transform["scale"])))
    asset = _raster_asset(path, width, height)
    canvas.alpha_composite(asset, (x, y))
    return [x, y, x + width, y + height]


def _paste_rotated_asset(
    canvas: Image.Image,
    path: Path,
    *,
    center_svg: tuple[float, float],
    width_svg: float,
    height_svg: float,
    angle_deg: float,
    transform: dict[str, float],
) -> list[int]:
    center_x, center_y = _svg_point_to_px(center_svg[0], center_svg[1], transform)
    width = max(1, int(round(width_svg * transform["scale"])))
    height = max(1, int(round(height_svg * transform["scale"])))
    asset = _raster_asset(path, width, height)
    rotated = asset.rotate(-angle_deg, expand=True, resample=Image.Resampling.BICUBIC)
    x = int(round(center_x - rotated.width / 2))
    y = int(round(center_y - rotated.height / 2))
    canvas.alpha_composite(rotated, (x, y))
    return [x, y, x + rotated.width, y + rotated.height]


def _svg_point_to_px(x: float, y: float, transform: dict[str, float]) -> tuple[int, int]:
    px = (x - transform["min_x"]) * transform["scale"] + transform["offset_x"]
    py = (y - transform["min_y"]) * transform["scale"] + transform["offset_y"]
    return int(round(px)), int(round(py))


@lru_cache(maxsize=512)
def _raster_asset(path: Path, width: int, height: int) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".svg":
        return _raster_svg(path, width, height)
    return Image.open(path).convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)


def _raster_svg(path: Path, width: int, height: int) -> Image.Image:
    binary = shutil.which("rsvg-convert")
    if binary is None:
        raise RuntimeError("rsvg-convert is required for Python Catan board rendering")
    result = subprocess.run(
        [binary, "-f", "png", "-w", str(width), "-h", str(height), str(path)],
        check=True,
        capture_output=True,
    )
    return Image.open(io.BytesIO(result.stdout)).convert("RGBA")


def _draw_hex_fallback(
    canvas: Image.Image,
    center: dict[str, float],
    resource: str | None,
    transform: dict[str, float],
) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    points = []
    for index in range(6):
        angle = math.pi / 3 * index - math.pi / 6
        x = center["x"] + HEX_SIZE * math.cos(angle)
        y = center["y"] + HEX_SIZE * math.sin(angle)
        points.append(_svg_point_to_px(x, y, transform))
    draw.polygon(points, fill=RESOURCE_COLORS.get(str(resource), (193, 154, 107, 230)))
