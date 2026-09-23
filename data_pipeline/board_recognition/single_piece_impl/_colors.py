"""Player colour words and the seeded novel probe colour sprites."""

from __future__ import annotations

import colorsys
import re
import shutil
from pathlib import Path

from data_pipeline.board_recognition.single_piece_impl._config import (
    HEX_COLOR_RE,
    NOVEL_HUE_INTERVALS,
    NOVEL_SPRITE_BASE,
    SPRITE_PIECES,
)
from data_pipeline.board_recognition.spatial_localization import _stable_rank
from evals.catan_board_bench import render as board_render


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
