"""Drawing primitives: the fonts, the two building shapes, and the legend."""

from PIL import ImageDraw, ImageFont

from cle.game_engine.models.player import Color

from .palette import PLAYER_COLORS, PLAYER_OUTLINE

__all__ = ["Font", "draw_city", "draw_legend", "draw_settlement", "load_fonts"]

Font = ImageFont.ImageFont | ImageFont.FreeTypeFont


def load_fonts() -> tuple[Font, Font, Font]:
    """Return the small, medium, and large faces, falling back to PIL's default."""
    small: Font = ImageFont.load_default()
    medium: Font = ImageFont.load_default()
    large: Font = ImageFont.load_default()
    try:
        small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 10)
        medium = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
        large = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 18)
    except OSError:
        try:
            small = ImageFont.truetype("DejaVuSansMono.ttf", 10)
            medium = ImageFont.truetype("DejaVuSansMono.ttf", 14)
            large = ImageFont.truetype("DejaVuSansMono.ttf", 18)
        except OSError:
            pass  # use defaults
    return small, medium, large


def draw_settlement(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    size: float,
    fill: str,
    outline: str,
) -> None:
    """Draw a house-shaped settlement."""
    # Simple pentagon: square base + triangle roof
    half = size
    points = [
        (cx, cy - half * 1.3),  # roof peak
        (cx + half, cy - half * 0.3),  # top right
        (cx + half, cy + half),  # bottom right
        (cx - half, cy + half),  # bottom left
        (cx - half, cy - half * 0.3),  # top left
    ]
    draw.polygon(points, fill=fill, outline=outline, width=2)


def draw_city(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    size: float,
    fill: str,
    outline: str,
) -> None:
    """Draw a larger city shape (house + tower)."""
    half = size
    # Main building
    points = [
        (cx - half * 0.3, cy - half * 1.5),  # tower peak
        (cx + half * 0.3, cy - half * 1.5),
        (cx + half * 0.3, cy - half * 0.5),
        (cx + half, cy - half * 0.5),  # house roof right
        (cx + half, cy + half),  # bottom right
        (cx - half, cy + half),  # bottom left
        (cx - half, cy - half * 0.5),  # house roof left
        (cx - half * 0.3, cy - half * 0.5),
    ]
    draw.polygon(points, fill=fill, outline=outline, width=2)


def draw_legend(
    draw: ImageDraw.ImageDraw,
    colors: list[Color],
    *,
    canvas_size: int,
    font: Font,
) -> None:
    """Draw player color legend in bottom-left corner."""
    x_start = 10
    y_start = canvas_size - 20 * len(colors) - 10

    for i, color in enumerate(colors):
        y = y_start + i * 20
        fill = PLAYER_COLORS.get(color, "#666666")
        outline = PLAYER_OUTLINE.get(color, "#333333")

        draw.rectangle([x_start, y, x_start + 14, y + 14], fill=fill, outline=outline, width=1)

        color_name = color.value if hasattr(color, "value") else str(color)
        draw.text(
            (x_start + 20, y + 7),
            color_name,
            fill="#ffffff",
            font=font,
            anchor="lm",
        )
