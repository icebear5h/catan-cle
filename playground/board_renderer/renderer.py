"""The renderer itself: one engine state painted onto one PIL canvas."""

import io
import math

from PIL import Image, ImageDraw

from cle.game_engine.models.enums import CITY, SETTLEMENT
from cle.game_engine.models.player import Color
from cle.game_engine.state import GameState

from .geometry import compute_node_positions, cube_to_pixel, hexagon_vertices
from .palette import (
    OCEAN_COLOR,
    PIPS,
    PLAYER_COLORS,
    PLAYER_OUTLINE,
    RESOURCE_COLORS,
    RESOURCE_LABELS,
    ROBBER_COLOR,
    TOKEN_BG,
    TOKEN_BORDER,
)
from .shapes import draw_city, draw_legend, draw_settlement, load_fonts

__all__ = ["CatanBoardRenderer"]


class CatanBoardRenderer:
    """Renders engine State to PIL Image."""

    def __init__(
        self,
        hex_size: int = 50,
        padding: int = 80,
        canvas_size: int = 800,
    ) -> None:
        self.hex_size = hex_size
        self.padding = padding
        self.canvas_size = canvas_size
        self._font_small, self._font_medium, self._font_large = load_fonts()

    def render(self, state: GameState, highlight_player: Color | None = None) -> Image.Image:
        """Render full board state to PIL Image.

        Args:
            state: engine State object
            highlight_player: optional color to emphasize

        Returns:
            PIL.Image.Image (RGBA, canvas_size x canvas_size)
        """
        board = state.board
        catan_map = board.map

        # Compute all positions
        node_positions = compute_node_positions(catan_map, self.hex_size)

        # Find bounds and compute transform to center on canvas
        tile_centers: list[tuple[float, float]] = []
        for coord in catan_map.land_tiles:
            tile_centers.append(cube_to_pixel(coord, self.hex_size))

        if not tile_centers:
            img = Image.new("RGB", (self.canvas_size, self.canvas_size), OCEAN_COLOR)
            return img

        min_x = min(p[0] for p in tile_centers) - self.hex_size * 2
        max_x = max(p[0] for p in tile_centers) + self.hex_size * 2
        min_y = min(p[1] for p in tile_centers) - self.hex_size * 2
        max_y = max(p[1] for p in tile_centers) + self.hex_size * 2

        board_w = max_x - min_x
        board_h = max_y - min_y
        usable = self.canvas_size - 2 * self.padding
        scale = min(usable / board_w, usable / board_h)

        def to_canvas(px: float, py: float) -> tuple[float, float]:
            cx = (px - min_x) * scale + self.padding
            cy = (py - min_y) * scale + self.padding
            return (cx, cy)

        scaled_hex = self.hex_size * scale

        # Create image
        img = Image.new("RGB", (self.canvas_size, self.canvas_size), OCEAN_COLOR)
        draw = ImageDraw.Draw(img)

        # Layer 1: Hex tiles
        for coord, tile in catan_map.land_tiles.items():
            center = cube_to_pixel(coord, self.hex_size)
            cx, cy = to_canvas(*center)
            vertices = hexagon_vertices(cx, cy, scaled_hex)

            fill = RESOURCE_COLORS.get(tile.resource, "#d2b48c")
            draw.polygon(vertices, fill=fill, outline="#2d2d2d", width=2)

            # Resource label at top of tile
            label = RESOURCE_LABELS.get(tile.resource, "")
            if label:
                draw.text(
                    (cx, cy - scaled_hex * 0.35),
                    label,
                    fill="#1a1a1a",
                    font=self._font_small,
                    anchor="mm",
                )

        # Layer 2: Number tokens
        for coord, tile in catan_map.land_tiles.items():
            if tile.number is None:
                continue
            center = cube_to_pixel(coord, self.hex_size)
            cx, cy = to_canvas(*center)

            token_r = scaled_hex * 0.30
            draw.ellipse(
                [cx - token_r, cy - token_r, cx + token_r, cy + token_r],
                fill=TOKEN_BG,
                outline=TOKEN_BORDER,
                width=2,
            )

            # Number text (red for 6 and 8)
            num_color = "#dc2626" if tile.number in (6, 8) else "#1a1a1a"
            draw.text(
                (cx, cy - 2),
                str(tile.number),
                fill=num_color,
                font=self._font_medium,
                anchor="mm",
            )

            # Pip dots below number
            pips = PIPS.get(tile.number, 0)
            if pips > 0:
                pip_y = cy + token_r * 0.55
                total_w = (pips - 1) * 4
                start_x = cx - total_w / 2
                for i in range(pips):
                    px = start_x + i * 4
                    draw.ellipse(
                        [px - 1.5, pip_y - 1.5, px + 1.5, pip_y + 1.5],
                        fill=num_color,
                    )

        # Layer 3: Ports
        for resource, port_node_ids in catan_map.port_nodes.items():
            port_node_list = list(port_node_ids)
            if len(port_node_list) < 2:
                continue

            # Position label between the two port nodes, shifted outward
            first, second = (node_positions.get(nid) for nid in port_node_list[:2])
            if first is None or second is None:
                continue

            mid_x = (first[0] + second[0]) / 2
            mid_y = (first[1] + second[1]) / 2

            # Push outward from board center
            board_cx = sum(p[0] for p in tile_centers) / len(tile_centers)
            board_cy = sum(p[1] for p in tile_centers) / len(tile_centers)
            dx = mid_x - board_cx
            dy = mid_y - board_cy
            dist = math.sqrt(dx * dx + dy * dy) or 1
            push = self.hex_size * 0.6
            label_x = mid_x + dx / dist * push
            label_y = mid_y + dy / dist * push

            cx, cy = to_canvas(label_x, label_y)

            if resource is None:
                port_text = "3:1"
            else:
                port_text = f"{RESOURCE_LABELS.get(resource, '?')}2:1"

            # Draw port indicator
            draw.rounded_rectangle(
                [cx - 16, cy - 8, cx + 16, cy + 8],
                radius=4,
                fill="#fef9c3",
                outline="#854d0e",
                width=1,
            )
            draw.text(
                (cx, cy),
                port_text,
                fill="#854d0e",
                font=self._font_small,
                anchor="mm",
            )

            # Lines from port label to nodes
            for nid in port_node_list[:2]:
                pos = node_positions.get(nid)
                if pos:
                    ncx, ncy = to_canvas(*pos)
                    draw.line([(cx, cy), (ncx, ncy)], fill="#854d0e", width=1)

        # Layer 4: Roads
        for edge, color in board.roads.items():
            n1, n2 = edge
            pos1 = node_positions.get(n1)
            pos2 = node_positions.get(n2)
            if not pos1 or not pos2:
                continue

            # Avoid drawing duplicate inverted edges
            if n1 > n2:
                continue

            c1 = to_canvas(*pos1)
            c2 = to_canvas(*pos2)
            road_color = PLAYER_COLORS.get(color, "#666666")
            road_outline = PLAYER_OUTLINE.get(color, "#333333")

            # Draw road with outline for visibility
            draw.line([c1, c2], fill=road_outline, width=6)
            draw.line([c1, c2], fill=road_color, width=4)

        # Layer 5: Settlements and Cities
        for node_id, (color, building_type) in board.buildings.items():
            pos = node_positions.get(node_id)
            if not pos:
                continue

            cx, cy = to_canvas(*pos)
            fill = PLAYER_COLORS.get(color, "#666666")
            outline = PLAYER_OUTLINE.get(color, "#333333")

            if building_type == SETTLEMENT:
                draw_settlement(draw, cx, cy, scaled_hex * 0.22, fill, outline)
            elif building_type == CITY:
                draw_city(draw, cx, cy, scaled_hex * 0.28, fill, outline)

        # Layer 6: Robber
        robber_coord = board.robber_coordinate
        if robber_coord:
            rcx, rcy = to_canvas(*cube_to_pixel(robber_coord, self.hex_size))
            robber_r = scaled_hex * 0.20
            draw.ellipse(
                [rcx - robber_r, rcy - robber_r, rcx + robber_r, rcy + robber_r],
                fill=ROBBER_COLOR,
                outline="#ef4444",
                width=2,
            )
            draw.text(
                (rcx, rcy), "R", fill="#ef4444", font=self._font_small, anchor="mm"
            )

        # Layer 7: Legend
        draw_legend(
            draw,
            list(state.colors),
            canvas_size=self.canvas_size,
            font=self._font_small,
        )

        return img

    def render_to_bytes(self, state: GameState, highlight_player: Color | None = None) -> bytes:
        """Render and return PNG bytes."""
        img = self.render(state, highlight_player)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def render_to_file(
        self,
        state: GameState,
        path: str,
        highlight_player: Color | None = None,
    ) -> str:
        """Render and save to file, return path."""
        img = self.render(state, highlight_player)
        img.save(path, format="PNG")
        return path
