"""
Server-side board renderer: engine State -> PIL Image -> PNG.

Renders Catan board with colored hex tiles, number tokens with pip dots,
roads, settlements, cities, robber, ports, and player color legend.

Dual purpose: VLM benchmark now, training data pipeline later.
"""

import io
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from cle.game_engine.models.board import Board
from cle.game_engine.models.coordinate_system import cube_to_axial
from cle.game_engine.models.enums import NodeRef, SETTLEMENT, CITY
from cle.game_engine.models.map import CatanMap, LandTile, Port
from cle.game_engine.models.player import Color

# --- Colors ---

RESOURCE_COLORS = {
    "WOOD": "#228b22",
    "BRICK": "#c45a2c",
    "SHEEP": "#90ee90",
    "WHEAT": "#f4d03f",
    "ORE": "#708090",
    None: "#d2b48c",  # desert
}

RESOURCE_LABELS = {
    "WOOD": "W",
    "BRICK": "B",
    "SHEEP": "S",
    "WHEAT": "Wh",
    "ORE": "O",
    None: "",
}

PLAYER_COLORS = {
    Color.RED: "#dc2626",
    Color.BLUE: "#2563eb",
    Color.WHITE: "#e5e7eb",
    Color.ORANGE: "#ea580c",
}

PLAYER_OUTLINE = {
    Color.RED: "#991b1b",
    Color.BLUE: "#1d4ed8",
    Color.WHITE: "#9ca3af",
    Color.ORANGE: "#c2410c",
}

OCEAN_COLOR = "#1e3a5f"
TOKEN_BG = "#fef3c7"
TOKEN_BORDER = "#92400e"
ROBBER_COLOR = "#1f2937"

# Number -> pip count (probability dots)
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

# --- Hex Geometry (pointy-top, y-down screen coords) ---

def hex_to_pixel(q: int, r: int, size: float) -> Tuple[float, float]:
    """Convert axial hex coordinates to pixel position (pointy-top, y-down)."""
    x = size * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
    y = size * (3.0 / 2 * r)
    return (x, y)


def cube_to_pixel(cube_coord: Tuple[int, int, int], size: float) -> Tuple[float, float]:
    """Convert engine cube coordinate to pixel position."""
    q, r = cube_to_axial(cube_coord)
    return hex_to_pixel(q, r, size)


# Node offsets from hex center for pointy-top hex (y-down)
NODE_OFFSETS = {
    NodeRef.NORTH: lambda s: (0, -s),
    NodeRef.NORTHEAST: lambda s: (s * math.sqrt(3) / 2, -s / 2),
    NodeRef.SOUTHEAST: lambda s: (s * math.sqrt(3) / 2, s / 2),
    NodeRef.SOUTH: lambda s: (0, s),
    NodeRef.SOUTHWEST: lambda s: (-s * math.sqrt(3) / 2, s / 2),
    NodeRef.NORTHWEST: lambda s: (-s * math.sqrt(3) / 2, -s / 2),
}


def hexagon_vertices(cx: float, cy: float, size: float) -> List[Tuple[float, float]]:
    """Get 6 vertices of a pointy-top hexagon centered at (cx, cy)."""
    vertices = []
    for i in range(6):
        angle = math.radians(60 * i - 30)  # pointy-top: start at -30 deg
        vx = cx + size * math.cos(angle)
        vy = cy + size * math.sin(angle)
        vertices.append((vx, vy))
    return vertices


def compute_node_positions(
    catan_map: CatanMap, hex_size: float
) -> Dict[int, Tuple[float, float]]:
    """Compute pixel positions for all nodes by averaging across shared tiles."""
    node_accum: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    for coord, tile in catan_map.tiles.items():
        if not hasattr(tile, "nodes"):
            continue
        tile_center = cube_to_pixel(coord, hex_size)
        for node_ref, node_id in tile.nodes.items():
            offset_fn = NODE_OFFSETS.get(node_ref)
            if offset_fn is None:
                continue
            dx, dy = offset_fn(hex_size)
            node_accum[node_id].append((tile_center[0] + dx, tile_center[1] + dy))

    result = {}
    for node_id, positions in node_accum.items():
        avg_x = sum(p[0] for p in positions) / len(positions)
        avg_y = sum(p[1] for p in positions) / len(positions)
        result[node_id] = (avg_x, avg_y)

    return result


class CatanBoardRenderer:
    """Renders engine State to PIL Image."""

    def __init__(self, hex_size: int = 50, padding: int = 80, canvas_size: int = 800):
        self.hex_size = hex_size
        self.padding = padding
        self.canvas_size = canvas_size

        # Try to load a monospace font, fall back to default
        self._font_small = ImageFont.load_default()
        self._font_medium = ImageFont.load_default()
        self._font_large = ImageFont.load_default()
        try:
            self._font_small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 10)
            self._font_medium = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
            self._font_large = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 18)
        except (OSError, IOError):
            try:
                self._font_small = ImageFont.truetype("DejaVuSansMono.ttf", 10)
                self._font_medium = ImageFont.truetype("DejaVuSansMono.ttf", 14)
                self._font_large = ImageFont.truetype("DejaVuSansMono.ttf", 18)
            except (OSError, IOError):
                pass  # use defaults

    def render(self, state, highlight_player: Optional[Color] = None) -> Image.Image:
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
        tile_centers = []
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

        def to_canvas(px: float, py: float) -> Tuple[float, float]:
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
            positions = [node_positions.get(nid) for nid in port_node_list[:2]]
            if not all(positions):
                continue

            mid_x = (positions[0][0] + positions[1][0]) / 2
            mid_y = (positions[0][1] + positions[1][1]) / 2

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
                self._draw_settlement(draw, cx, cy, scaled_hex * 0.22, fill, outline)
            elif building_type == CITY:
                self._draw_city(draw, cx, cy, scaled_hex * 0.28, fill, outline)

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
        self._draw_legend(draw, state.colors)

        return img

    def _draw_settlement(
        self,
        draw: ImageDraw.ImageDraw,
        cx: float,
        cy: float,
        size: float,
        fill: str,
        outline: str,
    ):
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

    def _draw_city(
        self,
        draw: ImageDraw.ImageDraw,
        cx: float,
        cy: float,
        size: float,
        fill: str,
        outline: str,
    ):
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

    def _draw_legend(self, draw: ImageDraw.ImageDraw, colors: List[Color]):
        """Draw player color legend in bottom-left corner."""
        x_start = 10
        y_start = self.canvas_size - 20 * len(colors) - 10

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
                font=self._font_small,
                anchor="lm",
            )

    def render_to_bytes(self, state, **kwargs) -> bytes:
        """Render and return PNG bytes."""
        img = self.render(state, **kwargs)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def render_to_file(self, state, path: str, **kwargs) -> str:
        """Render and save to file, return path."""
        img = self.render(state, **kwargs)
        img.save(path, format="PNG")
        return path
