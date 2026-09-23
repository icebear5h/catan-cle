"""Question categories, alias table, and the engine-scored system prompts."""

from __future__ import annotations

from cle.game_engine.models.player import Color
from evals.json_types import JsonDict as JsonDict

COLOR_TOKEN_NAMES = [color.value for color in Color]


VISUAL_CATEGORIES = [
    "robber_tile",
    "robber_resource_number",
    "tile_resource_number",
    "tile_has_robber",
    "node_occupancy",
    "edge_road_owner",
    "color_road_locations",
    "port_trade_type",
    "port_occupancy",
    "color_building_counts",
    "color_road_count",
]
LOGIC_CATEGORIES = [
    "nodes_connected",
    "edge_connects_nodes",
    "port_type_nodes",
    "node_adjacent_tiles",
]
PROBE_CATEGORIES = [
    "isolated_tile_resource_number",
    "isolated_road_owner",
    "isolated_node_occupancy",
    "isolated_port_trade_type",
    "isolated_robber_presence",
    "local_patch_tile_resource_number",
    "local_patch_edge_road_owner",
    "local_patch_node_occupancy",
    "local_patch_port_trade_type",
    "local_patch_robber_presence",
    "isolated_hex_direction_to_label",
    "isolated_hex_label_to_direction",
]
SUITE_CATEGORIES = {
    "visual": VISUAL_CATEGORIES,
    "logic": LOGIC_CATEGORIES,
    "probe": PROBE_CATEGORIES,
}
DEFAULT_CATEGORIES = VISUAL_CATEGORIES

CATEGORY_ALIASES = {
    "isolated_tile_resource_number": "tile_resource_number",
    "local_patch_tile_resource_number": "tile_resource_number",
    "isolated_road_owner": "edge_road_owner",
    "local_patch_edge_road_owner": "edge_road_owner",
    "isolated_node_occupancy": "node_occupancy",
    "local_patch_node_occupancy": "node_occupancy",
    "isolated_port_trade_type": "port_trade_type",
    "local_patch_port_trade_type": "port_trade_type",
    "isolated_robber_presence": "robber_presence",
    "local_patch_robber_presence": "robber_presence",
}


SYSTEM_PROMPT = """You are answering engine-scored questions about a Catan board screenshot.

Rules:
- Use only visible public board information from the image plus the supplied fixed atlas context.
- Do not infer hidden hands or hidden development cards.
- Return only the final answer string, with no explanation.
- Preserve exact tokens such as <T07>, <N18>, <E03_17>, <RED>, and <WOOD>.
- If the answer is empty or absent, use the exact sentinel requested by the question context."""

LOGIC_SYSTEM_PROMPT = """You are answering engine-scored symbolic questions about a fixed Catan board contract.

Rules:
- Use only the supplied fixed atlas/topology context and public board-state tokens.
- Do not infer hidden hands or hidden development cards.
- Return only the final answer string, with no explanation.
- Preserve exact tokens such as <T07>, <N18>, <E03_17>, <RED>, and <WOOD>.
- If the answer is empty or absent, use the exact sentinel requested by the question context."""

PROBE_SYSTEM_PROMPT = """You are answering engine-scored visual-primitive questions about Catan image crops.

Rules:
- Use only visible information from the image.
- Return only the final answer string, with no explanation.
- Preserve exact tokens such as <RED>, <GREEN>, <MYSTIC_BLUE>, <SETTLEMENT>, <CITY>, <WOOD>, and <ORE>.
- If the answer is empty or absent, use the exact sentinel requested by the question context."""

