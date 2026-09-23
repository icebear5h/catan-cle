"""Answer normalization and part classification for the board-parts probe."""

from __future__ import annotations

import re

CANONICAL_CATEGORY_MAP = {
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

MODEL_PREFIX = "model_key"
CATEGORY_KEYS = {
    "tile": {"tile_resource_number", "tile_has_robber", "robber_tile", "robber_resource_number"},
    "node": {"node_occupancy", "nodes_connected"},
    "edge": {"edge_road_owner", "edge_connects_nodes", "color_road_locations"},
    "port": {"port_trade_type", "port_occupancy", "port_type_nodes"},
    "player": {"color_building_counts", "color_road_count"},
    "robber": {"robber_presence", "robber_tile", "robber_resource_number", "tile_has_robber"},
    "board-logic": {"nodes_connected", "edge_connects_nodes", "node_adjacent_tiles"},
}

TILE_RESOURCE_RE = re.compile(r"<(WOOD|BRICK|SHEEP|WHEAT|ORE|DESERT)>")
COLOR_TOKEN_RE = re.compile(r"<(?:RED|BLUE|ORANGE|WHITE|BLACK|GREEN|BRONZE|SILVER|GOLD|PINK|MYSTIC_BLUE)>")
EDGE_TOKEN_RE = re.compile(r"<E(\d{2})_(\d{2})>")
NODE_TOKEN_RE = re.compile(r"<N\d{2}>")
ROAD_RE = re.compile(r"\bROAD(?:S)?\b")
SETTLEMENT_RE = re.compile(r"<SETTLEMENT>")
CITY_RE = re.compile(r"<CITY>")
NUMBER_RE = re.compile(r"\b(?:2|3|4|5|6|8|9|10|11|12)\b")
RATIO_RE = re.compile(r"\b3:1\b|\b2:1\b")

__all__ = [
    "CANONICAL_CATEGORY_MAP",
    "CATEGORY_KEYS",
    "CITY_RE",
    "COLOR_TOKEN_RE",
    "EDGE_TOKEN_RE",
    "MODEL_PREFIX",
    "NODE_TOKEN_RE",
    "NUMBER_RE",
    "RATIO_RE",
    "ROAD_RE",
    "SETTLEMENT_RE",
    "TILE_RESOURCE_RE",
    "canonical_category",
    "classify_part",
    "color_from_text",
    "extract_edge_tokens",
    "extract_numbers_for",
    "is_empty_like",
    "normalize_response",
    "normalize_text",
    "parse_int",
]


def canonical_category(category: str) -> str:
    return CANONICAL_CATEGORY_MAP.get(category, category)


def normalize_text(value: object) -> str:
    text = str(value).upper().strip()
    replacements = {
        "NO NUMBER": "NO_NUMBER",
        "NO-NUMBER": "NO_NUMBER",
        "NO PLAYER": "NONE",
        "NO ONE": "NONE",
        "NONE.": "NONE",
        "EMPTY.": "EMPTY",
        "DESERT": "<DESERT>",
        "WOOD": "<WOOD>",
        "BRICK": "<BRICK>",
        "SHEEP": "<SHEEP>",
        "WHEAT": "<WHEAT>",
        "ORE": "<ORE>",
        "SETTLEMENT": "<SETTLEMENT>",
        "CITY": "<CITY>",
        "GEN": "GENERIC",
    }
    replacements.update(
        {color: f"<{color}>" for color in ["RED", "BLUE", "WHITE", "BLACK", "GREEN", "ORANGE"]}
    )
    replacements.update(
        {
            color_name.replace("_", separator): f"<{color_name}>"
            for color_name in ["MYSTIC_BLUE", "BRONZE", "SILVER", "GOLD", "PINK"]
            if "_" in color_name
            for separator in (" ", "-")
        }
    )
    for src, dst in replacements.items():
        text = re.sub(rf"(?<![A-Z0-9_<]){re.escape(src)}(?![A-Z0-9_>])", dst, text)
    text = re.sub(r"<T(\d{1,2})(?![0-9_]*>)", lambda match: f"<T{int(match.group(1)):02d}>", text)
    text = re.sub(r"<N(\d{1,2})(?![0-9_]*>)", lambda match: f"<N{int(match.group(1)):02d}>", text)
    text = re.sub(r"\bT(\d{1,2})\b", lambda match: f"<T{int(match.group(1)):02d}>", text)
    text = re.sub(r"\bN(\d{1,2})\b", lambda match: f"<N{int(match.group(1)):02d}>", text)
    text = re.sub(r"<E(\d{1,2})[_-](\d{1,2})(?![0-9_]*>)", lambda match: f"<E{int(match.group(1)):02d}_{int(match.group(2)):02d}>", text)
    text = re.sub(r"\bE(\d{1,2})[_-](\d{1,2})\b", lambda match: f"<E{int(match.group(1)):02d}_{int(match.group(2)):02d}>", text)
    return re.sub(r"\s+", " ", text)


def normalize_response(value: object) -> str:
    return normalize_text(value).upper().strip()


def classify_part(category: str) -> str:
    canonical = canonical_category(category)
    for part, cats in CATEGORY_KEYS.items():
        if canonical in cats:
            return part
    return "other"


def parse_int(text: str) -> int | None:
    match = re.search(r"\b\d+\b", text)
    return int(match.group(0)) if match else None


def is_empty_like(text: str) -> bool:
    return text in {"EMPTY", "NONE", ""} or bool(re.fullmatch(r"\s*(EMPTY|NONE)\s*", text))


def color_from_text(text: str) -> str | None:
    match = COLOR_TOKEN_RE.search(text)
    return match.group(0) if match else None


def extract_edge_tokens(text: str) -> set[str]:
    return {f"<E{a}_{b}>" for a, b in EDGE_TOKEN_RE.findall(text)}


def extract_numbers_for(label: str, text: str) -> int | None:
    if label in {"SETTLEMENTS", "SETTLEMENT"}:
        m = re.search(r"<(?:SETTLEMENT|CITY)|\bSETTLEMENTS?\b.*?(\d+)|(\d+).*?\bSETTLEMENTS?\b", text)
        if not m:
            return parse_int(text) if text.isdigit() else None
        for value in m.groups():
            if value is not None:
                return int(value)
        return None
    if label in {"ROADS", "ROAD", "CITIES", "CITY"}:
        m = re.search(r"<?(?:CITY|ROAD)S?\>?.*?(\d+)|(\d+).*(?:CITY|ROAD)S?\>?", text)
        if not m:
            return parse_int(text) if text.isdigit() else None
        for value in m.groups():
            if value is not None:
                return int(value)
        return None
    return parse_int(text)
