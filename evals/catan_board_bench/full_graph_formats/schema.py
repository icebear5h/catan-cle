"""Format names, minimal graph schema, and integrated-diagram glyph tables."""

MINIMAL_SCHEMA = "catan_full_public_graph_minimal/v1"
FORMAT_NAMES = (
    "optimized_html",
    "full_graph_json",
    "datalog",
    "sql_relational",
    "integrated_ascii",
    "tile_rows",
)
FORMAT_EXTENSIONS = {
    "optimized_html": ".html",
    "full_graph_json": ".json",
    "datalog": ".dl",
    "sql_relational": ".sql",
    "integrated_ascii": ".txt",
    "tile_rows": ".txt",
}

_COLOR_TO_CODE = {
    "BLACK": "BK",
    "BLUE": "BL",
    "BRONZE": "BZ",
    "GOLD": "GD",
    "GREEN": "GR",
    "MYSTIC_BLUE": "MB",
    "ORANGE": "OR",
    "PINK": "PK",
    "RED": "RD",
    "SILVER": "SV",
    "WHITE": "WH",
}
_CODE_TO_COLOR = {code: color for color, code in _COLOR_TO_CODE.items()}
_BUILDING_TO_CODE = {"SETTLEMENT": "S", "CITY": "C"}
_CODE_TO_BUILDING = {code: value for value, code in _BUILDING_TO_CODE.items()}
_SIDE_ENDPOINTS = {
    "EAST": ("NORTHEAST", "SOUTHEAST"),
    "SOUTHEAST": ("SOUTHEAST", "SOUTH"),
    "SOUTHWEST": ("SOUTH", "SOUTHWEST"),
    "WEST": ("SOUTHWEST", "NORTHWEST"),
    "NORTHWEST": ("NORTHWEST", "NORTH"),
    "NORTHEAST": ("NORTH", "NORTHEAST"),
}
_EXPECTED_COUNTS = (19, 54, 72, 9)
