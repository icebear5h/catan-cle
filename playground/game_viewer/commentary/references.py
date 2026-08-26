"""Deterministically ground human Catan location shorthand against an engine map."""

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from game_engine.models.enums import ActionType


_DICE_NUMBERS = frozenset({2, 3, 4, 5, 6, 8, 9, 10, 11, 12})
_NUMBER_TOKEN = r"(?:1[0-2]|[2-689])"
_SEPARATED_MENTION = re.compile(
    rf"(?<!\d)({_NUMBER_TOKEN}(?:(?:\s*-\s*|\s+){_NUMBER_TOKEN}){{1,2}})(?!\d)"
)
_EIGHT_ALIAS_MENTION = re.compile(r"(?<!\w)[AEae](\d{1,4})(?![\w%])")
_COMPACT_MENTION = re.compile(r"(?<![\d:])\d{2,6}(?![\d:%])")
_DIRECTION_AFTER = re.compile(r"(?:'s)?\s+(up|down|left|right)\b", re.IGNORECASE)
_CORNER_MAP_PATH = Path(__file__).resolve().parents[1] / "corner_to_node_map.json"
_NODE_OFFSETS = {
    "NORTH": (0.0, -1.0),
    "NORTHEAST": (math.sqrt(3) / 2, -0.5),
    "SOUTHEAST": (math.sqrt(3) / 2, 0.5),
    "SOUTH": (0.0, 1.0),
    "SOUTHWEST": (-math.sqrt(3) / 2, 0.5),
    "NORTHWEST": (-math.sqrt(3) / 2, -0.5),
}


@dataclass(frozen=True)
class NumberMention:
    """One raw human number reference with one or more valid normalizations."""

    surface: str
    start: int
    end: int
    number_options: Tuple[Tuple[int, ...], ...]
    syntax: str
    direction: Optional[str] = None


@dataclass(frozen=True)
class TileFact:
    tile_id: int
    number: Optional[int]
    resource: Optional[str]
    coordinate: Tuple[int, int, int]


@dataclass(frozen=True)
class CornerFact:
    colonist_corner_id: int
    engine_node_id: int
    tiles: Tuple[TileFact, ...]
    numbers: Tuple[int, ...]
    ports: Tuple[str, ...]
    coast: bool
    cube_coordinate: Tuple[int, int, int]
    screen_coordinate: Tuple[float, float]


@dataclass(frozen=True)
class CornerState:
    corner: CornerFact
    occupied_by: Optional[str]
    building: Optional[str]
    legal_settlement_now: Optional[bool]


@dataclass(frozen=True)
class GroundingResult:
    mention: NumberMention
    status: str
    candidates: Tuple[CornerState, ...]


@lru_cache(maxsize=1)
def load_colonist_corner_mapping() -> Dict[int, int]:
    """Return Colonist corner ID -> engine node ID."""
    raw = json.loads(_CORNER_MAP_PATH.read_text(encoding="utf-8"))
    return {int(key.removeprefix("_")): int(value) for key, value in raw.items()}


def _valid_partitions(digits: str, pieces: int) -> Tuple[Tuple[int, ...], ...]:
    results = set()

    def visit(offset: int, values: Tuple[int, ...]) -> None:
        remaining = pieces - len(values)
        if remaining == 0:
            if offset == len(digits):
                results.add(values)
            return

        characters_left = len(digits) - offset
        if characters_left < remaining or characters_left > remaining * 2:
            return

        for width in (1, 2):
            token = digits[offset : offset + width]
            if len(token) != width or token.startswith("0"):
                continue
            value = int(token)
            if value in _DICE_NUMBERS:
                visit(offset + width, (*values, value))

    visit(0, ())
    return tuple(sorted(results))


def _compact_options(digits: str) -> Tuple[Tuple[int, ...], ...]:
    """Prefer a three-corner reading, falling back to a two-number region."""
    triples = _valid_partitions(digits, 3)
    return triples or _valid_partitions(digits, 2)


def _separated_numbers(surface: str) -> Tuple[int, ...]:
    return tuple(int(token) for token in re.findall(_NUMBER_TOKEN, surface))


def _overlaps(start: int, end: int, spans: Iterable[Tuple[int, int]]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def _direction_after(text: str, end: int) -> Optional[str]:
    match = _DIRECTION_AFTER.match(text[end : end + 16])
    return match.group(1).lower() if match else None


def extract_number_mentions(text: str) -> Tuple[NumberMention, ...]:
    """Extract separated, compact, and common ASR `eight` number references."""
    mentions = []
    occupied_spans = []

    for match in _SEPARATED_MENTION.finditer(text):
        surface = match.group(1)
        numbers = _separated_numbers(surface)
        mention = NumberMention(
            surface=surface,
            start=match.start(1),
            end=match.end(1),
            number_options=(numbers,),
            syntax="separated",
            direction=_direction_after(text, match.end(1)),
        )
        mentions.append(mention)
        occupied_spans.append((mention.start, mention.end))

    for match in _EIGHT_ALIAS_MENTION.finditer(text):
        if _overlaps(match.start(), match.end(), occupied_spans):
            continue
        options = _compact_options(f"8{match.group(1)}")
        if not options:
            continue
        mention = NumberMention(
            surface=match.group(0),
            start=match.start(),
            end=match.end(),
            number_options=options,
            syntax="asr_eight_alias",
            direction=_direction_after(text, match.end()),
        )
        mentions.append(mention)
        occupied_spans.append((mention.start, mention.end))

    for match in _COMPACT_MENTION.finditer(text):
        if _overlaps(match.start(), match.end(), occupied_spans):
            continue
        options = _compact_options(match.group(0))
        if not options:
            continue
        mention = NumberMention(
            surface=match.group(0),
            start=match.start(),
            end=match.end(),
            number_options=options,
            syntax="compact",
            direction=_direction_after(text, match.end()),
        )
        mentions.append(mention)
        occupied_spans.append((mention.start, mention.end))

    return tuple(sorted(mentions, key=lambda mention: (mention.start, -mention.end)))


def _node_screen_coordinates(catan_map: Any) -> Dict[int, Tuple[float, float]]:
    positions = {}
    for coordinate, tile in catan_map.tiles.items():
        if not hasattr(tile, "nodes"):
            continue
        q = coordinate[0]
        r = coordinate[2]
        tile_x = math.sqrt(3) * (q + r / 2)
        tile_y = 1.5 * r
        for node_ref, node_id in tile.nodes.items():
            if node_id in positions:
                continue
            offset_x, offset_y = _NODE_OFFSETS[node_ref.value]
            positions[node_id] = (tile_x + offset_x, tile_y + offset_y)
    return positions


def _node_cube_coordinates(catan_map: Any) -> Dict[int, Tuple[int, int, int]]:
    coordinates = defaultdict(list)
    for coordinate, tile in catan_map.tiles.items():
        if hasattr(tile, "nodes"):
            for node_id in tile.nodes.values():
                coordinates[node_id].append(coordinate)
    return {
        node_id: tuple(sum(coord[axis] for coord in values) for axis in range(3))
        for node_id, values in coordinates.items()
    }


def _port_labels(catan_map: Any, node_id: int) -> Tuple[str, ...]:
    labels = []
    for resource, nodes in catan_map.port_nodes.items():
        if node_id in nodes:
            labels.append("3:1" if resource is None else f"2:1 {resource}")
    return tuple(sorted(labels))


def build_corner_index(catan_map: Any) -> Tuple[CornerFact, ...]:
    """Build the canonical 54-corner index for one randomized board layout."""
    colonist_to_engine = load_colonist_corner_mapping()
    engine_to_colonist = {
        engine_node: colonist_corner
        for colonist_corner, engine_node in colonist_to_engine.items()
    }
    screen_coordinates = _node_screen_coordinates(catan_map)
    cube_coordinates = _node_cube_coordinates(catan_map)
    corners = []

    for engine_node_id in sorted(catan_map.land_nodes):
        tile_entries = []
        for tile in catan_map.adjacent_tiles[engine_node_id]:
            coordinate = next(
                coord for coord, candidate in catan_map.land_tiles.items() if candidate is tile
            )
            tile_entries.append(
                TileFact(
                    tile_id=tile.id,
                    number=tile.number,
                    resource=tile.resource,
                    coordinate=coordinate,
                )
            )
        tiles = tuple(sorted(tile_entries, key=lambda tile: tile.tile_id))
        numbers = tuple(sorted(tile.number for tile in tiles if tile.number is not None))
        corners.append(
            CornerFact(
                colonist_corner_id=engine_to_colonist[engine_node_id],
                engine_node_id=engine_node_id,
                tiles=tiles,
                numbers=numbers,
                ports=_port_labels(catan_map, engine_node_id),
                coast=len(tiles) < 3,
                cube_coordinate=cube_coordinates[engine_node_id],
                screen_coordinate=screen_coordinates[engine_node_id],
            )
        )

    return tuple(sorted(corners, key=lambda corner: corner.colonist_corner_id))


def _counter_contains(container: Sequence[int], required: Sequence[int]) -> bool:
    available = Counter(container)
    return all(available[number] >= count for number, count in Counter(required).items())


def _matches(corner: CornerFact, numbers: Sequence[int]) -> bool:
    if len(numbers) == 3:
        return Counter(corner.numbers) == Counter(numbers)
    if len(numbers) == 2:
        return _counter_contains(corner.numbers, numbers)
    return False


def _color_name(color: Any) -> str:
    if hasattr(color, "value"):
        return str(color.value)
    if hasattr(color, "name"):
        return str(color.name)
    return str(color)


def _corner_state(
    corner: CornerFact,
    game_state: Optional[Any],
    include_legality: bool,
) -> CornerState:
    if game_state is None:
        return CornerState(corner, None, None, None)

    building = game_state.board.buildings.get(corner.engine_node_id)
    occupied_by = _color_name(building[0]) if building else None
    building_type = str(building[1]) if building else None
    legal_settlement_now = None
    if include_legality:
        legal_nodes = {
            action.value
            for action in game_state.playable_actions
            if action.action_type == ActionType.BUILD_SETTLEMENT
            and isinstance(action.value, int)
        }
        legal_settlement_now = corner.engine_node_id in legal_nodes
    return CornerState(
        corner=corner,
        occupied_by=occupied_by,
        building=building_type,
        legal_settlement_now=legal_settlement_now,
    )


def resolve_corner_mention(
    corner_index: Sequence[CornerFact],
    mention: NumberMention,
    game_state: Optional[Any] = None,
    *,
    include_legality: bool = True,
) -> GroundingResult:
    """Resolve a number mention without using direction, occupancy, or future events."""
    matched = {
        corner
        for option in mention.number_options
        for corner in corner_index
        if _matches(corner, option)
    }
    candidates = tuple(
        _corner_state(corner, game_state, include_legality)
        for corner in sorted(matched, key=lambda item: item.colonist_corner_id)
    )
    status = "none" if not candidates else ("unique" if len(candidates) == 1 else "ambiguous")
    return GroundingResult(mention=mention, status=status, candidates=candidates)


def ground_text_references(
    text: str,
    corner_index: Sequence[CornerFact],
    game_state: Optional[Any] = None,
    *,
    include_legality: bool = True,
) -> Tuple[GroundingResult, ...]:
    """Extract and ground every number mention in source order."""
    return tuple(
        resolve_corner_mention(
            corner_index,
            mention,
            game_state,
            include_legality=include_legality,
        )
        for mention in extract_number_mentions(text)
    )
