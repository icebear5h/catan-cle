"""Exact ordinary Cartesian identity and a deliberately small radical grammar."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from sft.board.coordinate_comparison import coordinate_mapping as integer_mapping

SCHEMA = "catan_cartesian_eval/v1"
ATLAS_ATOM = re.compile(r"<[NTEP][0-9_]+>")
RATIONAL = r"[+-]?[0-9]+(?:/[+-]?[0-9]+)?"
ROOT = r"(?:sqrt\(3\)|√3)"
RADICAL = re.compile(rf"(?P<coefficient>{RATIONAL})\*{ROOT}")
ROOT_FIRST = re.compile(rf"(?P<sign>[+-]?){ROOT}(?:/(?P<den>[+-]?[0-9]+))?")
NUMERATOR_FIRST = re.compile(rf"(?P<num>[+-]?[0-9]+)\*{ROOT}/(?P<den>[+-]?[0-9]+)")
ATOM = re.compile(r"(?P<family>[NTEP])\((?P<x>[^,\s]+),(?P<y>[^,\s()]+)\)")
LEGEND = (
    "Use ordinary 2D Cartesian coordinates on a regular pointy-top hex board with hex side 1. "
    "The axes have equal units: x increases right and y increases up. The central tile is T(0,0). "
    "N(x,y) denotes a node point; T(x,y) a land tile center; E(x,y) a road-edge midpoint; "
    "P(x,y) the midpoint of a port's two attached nodes. A tile's six corner offsets are "
    "(0,1),(sqrt(3)/2,1/2),(sqrt(3)/2,-1/2),(0,-1),(-sqrt(3)/2,-1/2),(-sqrt(3)/2,1/2). "
    "Road edges join consecutive tile corners. Edge and port coordinates are arithmetic "
    "midpoints of their endpoint nodes; a port attaches to the coastal edge at its midpoint. "
    "Entity types distinguish coincident points. Write x as n*sqrt(3)/d in reduced form "
    "(omit coefficient 1 and denominator 1), y as a reduced rational, and zero as 0. "
    "Return only the requested bare answer. Entity sets use whitespace-separated typed atoms; "
    "color answers use the supplied participant names. Use no spaces inside atoms, "
    "no duplicates, and NONE for the empty set."
)


@dataclass(frozen=True)
class Point:
    """Physical X = x_root * sqrt(3); physical Y = y, both in side-length units."""

    family: str
    x_root: Fraction
    y: Fraction

    def atom(self) -> str:
        return f"{self.family}({radical_text(self.x_root)},{self.y})"


def radical_text(coefficient: Fraction) -> str:
    if coefficient == 0:
        return "0"
    numerator, denominator = coefficient.numerator, coefficient.denominator
    prefix = "" if numerator == 1 else "-" if numerator == -1 else f"{numerator}*"
    suffix = "" if denominator == 1 else f"/{denominator}"
    return f"{prefix}sqrt(3){suffix}"


def rational(text: str) -> Fraction:
    if len(text) > 128 or re.fullmatch(RATIONAL, text) is None:
        raise ValueError("expected an exact integer or fraction")
    parts = text.split("/")
    denominator = int(parts[1]) if len(parts) == 2 else 1
    if denominator == 0:
        raise ValueError("zero denominator")
    return Fraction(int(parts[0]), denominator)


def radical(text: str) -> Fraction:
    """Only zero or one rational multiple of sqrt(3), never general algebra."""
    if len(text) > 256:
        raise ValueError("radical too long")
    if re.fullmatch(RATIONAL, text):
        coefficient = rational(text)
        if coefficient == 0:
            return coefficient
    match = ROOT_FIRST.fullmatch(text)
    if match:
        return rational(f"{'-1' if match['sign'] == '-' else '1'}/{match['den'] or '1'}")
    match = RADICAL.fullmatch(text)
    if match:
        return rational(match["coefficient"])
    match = NUMERATOR_FIRST.fullmatch(text)
    if match:
        return rational(f"{match['num']}/{match['den']}")
    raise ValueError("expected zero or a rational multiple of sqrt(3)")


def parse_atom(text: str) -> Point:
    match = ATOM.fullmatch(text)
    if match is None:
        raise ValueError("expected one whitespace-free Cartesian atom")
    return Point(match["family"], radical(match["x"]), rational(match["y"]))


@lru_cache(maxsize=1)
def _points() -> tuple[tuple[str, Point], ...]:
    # Old coordinates are 2*atlas_geometry base coordinates, including midpoints.
    # Thus X = old_x/4 * sqrt(3), Y = -old_y/4, with no floating identity math.
    points = []
    for token, atom in integer_mapping().items():
        x, y = (int(part) for part in atom[2:-1].split(","))
        points.append((token, Point(token[1], Fraction(x, 4), Fraction(-y, 4))))
    if len(points) != 154 or len({point for _, point in points}) != 154:
        raise ValueError("Cartesian mapping is not a 154-entity bijection")
    return tuple(points)


def coordinate_points() -> dict[str, Point]:
    return dict(_points())


def coordinate_mapping() -> dict[str, str]:
    return {token: point.atom() for token, point in _points()}


def project_text(text: str) -> str:
    mapping = coordinate_mapping()

    def replace(match: re.Match[str]) -> str:
        if match[0] not in mapping:
            raise ValueError(f"unknown canonical entity: {match[0]}")
        return mapping[match[0]]

    return ATLAS_ATOM.sub(replace, text)


def canonical_response(text: str) -> str:
    """Parse every atom, bind its exact physical point, and reject alias duplicates.

    Scalar syntax/family/choice membership is checked by the original oracle.
    """
    if text in {"yes", "no", "NONE"} or re.fullmatch(r"[A-Z]+(?:_[A-Z]+)*", text):
        return text
    inverse = {point: token for token, point in _points()}
    atoms = text.split()
    if not atoms:
        raise ValueError("empty response")
    tokens = []
    for atom in atoms:
        point = parse_atom(atom)
        if point not in inverse:
            raise ValueError("unknown Cartesian point")
        tokens.append(inverse[point])
    if len(tokens) != len(set(tokens)):
        raise ValueError("duplicate entity, including algebraic aliases")
    return " ".join(tokens)


def mapping_artifact() -> dict[str, object]:
    mapping = coordinate_mapping()
    return {
        "schema": SCHEMA, "kind": "mapping", "hex_side": 1,
        "axes": "ordinary Cartesian, equal units, x right, y up",
        "convention": LEGEND, "inventory_order": list(mapping),
        "atlas_to_coordinates": mapping,
        "coordinates_to_atlas": {v: k for k, v in mapping.items()},
        "added_tokenizer_tokens": [],
    }
