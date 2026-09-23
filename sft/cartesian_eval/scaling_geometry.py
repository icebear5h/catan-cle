"""Exact factor-four projections of the immutable sparse-h coordinate panel."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction
from functools import lru_cache

from .contracts import digest, require
from .geometry import ATOM, Point, coordinate_points, radical, rational
from .shorthand_geometry import DECIMAL, H_MULTIPLE, LEGEND
from .shorthand_geometry import decode as parent_decode
from .shorthand_geometry import encode as parent_encode

SCHEMA = "catan_cartesian_scaled_eval/v1"
VERSION = "cartesian_scaling_eval_v1"
VARIANTS = ("scaled_h", "integer_xy")
CORNER_OFFSETS = "(0,1),(2h,0.5),(2h,-0.5),(0,-1),(-2h,-0.5),(-2h,0.5)."
WRITE_RULE = (
    "Here h=sqrt(3)/4. Write x as an integer multiple of h "
    "(h, -h, 2h, etc.), y as an exact decimal (integer, .25, .5, or .75), and zero as 0."
)


def check_variant(variant: str) -> None:
    require(variant in VARIANTS, "unknown scaling variant")


def legend(variant: str) -> str:
    """Byte-identical to the headers used in count_integer_scaling.py."""
    check_variant(variant)
    require(CORNER_OFFSETS in LEGEND and WRITE_RULE in LEGEND, "parent legend changed")
    result = LEGEND.replace("hex side 1.", "hex side 4.")
    if variant == "scaled_h":
        return result.replace(
            CORNER_OFFSETS, "(0,4),(2h,2),(2h,-2),(0,-4),(-2h,-2),(-2h,2).",
        ).replace(WRITE_RULE, "Here h=sqrt(3). Write x as an integer multiple of h and y as an integer.")
    return result.replace(
        "Use ordinary 2D Cartesian coordinates", "Use integer 2D grid coordinates",
    ).replace(
        "The axes have equal units: x increases right and y increases up.",
        "Physical Cartesian position is (sqrt(3)*x,y); x increases right and y increases up.",
    ).replace(
        CORNER_OFFSETS, "(0,4),(2,2),(2,-2),(0,-4),(-2,-2),(-2,2).",
    ).replace(WRITE_RULE, "Write both coordinate components as integers.")


def encode(point: Point, variant: str) -> str:
    """A Point always denotes the original side-one geometry, never scaled units."""
    check_variant(variant)
    k, y = 4 * point.x_root, 4 * point.y
    require(k.denominator == y.denominator == 1, "off-grid point")
    x = str(k.numerator)
    if variant == "scaled_h":
        x = "0" if k == 0 else "h" if k == 1 else "-h" if k == -1 else f"{k.numerator}h"
    return f"{point.family}({x},{y.numerator})"


def integer(value: str) -> Fraction:
    require(len(value) <= 128, "integer coordinate too long")
    result = Fraction(value) if DECIMAL.fullmatch(value) else rational(value)
    require(result.denominator == 1, "expected an exact integer coordinate")
    return result


def decode(atom: str, variant: str) -> Point:
    """Bounded grammar, exact inverse scaling; radicals are in NEW physical units."""
    check_variant(variant)
    require(len(atom) <= 512, "coordinate atom too long")
    match = ATOM.fullmatch(atom)
    require(match is not None, "expected one whitespace-free scaled coordinate atom")
    assert match is not None
    x_text, y = match["x"], integer(match["y"])
    require(len(x_text) <= 256, "x coordinate too long")
    if variant == "integer_xy":
        k = integer(x_text)
    elif multiple := H_MULTIPLE.fullmatch(x_text):
        coefficient = rational(multiple["coefficient"] or (multiple["sign"] + "1"))
        denominator = rational(multiple["den"] or "1")
        require(denominator != 0, "zero denominator")
        k = coefficient / denominator
    elif DECIMAL.fullmatch(x_text):
        k = integer(x_text)
        require(k == 0, "expected zero or a multiple of h or sqrt(3)")
    else:
        k = radical(x_text)
    require(k.denominator == 1, "off-grid x coordinate")
    return Point(match["family"], k / 4, y / 4)


def coordinate_mapping(variant: str) -> dict[str, str]:
    return {token: encode(point, variant) for token, point in coordinate_points().items()}


def project_text(value: str, variant: str, *, reverse: bool = False) -> str:
    """Project typed parent-h addresses only, leaving state facts and prose intact."""
    check_variant(variant)
    known = set(coordinate_points().values())

    def replace(match: re.Match[str]) -> str:
        point = decode(match[0], variant) if reverse else parent_decode(match[0])
        require(point in known, "unknown Cartesian point")
        return parent_encode(point) if reverse else encode(point, variant)

    return ATOM.sub(replace, value)


def parent_response(response: str, variant: str) -> str:
    check_variant(variant)
    require(len(response) <= 154 * 513, "entity answer too long")
    if response in {"yes", "no", "NONE"} or re.fullmatch(r"[A-Z]+(?:_[A-Z]+)*", response):
        return response
    atoms = response.split()
    require(0 < len(atoms) <= 154, "expected a nonempty bounded entity answer")
    points = [decode(atom, variant) for atom in atoms]
    require(set(points) <= set(coordinate_points().values()), "unknown Cartesian point")
    require(len(points) == len(set(points)), "duplicate entity, including algebraic aliases")
    return " ".join(parent_encode(point) for point in points)


def scaling_prompt(parentuser: str, variant: str) -> str:
    require(parentuser.startswith(LEGEND + "\n"), "unexpected sparse-h parent legend")
    body = parentuser.removeprefix(LEGEND + "\n")
    projected = project_text(body, variant)
    require(project_text(projected, variant, reverse=True) == body, "prompt roundtrip failed")
    return legend(variant) + "\n" + projected


def mapping_artifact(variant: str) -> dict[str, object]:
    mapping = coordinate_mapping(variant)
    require(len(mapping) == len(set(mapping.values())) == 154, "mapping is not bijective")
    require(all(decode(mapping[t], variant) == p for t, p in coordinate_points().items()),
            "mapping geometry roundtrip failed")
    return {
        "schema": SCHEMA, "kind": "mapping", "variant": variant, "representation": variant,
        "hex_side": 4, "physical_scale": 4,
        "axes": ("ordinary Cartesian, equal units, x right, y up" if variant == "scaled_h"
                 else "integer grid addresses; physical Cartesian position (sqrt(3)*x,y)"),
        "convention": legend(variant), "inventory_order": list(mapping),
        "atlas_to_coordinates": mapping,
        "coordinates_to_atlas": {value: key for key, value in mapping.items()},
        "component_counts": dict(Counter(p.family for p in coordinate_points().values())),
        "added_tokenizer_tokens": [],
    }


@lru_cache(maxsize=2)
def mapping_sha256(variant: str) -> str:
    return digest(mapping_artifact(variant))
