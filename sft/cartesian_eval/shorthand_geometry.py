"""Exact h spelling and lossless sparse projection of the visible dense prompt."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

from .contracts import digest, require
from .geometry import ATOM, Point, coordinate_points, parse_atom, rational
from .geometry import LEGEND as CARTESIAN_LEGEND

SCHEMA = "catan_cartesian_h_eval/v1"
VERSION = "cartesian_h_eval_v1"
EMPTY_DEFAULT = "Unlisted nodes and edges are empty."
H_MULTIPLE = re.compile(
    r"(?:(?P<coefficient>[+-]?[0-9]+(?:/[+-]?[0-9]+)?)\*?|(?P<sign>[+-]?))"
    r"h(?:/(?P<den>[+-]?[0-9]+))?"
)
DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)")
LEGEND = CARTESIAN_LEGEND.replace(
    "(0,1),(sqrt(3)/2,1/2),(sqrt(3)/2,-1/2),(0,-1),(-sqrt(3)/2,-1/2),(-sqrt(3)/2,1/2).",
    "(0,1),(2h,0.5),(2h,-0.5),(0,-1),(-2h,-0.5),(-2h,0.5).",
).replace(
    "Write x as n*sqrt(3)/d in reduced form "
    "(omit coefficient 1 and denominator 1), y as a reduced rational, and zero as 0.",
    "Here h=sqrt(3)/4. Write x as an integer multiple of h "
    "(h, -h, 2h, etc.), y as an exact decimal (integer, .25, .5, or .75), and zero as 0.",
)


def encode(point: Point) -> str:
    """No floating arithmetic participates in spelling or entity identity."""
    k = 4 * point.x_root
    require(k.denominator == 1 and point.y.denominator in {1, 2, 4}, "off-grid point")
    x = "0" if k == 0 else "h" if k == 1 else "-h" if k == -1 else f"{k.numerator}h"
    quarters = abs(point.y.numerator) * 4 // point.y.denominator
    integer, remainder = divmod(quarters, 4)
    y = ("-" if point.y < 0 else "") + str(integer) + ("", ".25", ".5", ".75")[remainder]
    return f"{point.family}({x},{y})"


def decode(atom: str) -> Point:
    """Bounded exact grammar: h multiples, old radicals, rational/finite-decimal y."""
    require(len(atom) <= 512, "coordinate atom too long")
    match = ATOM.fullmatch(atom)
    require(match is not None, "expected one whitespace-free Cartesian atom")
    assert match is not None
    y_text = match["y"]
    require(len(y_text) <= 128, "y coordinate too long")
    y = Fraction(y_text) if DECIMAL.fullmatch(y_text) else rational(y_text)
    multiple = H_MULTIPLE.fullmatch(match["x"])
    if multiple is None:
        return parse_atom(f"{match['family']}({match['x']},{y})")
    coefficient = rational(multiple["coefficient"] or (multiple["sign"] + "1"))
    denominator = rational(multiple["den"] or "1")
    require(denominator != 0, "zero denominator")
    return Point(match["family"], coefficient / denominator / 4, y)


def coordinate_mapping() -> dict[str, str]:
    return {token: encode(point) for token, point in coordinate_points().items()}


def project_text(value: str, *, reverse: bool = False) -> str:
    """Convert typed addresses only; leave all state values and query prose intact."""
    known = set(coordinate_points().values())

    def replace(match: re.Match[str]) -> str:
        point = decode(match[0]) if reverse else parse_atom(match[0])
        require(point in known, "unknown Cartesian point")
        return point.atom() if reverse else encode(point)

    return ATOM.sub(replace, value)


def root_response(response: str) -> str:
    if response in {"yes", "no", "NONE"} or re.fullmatch(r"[A-Z]+(?:_[A-Z]+)*", response):
        return response
    atoms = response.split()
    require(0 < len(atoms) <= 154, "expected a nonempty bounded entity answer")
    points = [decode(atom) for atom in atoms]
    require(set(points) <= set(coordinate_points().values()), "unknown Cartesian point")
    require(len(points) == len(set(points)), "duplicate entity, including algebraic aliases")
    return " ".join(point.atom() for point in points)


def mapping_artifact() -> dict[str, object]:
    mapping = coordinate_mapping()
    require(len(mapping) == len(set(mapping.values())) == 154, "mapping is not bijective")
    require(all(decode(mapping[t]) == p for t, p in coordinate_points().items()),
            "mapping geometry roundtrip failed")
    return {
        "schema": SCHEMA, "kind": "mapping", "hex_side": 1, "h": "sqrt(3)/4",
        "axes": "ordinary Cartesian, equal units, x right, y up", "convention": LEGEND,
        "inventory_order": list(mapping), "atlas_to_coordinates": mapping,
        "coordinates_to_atlas": {value: key for key, value in mapping.items()},
        "component_counts": dict(Counter(p.family for p in coordinate_points().values())),
        "added_tokenizer_tokens": [],
    }


def mapping_sha256() -> str:
    return digest(mapping_artifact())


def sparse_prompt(dense: str) -> tuple[str, dict[str, int]]:
    """Filter the actual parent board first, then project every surviving address."""
    require(dense.startswith(CARTESIAN_LEGEND + "\n"), "unexpected parent legend")
    body = dense.removeprefix(CARTESIAN_LEGEND + "\n")
    require(EMPTY_DEFAULT not in body, "parent already has an empty default")
    lines = body.split("\n")
    boards = [i for i, line in enumerate(lines) if line.startswith("Board: ")]
    counts = dict.fromkeys(("tiles", "nodes", "edges", "ports", "robber", "omitted_nodes",
                            "omitted_edges", "inventory_entities", "dense_records", "sparse_records"), 0)
    if not boards:
        inventory = "Entity inventory: " + " ".join(p.atom() for p in coordinate_points().values())
        require(lines[0] == inventory, "static topology inventory mismatch")
        counts["inventory_entities"] = 154
    else:
        require(len(boards) == 1 and "Entity inventory:" not in body, "invalid dynamic board")
        index = boards[0]
        clauses = lines[index].removeprefix("Board: ").split("; ")
        records = [clause.split(" ", 1) for clause in clauses]
        require(all(len(record) == 2 for record in records), "malformed board clause")
        facts = dict(records)
        inventory_keys = {point.atom() for point in coordinate_points().values()} | {"robber"}
        require(len(records) == len(facts) == 155 and facts.keys() == inventory_keys,
                "dense board inventory mismatch")
        retained = {key: value for key, value in facts.items()
                    if not (key.startswith(("N(", "E(")) and value == "empty")}
        recovered = {key: retained.get(key, "empty") for key in facts}
        require(recovered == facts, "sparse values cannot recover every parent fact")
        for family, name in (("T(", "tiles"), ("N(", "nodes"), ("E(", "edges"), ("P(", "ports")):
            counts[name] = sum(key.startswith(family) for key in retained)
        counts.update(robber=int("robber" in retained), omitted_nodes=54 - counts["nodes"],
                      omitted_edges=72 - counts["edges"], dense_records=155, sparse_records=len(retained))
        require((counts["tiles"], counts["ports"], counts["robber"]) == (19, 9, 1),
                "tile/port/robber fact lost")
        lines[index] = "Board: " + "; ".join(f"{key} {value}" for key, value in retained.items())
        lines.insert(index + 1, EMPTY_DEFAULT)
    sparse = "\n".join(lines)
    projected = project_text(sparse)
    require(project_text(projected, reverse=True) == sparse, "prompt geometry roundtrip failed")
    return LEGEND + "\n" + projected, counts
