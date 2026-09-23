"""Quota apportionment and deterministic per-cell row drawing."""


from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence, TypeVar

from data_pipeline.board_recognition.mix_rung_data._config import (
    ATLAS_TOKEN_RE,
    WORD_RE,
    JsonDict,
)
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _stable_rank,
)
from data_pipeline.json_coerce import as_dict, as_float, as_int, as_list, as_str
from data_pipeline.json_types import JsonValue


def estimate_tokens(answer: str) -> int:
    """Rough completion tokens: one per atlas token, one per word or separator, one end token."""

    without_tokens = ATLAS_TOKEN_RE.sub(" ", answer)
    return len(ATLAS_TOKEN_RE.findall(answer)) + len(WORD_RE.findall(without_tokens)) + 1


def answer_of(row: JsonDict) -> str:
    return as_str(as_dict(as_list(row["messages"])[1])["content"])


def cell_key(row: JsonDict, balance: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(key)) for key in balance)


CellKey = TypeVar("CellKey", str, tuple[str, ...])


def apportion(weights: dict[CellKey, float], total: int) -> dict[CellKey, int]:
    """Largest-remainder split of ``total`` over ``weights``."""

    mass = sum(weights.values())
    if mass <= 0:
        raise SpatialLocalizationError("apportion needs positive weights")
    raw = {key: total * value / mass for key, value in weights.items()}
    counts = {key: int(value) for key, value in raw.items()}
    for key, _ in sorted(((key, raw[key] - counts[key]) for key in raw), key=lambda item: (-item[1], item[0]))[: total - sum(counts.values())]:
        counts[key] += 1
    return counts


def matches(row: JsonDict, group: JsonDict) -> bool:
    if row["category"] not in as_list(group["categories"]):
        return False
    if group.get("polarity") and row.get("polarity") != group["polarity"]:
        return False
    return True


def ranked(rows: Sequence[JsonDict], salt: str) -> list[JsonDict]:
    return sorted(rows, key=lambda row: (_stable_rank(salt, row["row_id"]), row["row_id"]))


def repeated(row: JsonDict, repeat: int) -> JsonDict:
    """A copy of ``row`` marked as its ``repeat``-th oversample; the first copy is the row itself."""

    if repeat == 0:
        return row
    return {**row, "row_id": f"{row['row_id']}#r{repeat}", "mix_repeat": repeat}


def draw_cells(pool: Sequence[JsonDict], quotas: dict[tuple[str, ...], int], balance: Sequence[str], salt: str, max_repeats: int = 1) -> tuple[list[JsonDict], list[JsonDict]]:
    """Take up to ``quotas[cell]`` rows per balance cell, oversampling a short cell up to ``max_repeats`` times.

    Returns the picks and the untouched remainder in rank order.
    """

    by_cell: dict[tuple[str, ...], list[JsonDict]] = defaultdict(list)
    for row in ranked(pool, salt):
        by_cell[cell_key(row, balance)].append(row)
    picked: list[JsonDict] = []
    remainder: list[JsonDict] = []
    for cell, rows in by_cell.items():
        want = quotas.get(cell, 0)
        picked.extend(rows[:want])
        remainder.extend(rows[want:])
        for repeat in range(1, max_repeats):
            missing = want - len(rows) * repeat
            if missing <= 0 or not rows:
                break
            picked.extend(repeated(row, repeat) for row in rows[:missing])
    return picked, ranked(remainder, salt + ":fill")


def sample_group(group: JsonDict, rows: Sequence[JsonDict]) -> tuple[list[JsonDict], JsonDict]:
    """Draw one group by density weights, balance keys and optional per-key shares."""

    name = as_str(group["name"])
    pool = [row for row in rows if matches(row, group)]
    if not pool:
        raise SpatialLocalizationError(f"group {name} matches no rows")
    balance = [as_str(key) for key in as_list(group.get("balance", []))]
    shares: dict[str, dict[str, float]] = {}
    for key in balance:
        given = group.get(f"{key}_shares")
        if given:
            shares[key] = {str(k): as_float(v) for k, v in as_dict(given).items()}
    by_density: dict[str, list[JsonDict]] = defaultdict(list)
    for row in pool:
        by_density[str(row.get("density_bin"))].append(row)
    # no "density": every bin in the pool, equally
    raw_density = group.get("density")
    requested: dict[str, JsonValue] = (
        as_dict(raw_density) if raw_density else {bin_name: 1 for bin_name in by_density}
    )
    density_weights = {
        bin_name: as_float(weight)
        for bin_name, weight in requested.items()
        if bin_name in by_density
    }
    if not density_weights:
        raise SpatialLocalizationError(f"group {name}: no requested density bin has rows")
    per_bin = apportion(density_weights, as_int(group["rows"]))
    picked_all: list[JsonDict] = []
    leftovers: list[JsonDict] = []
    shortfall = 0
    for bin_name, want in per_bin.items():
        bin_pool = by_density[bin_name]
        if balance:
            cells = sorted({cell_key(row, balance) for row in bin_pool})
            weights: dict[tuple[str, ...], float] = {}
            for cell in cells:
                weight = 1.0
                for index, key in enumerate(balance):
                    if key in shares:
                        weight *= shares[key].get(cell[index], 0.0)
                weights[cell] = weight
            if sum(weights.values()) <= 0:
                weights = {cell: 1.0 for cell in cells}
            quotas = apportion(weights, want)
            picked, remainder = draw_cells(
                bin_pool, quotas, balance, f"{name}:{bin_name}", as_int(group.get("max_repeats", 1))
            )
        else:
            ordered = ranked(bin_pool, f"{name}:{bin_name}")
            picked, remainder = ordered[:want], ordered[want:]
        if len(picked) < want:
            top_up = remainder[: want - len(picked)]
            picked.extend(top_up)
            remainder = remainder[len(top_up):]
        shortfall += max(0, want - len(picked))
        picked_all.extend(picked)
        leftovers.extend(remainder)
    if shortfall and leftovers:
        extra = ranked(leftovers, f"{name}:cross-bin")[:shortfall]
        picked_all.extend(extra)
        shortfall -= len(extra)
    report: JsonDict = {
        "requested": as_int(group["rows"]),
        "taken": len(picked_all),
        "shortfall": shortfall,
        "pool": len(pool),
        "by_density": dict(sorted(Counter(str(row.get("density_bin")) for row in picked_all).items())),
        "by_category": dict(sorted(Counter(as_str(row["category"]) for row in picked_all).items())),
        "estimated_tokens": sum(estimate_tokens(answer_of(row)) for row in picked_all),
        "repeated_rows": sum(1 for row in picked_all if row.get("mix_repeat")),
    }
    for key in balance:
        report[f"by_{key}"] = dict(sorted(Counter(str(row.get(key)) for row in picked_all).items()))
    return picked_all, report


__all__ = ["answer_of", "apportion", "cell_key", "draw_cells", "estimate_tokens", "matches", "ranked", "repeated", "sample_group"]
