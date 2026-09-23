"""Deterministic probe-row selection and its immutable selection manifest."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

from sft.json_types import JsonList, as_dict, loads_json

from ._types import (
    PAIR_KINDS,
    PROBE_BEHAVIORS,
    TILE_TASKS,
    JsonDict,
    SelectedProbeRow,
)


def iter_jsonl(path: str | Path) -> Iterable[JsonDict]:
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield as_dict(loads_json(line))


def _stable_key(row: JsonDict, seed: int) -> str:
    identity = str(row.get("row_id") or row.get("id") or row.get("state_id") or row)
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def _balanced_sample(
    rows: Sequence[JsonDict],
    count: int,
    *,
    seed: int,
    used_states: set[str],
    balance_key: Callable[[JsonDict], str],
    behavior_states: set[str] | None = None,
) -> list[JsonDict]:
    """Round-robin stable buckets while keeping states unique when possible."""

    buckets: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        buckets[balance_key(row)].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: _stable_key(row, seed))
    bucket_names = sorted(buckets, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
    selected: list[JsonDict] = []
    selected_ids: set[str] = set()
    local_states = behavior_states if behavior_states is not None else set()

    def draw(*, require_unique: bool) -> None:
        while len(selected) < count:
            progressed = False
            for name in bucket_names:
                bucket = buckets[name]
                while bucket:
                    candidate = bucket.pop(0)
                    state = str(candidate.get("state_id") or candidate.get("row_id"))
                    if state in local_states or (require_unique and state in used_states):
                        continue
                    selected.append(candidate)
                    selected_ids.add(str(candidate.get("row_id") or candidate.get("id")))
                    local_states.add(state)
                    used_states.add(state)
                    progressed = True
                    break
                if len(selected) == count:
                    return
            if not progressed:
                return

    draw(require_unique=True)
    if len(selected) < count:
        # Rebuild the buckets because the first pass consumed rows whose state
        # was already used by an earlier behavior. The fallback permits
        # cross-behavior state reuse but still keeps this behavior internally
        # unique.
        buckets = defaultdict(list)
        for row in rows:
            row_id = str(row.get("row_id") or row.get("id"))
            state = str(row.get("state_id") or row_id)
            if row_id not in selected_ids and state not in local_states:
                buckets[balance_key(row)].append(row)
        for bucket in buckets.values():
            bucket.sort(key=lambda row: _stable_key(row, seed))
        bucket_names = sorted(
            buckets,
            key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
        )
        draw(require_unique=False)
    if len(selected) != count:
        raise ValueError(f"requested {count} probe rows but found only {len(selected)}")
    return selected


def _stratified_sample(
    rows: Sequence[JsonDict],
    strata: Sequence[str],
    *,
    stratum_key: Callable[[JsonDict], str],
    per_stratum: int,
    seed: int,
    balance_key: Callable[[JsonDict], str],
    used_states: set[str] | None = None,
) -> list[JsonDict]:
    selected: list[JsonDict] = []
    states = used_states if used_states is not None else set()
    behavior_states: set[str] = set()
    for index, stratum in enumerate(strata):
        candidates = [row for row in rows if stratum_key(row) == stratum]
        selected.extend(
            _balanced_sample(
                candidates,
                per_stratum,
                seed=seed + index,
                used_states=states,
                balance_key=balance_key,
                behavior_states=behavior_states,
            )
        )
    return selected


def select_probe_rows(
    pair_rows: Sequence[JsonDict],
    single_rows: Sequence[JsonDict],
    *,
    rows_per_behavior: int = 24,
    seed: int = 42,
) -> list[SelectedProbeRow]:
    """Select the four agreed training-source behavior probes."""

    if rows_per_behavior <= 0 or rows_per_behavior % 3:
        raise ValueError("rows_per_behavior must be a positive multiple of three")
    per_stratum = rows_per_behavior // 3
    used_states: set[str] = set()
    pair_train = [row for row in pair_rows if row.get("split", "train") == "train"]
    single_train = [row for row in single_rows if row.get("split", "train") == "train"]

    pair_positive = _stratified_sample(
        [
            row
            for row in pair_train
            if row.get("task_family") == "adjacent_pair_localization"
            and row.get("task_type") == "occupancy_positive"
        ],
        PAIR_KINDS,
        stratum_key=lambda row: str(row.get("pair_kind")),
        per_stratum=per_stratum,
        seed=seed,
        balance_key=lambda row: str(row.get("color")),
        used_states=used_states,
    )
    pair_negative = _stratified_sample(
        [
            row
            for row in pair_train
            if row.get("task_family") == "adjacent_pair_localization"
            and row.get("task_type") == "occupancy_negative_adjacent"
        ],
        PAIR_KINDS,
        stratum_key=lambda row: str(row.get("pair_kind")),
        per_stratum=per_stratum,
        seed=seed + 100,
        balance_key=lambda row: str(row.get("color")),
        used_states=used_states,
    )
    single_road = _balanced_sample(
        [
            row
            for row in single_train
            if row.get("task_family") == "single_piece_localization"
            and row.get("task_type") == "occupancy_positive"
            and str(row.get("piece")).upper() == "ROAD"
        ],
        rows_per_behavior,
        seed=seed + 200,
        used_states=used_states,
        balance_key=lambda row: str(row.get("color")),
        behavior_states=set(),
    )
    tile_anchor = _stratified_sample(
        [row for row in single_train if row.get("task_type") in TILE_TASKS],
        TILE_TASKS,
        stratum_key=lambda row: str(row.get("task_type")),
        per_stratum=per_stratum,
        seed=seed + 300,
        balance_key=lambda row: str(row.get("target_token")),
        used_states=used_states,
    )

    selected: list[SelectedProbeRow] = []
    for behavior, source, rows in (
        ("pair_positive", "pairs", pair_positive),
        ("pair_adjacent_negative", "pairs", pair_negative),
        ("single_road_positive", "singles", single_road),
        ("tile_anchor", "singles", tile_anchor),
    ):
        selected.extend(SelectedProbeRow(behavior=behavior, source=source, row=row) for row in rows)
    return selected


def selection_manifest(
    selected: Sequence[SelectedProbeRow],
    *,
    seed: int,
    rows_per_behavior: int,
) -> JsonDict:
    by_behavior: JsonDict = {}
    for behavior in PROBE_BEHAVIORS:
        indexed = [(index, item) for index, item in enumerate(selected) if item.behavior == behavior]
        if len(indexed) != rows_per_behavior:
            raise ValueError(f"{behavior} has {len(indexed)} rows, expected {rows_per_behavior}")
        indices: JsonList = [index for index, _ in indexed]
        rows = [item for _, item in indexed]
        by_behavior[behavior] = {
            "indices": indices,
            "row_ids": [item.row_id for item in rows],
            "state_ids": [str(item.row.get("state_id")) for item in rows],
            "colors": [str(item.row.get("color")) for item in rows],
            "task_types": [str(item.row.get("task_type")) for item in rows],
            "pair_kinds": [str(item.row.get("pair_kind")) for item in rows],
            "sources": [item.source for item in rows],
        }
    identity_payload = {
        "seed": seed,
        "rows_per_behavior": rows_per_behavior,
        "row_ids": [item.row_id for item in selected],
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "catan_gradient_probe_selection/v1",
        "identity": identity,
        "seed": seed,
        "rows_per_behavior": rows_per_behavior,
        "total_rows": len(selected),
        "unique_states": len({str(item.row.get("state_id")) for item in selected}),
        "cross_behavior_state_reuses": len(selected)
        - len({str(item.row.get("state_id")) for item in selected}),
        "behavior_order": list(PROBE_BEHAVIORS),
        "behaviors": by_behavior,
    }
