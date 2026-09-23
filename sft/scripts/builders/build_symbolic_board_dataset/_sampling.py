from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import TypeAlias, cast

from data_pipeline.board_recognition.sources import canonical_sha256
from sft.board.symbolic_board_tasks import (
    STATIC_TASKS,
    atlas_geometry,
    decode_state,
    symbolic_answer,
    symbolic_prompt,
    symbolic_task_role,
)
from sft.board.symbolic_board_tasks._types import Atlas, DecodedState
from sft.json_types import JsonDict, JsonLikeDict, JsonValue, as_dict, as_str

from ._components import _component_options, _route_query, component_slots, static_queries
from ._sources import DENSITIES, VERSION, _check
from ._types import QueryDict, SlotDict, SourceRecord

# (task, mode, roster position, polarity) identifies one conditional support cell.
CellKey: TypeAlias = tuple[str, str | int | None, str | int | None, str | int | None]


class ComponentSampler:
    """Shared state/map usage across all dynamic families; no per-family cursor reset."""

    def __init__(self, records: list[SourceRecord], rng: random.Random, atlas: Atlas) -> None:
        self.records = sorted(records, key=lambda r: as_str(r["provenance"]["state_id"]))
        rng.shuffle(self.records)
        self.data: list[DecodedState] = [decode_state(r["state"]) for r in self.records]
        self.rng, self.atlas = rng, atlas
        self.pools: dict[CellKey, list[int]] = {}
        self.cells: dict[CellKey, JsonLikeDict] = {}
        self.state_uses: Counter[int] = Counter()
        self.map_uses: Counter[str] = Counter()
        self.density_uses: Counter[tuple[CellKey, str]] = Counter()

    def choose(self, task: str, slot: SlotDict) -> tuple[SourceRecord, QueryDict]:
        key: CellKey = (task, slot["mode"], slot["position"], slot["polarity"])
        if key not in self.pools:
            pool: list[int] = []
            for i, data in enumerate(self.data):
                if task in ("symbolic_reachable", "symbolic_shortest_route"):
                    supported = (slot["mode"] != "route" or
                                 data["colors"][cast("int", slot["position"])]
                                 in data["roads"].values())
                elif task in ("symbolic_near", "symbolic_near_nodes", "symbolic_scene_tiles"):
                    supported = True  # Every atlas selector has touching and non-touching nodes.
                else:
                    supported = bool(_component_options(task, data, slot, self.atlas))
                if supported:
                    pool.append(i)
            _check(bool(pool), f"unsupported component cell: {key}")
            counts = Counter(as_str(self.records[i]["provenance"]["density_bin"]) for i in pool)
            self.cells[key] = dict(task_type=task, **slot,
                                   eligible_states_by_density={d: counts[d] for d in DENSITIES},
                                   selected_by_density=dict.fromkeys(DENSITIES, 0))
            # Occupied-board negatives are the first choice, not empty-board shortcuts.
            nonempty = [i for i in pool
                        if self.records[i]["provenance"]["density_bin"] != "empty"]
            self.pools[key] = nonempty or pool
        pool = self.pools[key]

        def rank(i: int) -> tuple[int, int, int, int]:
            p = self.records[i]["provenance"]
            return (self.state_uses[i], self.density_uses[key, as_str(p["density_bin"])],
                    self.map_uses[as_str(p["board_map_sha256"])], i)

        index = min(pool, key=rank)
        record, data = self.records[index], self.data[index]
        if task in ("symbolic_reachable", "symbolic_shortest_route"):
            query = _route_query(record["state"], slot, self.rng, self.atlas)
        else:
            query = self.rng.choice(_component_options(task, data, slot, self.atlas))
        p = record["provenance"]
        density = as_str(p["density_bin"])
        self.state_uses[index] += 1
        self.map_uses[as_str(p["board_map_sha256"])] += 1
        self.density_uses[key, density] += 1
        cast("dict[str, int]", self.cells[key]["selected_by_density"])[density] += 1
        return record, query

    def report(self) -> JsonLikeDict:
        return {"conditional_support": list(self.cells.values()),
                "unique_states": len(self.state_uses), "unique_maps": len(self.map_uses),
                "max_presentations_per_state": max(self.state_uses.values(), default=0),
                "policy": "Global least-used state, then conditional density and map usage; nonempty boards preferred where supported. Exact mode/roster/polarity slots. Empty conditional-support cells are genuine population limitations."}


def _ordered_sources(records: list[SourceRecord], rng: random.Random) -> list[SourceRecord]:
    buckets: defaultdict[tuple[str, str], list[SourceRecord]] = defaultdict(list)
    for record in records:
        p = record["provenance"]
        buckets[(as_str(as_dict(p["source"])["kind"]), as_str(p["density_bin"]))].append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: as_str(r["provenance"]["state_id"]))
        rng.shuffle(bucket)
    result: list[SourceRecord] = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                result.append(buckets[key].pop())
    return result


def _row(task: str, query: QueryDict, record: SourceRecord | None, split: str,
         index: int) -> JsonDict:
    target = cast("JsonDict", {"state": record["state"] if record else None, "query": query})
    rid = f"{VERSION}/{split}/{task}/{index:05d}"
    role = symbolic_task_role(task, split)
    metadata: JsonDict = {"task_type": task, "training_family": task, "task_role": role,
                          "target": target, "split": split,
                          "query_sha256": canonical_sha256(query)}
    if record:
        metadata["provenance"] = record["provenance"]
    if task in ("symbolic_direction", "symbolic_direction_choice"):
        metadata["directional_pair"] = cast(
            "JsonValue", sorted([as_str(query["a"]), as_str(query["b"])]))
    return {"schema": "catan_symbolic_board_row/v2", "id": rid, "row_id": rid,
            "task_type": task, "training_family": task, "task_role": role, "split": split,
            "messages": [{"role": "user", "content": symbolic_prompt(task, target)},
                         {"role": "assistant", "content": symbolic_answer(task, target)}],
            "metadata": metadata}


def component_rows(records: list[SourceRecord], split: str, pairs: list[list[str]],
                   quotas: dict[str, int],
                   seed: int) -> tuple[list[JsonDict], JsonLikeDict]:
    rng, atlas = random.Random(seed), atlas_geometry()
    sampler = ComponentSampler(records, rng, atlas)
    families: dict[str, list[JsonDict]] = {}
    for task, count in quotas.items():
        family: list[JsonDict] = []
        if task in STATIC_TASKS:
            family = [_row(task, query, None, split, i) for i, query in enumerate(static_queries(task, pairs, count, rng))]
        else:
            for i, slot in enumerate(component_slots(task, count, rng)):
                record, query = sampler.choose(task, slot)
                row = _row(task, query, record, split, i)
                as_dict(row["metadata"])["sampling_cell"] = cast("JsonValue", slot)
                family.append(row)
        families[task] = family
    # Deterministic family round-robin presentations, not a claimed optimizer schedule.
    result: list[JsonDict] = []
    for i in range(max(quotas.values())):
        for task in quotas:
            if i < len(families[task]):
                result.append(families[task][i])
    return result, sampler.report()
