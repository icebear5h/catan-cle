from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from typing import TypeAlias

from data_pipeline.board_recognition.spatial_robber import spatial_query_bank
from sft.board.spatial_tasks import (
    atlas_node_graph,
    dice_production,
    local_node_tiles,
    node_tile_tokens,
    shortest_node_path,
)
from sft.json_types import JsonDict, as_int, as_list, as_str

from ._sources import BANK_QUOTAS, STATIC_GUIDANCE

# A canonical unordered node pair, stored in sorted order.
Pair: TypeAlias = tuple[str, str]


def _pair(a: str, b: str) -> Pair:
    """`tuple(sorted((a, b)))` with a fixed arity."""
    return (a, b) if a <= b else (b, a)


def _varied_states(rows: list[JsonDict], rng: random.Random) -> list[JsonDict]:
    """Cycle densities, choosing the least-used available board map each time."""
    groups: defaultdict[str, defaultdict[str, list[JsonDict]]] = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda r: as_str(r["state_id"])):
        groups[as_str(row["density_bin"])][as_str(row["board_map_sha256"])].append(row)
    densities = sorted(groups)
    rng.shuffle(densities)
    maps = sorted({as_str(r["board_map_sha256"]) for r in rows})
    rng.shuffle(maps)
    rank = {key: index for index, key in enumerate(maps)}
    counts: Counter[str] = Counter()
    for buckets in groups.values():
        for bucket in buckets.values():
            rng.shuffle(bucket)
    result: list[JsonDict] = []
    while any(groups.values()):
        for density in densities:
            buckets = groups[density]
            if not buckets:
                continue
            key = min(buckets, key=lambda k: (counts[k], rank[k]))
            result.append(buckets[key].pop())
            counts[key] += 1
            if not buckets[key]:
                del buckets[key]
    return result


def _path_pairs(seed: int) -> tuple[dict[str, list[Pair]], dict[Pair, JsonDict]]:
    """Partition unordered endpoints, not rendered routes or their reversals."""
    graph = atlas_node_graph()
    nodes = sorted(graph)
    distances = {pair: len(shortest_node_path(*pair)) - 1 for pair in combinations(nodes, 2)}
    info: dict[Pair, JsonDict] = {}
    for end in nodes:
        distance = {
            node: distances[_pair(node, end)] if node != end else 0 for node in nodes
        }
        counts = {end: 1}
        # Count shortest-route multiplicity for coverage, without another path oracle.
        for node in sorted(nodes, key=lambda n: (distance[n], n)):
            if node != end:
                counts[node] = sum(
                    counts[n] for n in graph[node] if distance[n] == distance[node] - 1
                )
            if node < end:
                info[(node, end)] = {
                    "distance": distance[node],
                    "shortest_path_count": counts[node],
                }
    strata: defaultdict[tuple[int, bool], list[Pair]] = defaultdict(list)
    for pair, facts in sorted(info.items()):
        strata[(as_int(facts["distance"]), as_int(facts["shortest_path_count"]) > 1)].append(pair)
    rng = random.Random(seed)
    membership: dict[str, list[Pair]] = {split: [] for split in ("train", "validation", "test")}
    for pairs in strata.values():
        rng.shuffle(pairs)
        if len(pairs) < 3:
            raise ValueError("path stratum cannot cover all three splits")
        heldout = max(1, len(pairs) // 5)
        membership["validation"].extend(pairs[:heldout])
        membership["test"].extend(pairs[heldout : 2 * heldout])
        membership["train"].extend(pairs[2 * heldout :])
    return {split: sorted(pairs) for split, pairs in membership.items()}, info


def _select_paths(
    pairs: list[Pair], info: dict[Pair, JsonDict], count: int, rng: random.Random,
) -> list[JsonDict]:
    strata: defaultdict[tuple[int, bool], list[Pair]] = defaultdict(list)
    for pair in pairs:
        facts = info[pair]
        strata[(as_int(facts["distance"]), as_int(facts["shortest_path_count"]) > 1)].append(pair)
    keys = sorted(strata)
    rng.shuffle(keys)
    for bucket in strata.values():
        rng.shuffle(bucket)
    result: list[JsonDict] = []
    while len(result) < count:
        if not any(strata.values()):
            raise ValueError("insufficient distinct path endpoints")
        for key in keys:
            if strata[key] and len(result) < count:
                start, end = strata[key].pop()
                if rng.randrange(2):
                    start, end = end, start
                result.append({"start": start, "end": end})
    return result


def _query(task: str, target: JsonDict, contract: JsonDict) -> JsonDict:
    if task == "node_tiles":
        prompt = (
            STATIC_GUIDANCE + f"Which land tiles touch node {target['node']}? "
            "Output only the touching tile tokens, separated by spaces, in any order. "
            "Include every touching tile exactly once; no explanation."
        )
        answer = " ".join(node_tile_tokens(as_str(target["node"])))
    elif task == "shortest_node_path":
        prompt = (
            STATIC_GUIDANCE + f"Give a shortest node path from {target['start']} "
            f"to {target['end']} along atlas edges. Output only node tokens separated "
            "by spaces, in route order, including both endpoints. Any equally short "
            "valid path is accepted; no explanation."
        )
        answer = " ".join(shortest_node_path(as_str(target["start"]), as_str(target["end"])))
    elif task == "local_node_tiles":
        prompt = (
            f"Read the land tiles touching node {target['node']} in this board image. "
            "Output only a JSON object mapping every touching tile token to an object "
            'with exactly the keys "resource" and "number". Use lowercase resource '
            'names, "desert" for desert, integer dice numbers, and null for the '
            "desert number. Include no other tiles. No explanation."
        )
        answer = json.dumps(local_node_tiles(contract, as_str(target["node"])), sort_keys=True)
    elif task == "dice_production":
        color = as_str(target["color"]).lower().replace("_", " ")
        prompt = (
            f"For the {color} player, what resources are produced on a dice roll of "
            f"{target['roll']} on this board? Count one per settlement and two per city "
            "on matching tiles; the robber blocks its tile. Ignore bank shortages. "
            'Output only a JSON object with exactly the five keys "wood", "brick", '
            '"sheep", "wheat", "ore", and integer counts, including zeros. No explanation.'
        )
        answer = json.dumps(dice_production(contract, as_str(target["color"]), as_int(target["roll"])))
    else:
        raise ValueError(f"unknown new task: {task}")
    return {"task_type": task, "target": target, "prompt": prompt, "answer": answer}


def _bank_queries(rng: random.Random) -> dict[str, list[JsonDict]]:
    bank = spatial_query_bank()
    selected: dict[str, list[JsonDict]] = {"directions": [], "adjacency_connectivity": []}
    for task, quota in BANK_QUOTAS.items():
        pool = bank[task]
        if task.endswith("_token"):
            pools = [
                [q for q in pool if q["answer"] == as_list(q["tokens"])[position]] for position in (0, 1)
            ]
        else:
            pools = [pool]
        chosen: list[JsonDict] = []
        for bucket in pools:
            bucket = list(bucket)
            rng.shuffle(bucket)
            needed = quota // len(pools)
            if len(bucket) < needed:
                raise ValueError(f"insufficient corrected bank queries: {task}")
            chosen.extend(bucket[:needed])
        family = "directions" if "direction" in task else "adjacency_connectivity"
        for source in chosen:
            guidance = (
                "Output only one of the two listed tokens; no explanation."
                if task.endswith("_token")
                else "Answer only yes or no; no explanation."
            )
            selected[family].append(
                {
                    **source,
                    "task_type": task,
                    "target": {"tokens": source["tokens"], "relationship": source["relationship"]},
                    "prompt": STATIC_GUIDANCE + as_str(source["prompt"]) + " " + guidance,
                    "bank_prompt": source["prompt"],
                }
            )
    # Distribute each subtype across the 16 batches instead of clustering its quota.
    for family, queries in selected.items():
        buckets: defaultdict[str, list[JsonDict]] = defaultdict(list)
        for query in queries:
            key = as_str(query["task_type"])
            if key.endswith("_token"):
                key += str(as_list(query["tokens"]).index(query["answer"]))
            buckets[key].append(query)
        keys = sorted(buckets)
        rng.shuffle(keys)
        ordered: list[JsonDict] = []
        while any(buckets.values()):
            for key in keys:
                if buckets[key]:
                    ordered.append(buckets[key].pop())
        selected[family] = ordered
    return selected
