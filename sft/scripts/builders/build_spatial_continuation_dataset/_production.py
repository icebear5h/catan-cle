from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable

from data_pipeline.board_recognition.sources import canonical_sha256
from sft.board.spatial_tasks import node_tile_tokens
from sft.json_types import JsonDict, as_bool, as_dict, as_int, as_list, as_str, loads_json

from ._queries import _query
from ._sources import VERSION


def _production_options(contract: JsonDict, rng: random.Random) -> list[JsonDict]:
    options: list[JsonDict] = []
    players = [as_dict(p) for p in as_list(contract["players"])]
    nodes = [as_dict(n) for n in as_list(contract["nodes"])]
    tiles = [as_dict(t) for t in as_list(contract["tiles"])]
    for player in sorted(players, key=lambda p: as_str(p["color"])):
        color = player["color"]
        buildings = [n for n in nodes if n["color"] == color and n["building"]]
        for roll in (2, 3, 4, 5, 6, 8, 9, 10, 11, 12):
            target: JsonDict = {"color": color, "roll": roll}
            query = _query("dice_production", target, contract)
            matching = [t for t in tiles if t["number"] == roll]
            city = any(
                n["building"] == "CITY"
                and not t["has_robber"]
                and t["token"] in node_tile_tokens(as_str(n["token"]))
                for n in buildings
                for t in matching
            )
            robber = any(
                t["has_robber"] and t["token"] in node_tile_tokens(as_str(n["token"]))
                for n in buildings
                for t in matching
            )
            total = sum(as_int(v) for v in as_dict(loads_json(as_str(query["answer"]))).values())
            query["production"] = {
                "zero": total == 0,
                "city_relevant": city,
                "robber_relevant": robber,
                "total": total,
            }
            options.append(query)
    rng.shuffle(options)
    return options


def _select_production(states: list[JsonDict], count: int,
                       load_contract: Callable[[JsonDict], JsonDict],
                       rng: random.Random) -> list[tuple[JsonDict, JsonDict]]:
    selected: list[tuple[JsonDict, JsonDict]] = []
    used: set[str] = set()
    options: dict[str, list[JsonDict]] = {}
    counts: Counter[str] = Counter()
    # Reserve genuinely city/robber-sensitive cases before filling zero/nonzero quotas.
    for criterion, quota in (
        ("city_relevant", count // 8),
        ("robber_relevant", count // 8),
        ("nonzero", count // 2),
        ("zero", count // 2),
    ):
        for state in states:
            if counts[criterion] >= quota:
                break
            sid = as_str(state["state_id"])
            if sid in used:
                continue
            if sid not in options:
                options[sid] = _production_options(load_contract(state), rng)
            for query in options[sid]:
                facts = as_dict(query["production"])
                polarity = "zero" if facts["zero"] else "nonzero"
                qualifies = polarity == criterion or facts.get(criterion, False)
                if not qualifies or counts[polarity] >= count // 2:
                    continue
                selected.append((state, query))
                used.add(sid)
                counts[polarity] += 1
                counts["city_relevant"] += as_bool(facts["city_relevant"])
                counts["robber_relevant"] += as_bool(facts["robber_relevant"])
                break
        if counts[criterion] < quota:
            raise ValueError(
                f"insufficient real production cases: {criterion} {counts[criterion]}/{quota}"
            )
    if len(selected) != count:
        raise AssertionError("production quotas drifted")
    rng.shuffle(selected)
    return selected


def _row(state: JsonDict, family: str, query: JsonDict, split: str) -> JsonDict:
    task = query["task_type"]
    query_hash = canonical_sha256({"task": task, "target": query["target"]})
    row_id = f"{VERSION}/{split}/{family}/{state['state_id']}/{query_hash}"
    metadata: JsonDict = {
        "task_type": task,
        "training_family": family,
        "target": query["target"],
        "split": split,
        "state_id": state["state_id"],
        "layout_id": state["layout_id"],
        "density_bin": state["density_bin"],
        "board_map_sha256": state["board_map_sha256"],
        "source_readout_row_id": state["row_id"],
        "query_sha256": query_hash,
    }
    for key in ("production", "bank_prompt", "tokens", "relationship", "polarity"):
        if key in query:
            metadata[key] = query[key]
    return {
        "schema": "catan_spatial_continuation_row/v1",
        "id": row_id,
        "row_id": row_id,
        "task_type": task,
        "training_family": family,
        "images": state["images"],
        "messages": [
            {"role": "user", "content": "<image>\n" + as_str(query["prompt"])},
            {"role": "assistant", "content": query["answer"]},
        ],
        "metadata": metadata,
    }


def _coverage(rows: list[JsonDict]) -> JsonDict:
    metadata = [as_dict(row["metadata"]) for row in rows]
    return {
        "rows": len(rows),
        "unique_images": len({as_list(r["images"])[0] for r in rows}),
        "unique_image_hashes": len(
            {as_dict(as_dict(m["provenance"])["sha256"])["image"] for m in metadata}
        ),
        "unique_layouts": len({m["layout_id"] for m in metadata}),
        "unique_board_maps": len({m["board_map_sha256"] for m in metadata}),
        "by_density": dict(sorted(Counter(as_str(m["density_bin"]) for m in metadata).items())),
    }
