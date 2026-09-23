from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable

from data_pipeline.board_recognition.sources import canonical_sha256
from sft.board.symbolic_board_tasks import atlas_geometry
from sft.json_types import JsonDict, as_dict, as_int, as_list, as_str
from sft.scripts.builders import build_board_fluency_review as build_board_fluency_review

from ._selection import apportion, quotas, roster_positions
from ._sources import ATLAS_PATTERN, SCHEMA, VERSION, Donor, Exposure, Profile, meta

review = build_board_fluency_review


def make_rows(selected: list[review.Candidate[Donor]], split: str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    counts: Counter[str] = Counter()
    atlas = atlas_geometry()
    for c in selected:
        d, op = c.donor, c.operation
        q: JsonDict = {"operation": op, **c.query}
        text = review.question(op, c.query)
        gold = review.answer(review.Facts(d.data, atlas), op, c.query)
        row_id = f"{VERSION}/{split}/{op}/{counts[op]:04d}"
        counts[op] += 1
        declarations: JsonDict = {"schema": SCHEMA, "split": split,
                        "task_role": "train" if split == "train" else "component_eval",
                        "review_only": False, "admitted_for_training": split == "train"}
        metadata: JsonDict = {**declarations, "class": "board_fluency", "family": review.OPERATION_FAMILY[op],
                    "operation": op, "target": {"state": d.state, "query": q},
                    "question": text, "answer": gold, "state_id": d.provenance["state_id"],
                    "state_sha256": d.state_hash, "query_sha256": canonical_sha256(q),
                    "terrain_sha256": d.terrain_hash, "provenance": d.provenance,
                    "selection_stratum": c.stratum, "queried_roster_positions": list(roster_positions(c)),
                    "row_position": len(rows)}
        rows.append({**declarations, "id": row_id, "row_id": row_id,
                     "messages": [{"role": "user", "content": review.prompt(d.state, text)},
                                  {"role": "assistant", "content": gold}], "metadata": metadata})
    return rows


def queried_positions(m: JsonDict) -> tuple[int, ...]:
    return tuple(as_int(position) for position in as_list(m["queried_roster_positions"]))


def validation_subset(rows: list[JsonDict]) -> list[JsonDict]:
    """A fixed stratified subset, retaining original validation IDs and row bytes."""
    chosen: list[JsonDict] = []
    uses: Counter[Hashable] = Counter()
    for op, count in quotas("validation_eval").items():
        options = [r for r in rows if meta(r)["operation"] == op]
        groups = Counter(as_str(meta(r)["selection_stratum"]) for r in options)
        targets = apportion(groups, count)
        for cell, n in targets.items():
            cell_rows = [r for r in options if meta(r)["selection_stratum"] == cell]
            for _ in range(n):
                row = min(cell_rows, key=lambda r: (
                    uses[op, cell, queried_positions(meta(r))],
                    uses[as_str(meta(r)["state_sha256"])], as_int(meta(r)["row_position"])))
                cell_rows.remove(row)
                chosen.append(row)
                uses[as_str(meta(row)["state_sha256"])] += 1
                uses[op, cell, queried_positions(meta(row))] += 1
    return sorted(chosen, key=lambda r: as_int(meta(r)["row_position"]))


def profile(rows: list[JsonDict]) -> Profile:
    metadata = [meta(r) for r in rows]
    provenance = [as_dict(m["provenance"]) for m in metadata]
    sources = [as_dict(p["source"]) for p in provenance]
    states = Counter(as_str(m["state_sha256"]) for m in metadata)
    cells: defaultdict[str, Counter[str]] = defaultdict(Counter)
    roster: Counter[tuple[str, str, tuple[int, ...]]] = Counter()
    for m in metadata:
        op, stratum = as_str(m["operation"]), as_str(m["selection_stratum"])
        cells[op][stratum] += 1
        roster[op, stratum, queried_positions(m)] += 1
    return {"rows": len(rows), "unique_states": len(states),
            "unique_state_ids": len({m["state_id"] for m in metadata}),
            "unique_maps": len({as_str(p["board_map_sha256"]) for p in provenance}),
            "unique_terrains": len({m["terrain_sha256"] for m in metadata}),
            "unique_games": len({as_str(s["game_id"]) for s in sources if s["game_id"] is not None}),
            "unique_trajectories": len({as_str(s["trajectory_id"]) for s in sources}),
            "max_presentations_per_state": max(states.values()),
            "state_reuse_histogram": dict(sorted(Counter(states.values()).items())),
            "by_operation": dict(Counter(as_str(m["operation"]) for m in metadata)),
            "by_family": dict(Counter(as_str(m["family"]) for m in metadata)),
            "by_density": dict(Counter(as_str(p["density_bin"]) for p in provenance)),
            "by_source_kind": dict(Counter(as_str(s["kind"]) for s in sources)),
            "by_stratum": {op: dict(values) for op, values in cells.items()},
            "roster_crosstabs": [{"operation": op, "stratum": cell, "positions": list(pos), "rows": n}
                                 for (op, cell, pos), n in sorted(roster.items())]}


def exposure(rows: list[JsonDict]) -> Exposure:
    by_operation: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_family: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_step: defaultdict[int, Counter[str]] = defaultdict(Counter)
    atlas_occurrences: dict[str, Counter[str]] = {role: Counter() for role in ("prompt", "completion")}
    for position, row in enumerate(rows):
        m = meta(row)
        counts: Counter[str] = Counter(presentations=1)
        for role, message in zip(("prompt", "completion"), as_list(row["messages"]), strict=True):
            content = as_str(as_dict(message)["content"])
            tokens = ATLAS_PATTERN.findall(content)
            atlas_occurrences[role].update(tokens)
            counts[role + "_atlas_tokens"] += len(tokens)
            counts[role + "_utf8_bytes"] += len(content.encode())
        by_operation[as_str(m["operation"])].update(counts)
        by_family[as_str(m["family"])].update(counts)
        by_step[position // 8 + 1].update(counts)
    totals: Counter[str] = Counter()
    for counts in by_operation.values():
        totals.update(counts)
    return {"totals": dict(totals), "by_operation": dict(by_operation), "by_family": dict(by_family),
            "by_planned_step_at_batch_8": dict(by_step), "atlas_token_occurrences": atlas_occurrences,
            "model_token_counts": None,
            "token_scope": "Exact canonical atlas-token occurrences in message text; UTF-8 bytes are not tokens. "
                           "Full prompt/completion/EOT token counts require the saved checkpoint tokenizer in launcher "
                           "preflight. No tokenizer, model, or chat template is substituted here."}
