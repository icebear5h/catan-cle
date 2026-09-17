"""Build the approved, admitted symbolic board-fluency SFT corpus locally.

Reuse source admission and the reviewed candidates/questions/independent oracles.
No source generation, model calls, or historical artifact writes. --dry-run checks
the complete proposed corpus without writing; --validate is read-only admission
for launchers (the messages-only trainer cannot enforce these declarations).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from data_pipeline.board_recognition.sources import canonical_sha256, file_sha256
from sft.board_fluency_scoring import SFT_SCHEMA, score_board_fluency
from sft.scripts import build_board_fluency_review as review
from sft.scripts.build_symbolic_board_dataset import (
    DEFAULT_ROOT, graph_case_coverage, load_sources, read_json, read_jsonl,
)
from sft.symbolic_board_tasks import atlas_geometry, decode_state

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/generated/sft/symbolic_board_fluency_sft_v1"
REVIEW = review.OUTPUT / "review.jsonl"
REVIEW_SHA256 = "40469f58d060dd9a1d24354df42d959971172ba6e0de5e70b153cdda747e5b12"
INVENTORY = ROOT / "artifacts/generated/board_recognition/replay_v1/ms_swift_bidirectional_v1/trainable_tokens.json"
VERSION = "symbolic_board_fluency_sft_v1"
SCHEMA = SFT_SCHEMA
SEED = 20260915
OPERATIONS = tuple(review.OPERATION_FAMILY)
PIP_OPERATIONS = frozenset({"resource_pip_totals", "resource_pip_argmax"})
SPLITS = ("train", "validation", "test")
COUNTS = dict(train=3200, validation=370, test=370, validation_eval=190)
ATLAS_PATTERN = re.compile(r"<[NETP][0-9_]+>")
require = review.require


@dataclass
class Donor:
    """The source-backed portion of review.Donor, without fictitious v2 lineage."""

    state: dict
    data: dict
    provenance: dict
    state_hash: str
    terrain_hash: str


def donor_for(record: dict) -> Donor:
    state = record["state"]
    data = decode_state(state)
    return Donor(state, data, record["provenance"], canonical_sha256(state),
                 canonical_sha256(data["tiles"]))


def pin(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def review_exclusions() -> dict:
    require(file_sha256(REVIEW) == REVIEW_SHA256, "approved review bytes changed")
    rows = read_jsonl(REVIEW)
    ids, hashes, facts = set(), set(), set()
    for row in rows:
        m = {"schema": row["schema"], **row["metadata"]}
        require(row["schema"] == review.SCHEMA, "review schema changed")
        gold = row["messages"][1]["content"]
        require(score_board_fluency(gold, gold, m)["correct"], "review gold/flags changed")
        require(m["state_sha256"] == canonical_sha256(m["target"]["state"]), "review state hash")
        ids.add(m["state_id"])
        hashes.add(m["state_sha256"])
        facts.add(m["provenance"]["board_fact_sha256"])
    require(len(rows) == len(ids) == len(hashes) == 200, "expected 200 distinct reviewed states")
    return {"file": pin(REVIEW), "state_ids": sorted(ids), "state_sha256": sorted(hashes),
            "board_fact_sha256": sorted(facts)}


def eligible_sources(sources: dict, exclusions: dict) -> tuple[dict, dict]:
    ids, hashes, facts = (set(exclusions[key]) for key in
                          ("state_ids", "state_sha256", "board_fact_sha256"))
    result, report = {}, {}
    for split in SPLITS:
        unique, skipped = {}, Counter()
        for record in sources[split]:
            donor = donor_for(record)
            p = donor.provenance
            if p["state_id"] in ids or donor.state_hash in hashes or p["board_fact_sha256"] in facts:
                skipped["review_id_or_content"] += 1
            elif not donor.data["buildings"] or not donor.data["roads"]:
                skipped["no_buildings_or_no_roads"] += 1
            elif donor.state_hash in unique:
                skipped["duplicate_state_content"] += 1
            else:
                unique[donor.state_hash] = donor
        result[split] = list(unique.values())
        report[split] = {"admitted_source_states": len(sources[split]), "eligible_states": len(unique),
                         "terrain_layouts": len({d.terrain_hash for d in unique.values()}),
                         "skipped": dict(skipped)}
    return result, report


def roster_positions(candidate: review.Candidate) -> tuple:
    return tuple(candidate.donor.state["colors"].index(candidate.query[key])
                 for key in ("color", "a", "b") if key in candidate.query)


def quotas(split: str) -> dict:
    dynamic = 160 if split == "train" else 10 if split == "validation_eval" else 20
    return {op: (5 if split != "train" and op in PIP_OPERATIONS else dynamic) for op in OPERATIONS}


def apportion(capacities: dict, total: int) -> dict:
    """Equal supported cells, saturating scarce cells rather than padding them."""
    require(sum(capacities.values()) >= total,
            f"unsupported quota: need {total}, unique capacity {sum(capacities.values())}")
    counts = dict.fromkeys(sorted(capacities), 0)
    for _ in range(total):
        key = min((k for k in counts if counts[k] < capacities[k]), key=lambda k: (counts[k], k))
        counts[key] += 1
    return counts


def select(donors: list[Donor], split: str, seed: int) -> tuple[list, dict]:
    """Min-cost integral flow: exact quotas, unique questions, maximum distinct states.

    Dynamic queries partition by operation/stratum/queried roster positions.
    Pip queries pass through a capacity-one terrain gate per operation. A free
    first use of every state and costly later uses maximize distinct state use.
    """
    rng, atlas = random.Random(seed), atlas_geometry()
    bank = defaultdict(lambda: defaultdict(list))
    for donor in donors:
        for c in review.candidates_for(donor, atlas, rng):
            bank[c.operation, c.stratum, roster_positions(c)][donor.state_hash].append(c)
    by_state = {d.state_hash: d for d in donors}
    capacities = {}
    for group, states in bank.items():
        capacities[group] = (len({by_state[h].terrain_hash for h in states}) if group[0] in PIP_OPERATIONS
                             else sum(len(cs) for cs in states.values()))
    requested = {}
    for op, count in quotas(split).items():
        cells = sorted({g[1] for g in bank if g[0] == op})
        cell_counts = apportion({cell: sum(n for g, n in capacities.items() if g[:2] == (op, cell))
                                 for cell in cells}, count)
        for cell, n in cell_counts.items():
            requested.update(apportion({g: capacity for g, capacity in capacities.items()
                                         if g[:2] == (op, cell)}, n))
    total = sum(requested.values())
    graph = nx.DiGraph()
    graph.add_node("source", demand=-total)
    graph.add_node("sink", demand=total)
    state_order = sorted(by_state)
    rng.shuffle(state_order)
    # Losing one distinct state costs more than every possible rank tie combined.
    reuse_cost = (len(state_order) + 1) * total
    for rank, h in enumerate(state_order):
        graph.add_edge(("state", h), "sink", capacity=1, weight=rank)
        graph.add_edge(("state", h), ("reuse", h), capacity=3, weight=reuse_cost + rank)
        graph.add_edge(("reuse", h), "sink", capacity=total, weight=0)
    for group, n in sorted(requested.items()):
        if not n:
            continue
        graph.add_edge("source", group, capacity=n, weight=0)
        for h, candidates in sorted(bank[group].items()):
            if group[0] in PIP_OPERATIONS:
                terrain = ("terrain", group[0], by_state[h].terrain_hash)
                graph.add_edge(group, terrain, capacity=1, weight=0)
                graph.add_edge(terrain, ("state", h), capacity=1, weight=0)
            else:
                graph.add_edge(group, ("state", h), capacity=len(candidates), weight=0)
    cap, failed_caps = 4, []
    while True:
        try:
            _, flow = nx.network_simplex(graph)
            break
        except nx.NetworkXUnfeasible:
            failed_caps.append(cap)
            # Training stays within the requested cap. Heldout has only existing
            # states: raise its cap only after proving the previous cap infeasible.
            require(split != "train" and cap < total,
                    f"{split}: quotas unsupported at {cap} queries/state; no corpus written")
            cap += 1
            for h in state_order:
                graph[("state", h)][("reuse", h)]["capacity"] = cap - 1
    selected, query_uses = [], Counter()
    for group, n in sorted(requested.items()):
        if not n:
            continue
        assignments = Counter()
        for node, amount in flow[group].items():
            if not amount:
                continue
            if node[0] == "terrain":
                assignments.update({dest[1]: used for dest, used in flow[node].items() if used})
            else:
                assignments[node[1]] += amount
        for h, amount in sorted(assignments.items()):
            options = list(bank[group][h])
            for _ in range(amount):
                chosen = min(options, key=lambda c: (
                    sum(query_uses[c.operation, k, str(v)] for k, v in c.query.items()),
                    -c.quality, c.tie_rank))
                options.remove(chosen)
                selected.append(chosen)
                query_uses.update((chosen.operation, k, str(v)) for k, v in chosen.query.items())
    require(len(selected) == total, "flow extraction did not fill quotas")
    used = Counter(c.donor.state_hash for c in selected)
    selected_counts = Counter((c.operation, c.stratum, roster_positions(c)) for c in selected)
    support = [{"operation": g[0], "stratum": g[1], "roster_positions": list(g[2]),
                "eligible_unique_states": len(bank[g]), "unique_query_capacity": capacities[g],
                "selected": selected_counts[g]} for g in sorted(bank)]
    missing = [{"operation": op, "stratum": cell} for op in OPERATIONS for cell in review.CELLS[op]
               if not any(g[:2] == (op, cell) for g in bank)]
    return interleave(selected, seed + 100), {
        "initial_state_cap": 4, "state_cap": cap, "proven_infeasible_caps": failed_caps,
        "maximum_distinct_states_at_cap": len(used), "max_presentations_per_state": max(used.values()),
        "support_by_cell_and_roster": support, "absent_review_strata": missing,
        "retained_candidate_queries": sum(len(cs) for states in bank.values() for cs in states.values()),
        "policy": "Equal supported strata, then equal supported queried roster positions, saturating finite "
                  "capacity; integral min-cost flow maximizes distinct states under the cap. "
                  "One question per terrain per pip operation. Reviewed candidate enumeration only.",
    }


def interleave(candidates: list, seed: int) -> list:
    rng = random.Random(seed)
    groups = {op: [] for op in OPERATIONS}
    for c in candidates:
        groups[c.operation].append(c)
    for values in groups.values():
        rng.shuffle(values)
    return [groups[op][i] for i in range(max(map(len, groups.values())))
            for op in OPERATIONS if i < len(groups[op])]


def make_rows(selected: list, split: str) -> list[dict]:
    rows, counts = [], Counter()
    atlas = atlas_geometry()
    for c in selected:
        d, op = c.donor, c.operation
        q = {"operation": op, **c.query}
        text = review.question(op, c.query)
        gold = review.answer(review.Facts(d.data, atlas), op, c.query)
        row_id = f"{VERSION}/{split}/{op}/{counts[op]:04d}"
        counts[op] += 1
        declarations = {"schema": SCHEMA, "split": split,
                        "task_role": "train" if split == "train" else "component_eval",
                        "review_only": False, "admitted_for_training": split == "train"}
        metadata = {**declarations, "class": "board_fluency", "family": review.OPERATION_FAMILY[op],
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


def validation_subset(rows: list[dict]) -> list[dict]:
    """A fixed stratified subset, retaining original validation IDs and row bytes."""
    chosen, uses = [], Counter()
    for op, count in quotas("validation_eval").items():
        options = [r for r in rows if r["metadata"]["operation"] == op]
        groups = Counter(r["metadata"]["selection_stratum"] for r in options)
        targets = apportion(groups, count)
        for cell, n in targets.items():
            cell_rows = [r for r in options if r["metadata"]["selection_stratum"] == cell]
            for _ in range(n):
                row = min(cell_rows, key=lambda r: (
                    uses[op, cell, tuple(r["metadata"]["queried_roster_positions"])],
                    uses[r["metadata"]["state_sha256"]], r["metadata"]["row_position"]))
                cell_rows.remove(row)
                chosen.append(row)
                uses[row["metadata"]["state_sha256"]] += 1
                uses[op, cell, tuple(row["metadata"]["queried_roster_positions"])] += 1
    return sorted(chosen, key=lambda r: r["metadata"]["row_position"])


def profile(rows: list[dict]) -> dict:
    metadata = [r["metadata"] for r in rows]
    states = Counter(m["state_sha256"] for m in metadata)
    cells = defaultdict(Counter)
    roster = Counter()
    for m in metadata:
        cells[m["operation"]][m["selection_stratum"]] += 1
        roster[m["operation"], m["selection_stratum"], tuple(m["queried_roster_positions"])] += 1
    return {"rows": len(rows), "unique_states": len(states),
            "unique_state_ids": len({m["state_id"] for m in metadata}),
            "unique_maps": len({m["provenance"]["board_map_sha256"] for m in metadata}),
            "unique_terrains": len({m["terrain_sha256"] for m in metadata}),
            "unique_games": len({m["provenance"]["source"]["game_id"] for m in metadata
                                 if m["provenance"]["source"]["game_id"] is not None}),
            "unique_trajectories": len({m["provenance"]["source"]["trajectory_id"] for m in metadata}),
            "max_presentations_per_state": max(states.values()),
            "state_reuse_histogram": dict(sorted(Counter(states.values()).items())),
            "by_operation": dict(Counter(m["operation"] for m in metadata)),
            "by_family": dict(Counter(m["family"] for m in metadata)),
            "by_density": dict(Counter(m["provenance"]["density_bin"] for m in metadata)),
            "by_source_kind": dict(Counter(m["provenance"]["source"]["kind"] for m in metadata)),
            "by_stratum": {op: dict(values) for op, values in cells.items()},
            "roster_crosstabs": [{"operation": op, "stratum": cell, "positions": list(pos), "rows": n}
                                 for (op, cell, pos), n in sorted(roster.items())]}


def exposure(rows: list[dict]) -> dict:
    by_operation, by_family, by_step = (defaultdict(Counter) for _ in range(3))
    atlas_occurrences = {role: Counter() for role in ("prompt", "completion")}
    for position, row in enumerate(rows):
        m = row["metadata"]
        counts = Counter(presentations=1)
        for role, message in zip(("prompt", "completion"), row["messages"], strict=True):
            tokens = ATLAS_PATTERN.findall(message["content"])
            atlas_occurrences[role].update(tokens)
            counts[role + "_atlas_tokens"] += len(tokens)
            counts[role + "_utf8_bytes"] += len(message["content"].encode())
        by_operation[m["operation"]].update(counts)
        by_family[m["family"]].update(counts)
        by_step[position // 8 + 1].update(counts)
    totals = Counter()
    for counts in by_operation.values():
        totals.update(counts)
    return {"totals": dict(totals), "by_operation": dict(by_operation), "by_family": dict(by_family),
            "by_planned_step_at_batch_8": dict(by_step), "atlas_token_occurrences": atlas_occurrences,
            "model_token_counts": None,
            "token_scope": "Exact canonical atlas-token occurrences in message text; UTF-8 bytes are not tokens. "
                           "Full prompt/completion/EOT token counts require the saved checkpoint tokenizer in launcher "
                           "preflight. No tokenizer, model, or chat template is substituted here."}


def validate_rows(files: dict, sources: dict, exclusions: dict) -> dict:
    """Source admission, split/exclusion checks, independent gold and strict self-score."""
    require({name: len(rows) for name, rows in files.items()} == COUNTS, "corpus counts differ")
    atlas, checks = atlas_geometry(), Counter()
    source_index = {split: {r["provenance"]["state_id"]: r for r in sources[split]} for split in SPLITS}
    blocked = {key: set(exclusions[key]) for key in ("state_ids", "state_sha256", "board_fact_sha256")}
    groups = {key: defaultdict(set) for key in
              ("state_id", "state_sha256", "board_fact_sha256", "board_map_sha256", "terrain_sha256",
               "game_id", "trajectory_id", "layout_id")}
    seen_ids, contracts, donors = set(), {}, {}
    for split in SPLITS:
        rows, presentations, prompts = files[split], set(), set()
        require(Counter(r["metadata"]["operation"] for r in rows) == quotas(split), f"{split}: operation quotas")
        for position, row in enumerate(rows):
            m, messages = row["metadata"], row["messages"]
            require(set(row) == {"schema", "split", "task_role", "review_only", "admitted_for_training",
                                 "id", "row_id", "messages", "metadata"}, "unexpected model row keys")
            for key, value in (("schema", SCHEMA), ("split", split), ("review_only", False),
                               ("admitted_for_training", split == "train"),
                               ("task_role", "train" if split == "train" else "component_eval")):
                require(row.get(key) == m.get(key) == value and type(row[key]) is type(value)
                        and type(m[key]) is type(value), f"conflicting declaration: {key}")
            require(row["id"] == row["row_id"] and row["id"] not in seen_ids, "duplicate row identity")
            seen_ids.add(row["id"])
            require(m["row_position"] == position, "row order mismatch")
            require(len(messages) == 2 and [msg["role"] for msg in messages] == ["user", "assistant"]
                    and all(set(msg) == {"role", "content"} and isinstance(msg["content"], str) for msg in messages),
                    "only two text messages are admitted")
            record = source_index[split].get(m["state_id"])
            require(record is not None and record["provenance"] == m["provenance"]
                    and record["state"] == m["target"]["state"], "row is not an exact admitted source")
            if m["state_id"] not in donors:
                donors[m["state_id"]] = donor_for(record)
            donor = donors[m["state_id"]]
            p, op, q = donor.provenance, m["operation"], m["target"]["query"]
            require(donor.data["buildings"] and donor.data["roads"], "empty donor")
            require(m["state_sha256"] == donor.state_hash and m["terrain_sha256"] == donor.terrain_hash
                    and m["query_sha256"] == canonical_sha256(q), "state/query/terrain hash mismatch")
            require(m["state_id"] not in blocked["state_ids"] and donor.state_hash not in blocked["state_sha256"]
                    and p["board_fact_sha256"] not in blocked["board_fact_sha256"], "review state/content leakage")
            args = {key: value for key, value in q.items() if key != "operation"}
            candidate = review.Candidate(donor, op, args, m["selection_stratum"], 0, 0.0)
            require(m["queried_roster_positions"] == list(roster_positions(candidate)), "roster metadata mismatch")
            text = review.question(op, args)
            gold = review.answer(review.Facts(donor.data, atlas), op, args)
            require(m["question"] == text and messages[0]["content"] == review.prompt(donor.state, text),
                    "prompt not exact symbolic state plus reviewed query")
            require(m["answer"] == messages[1]["content"] == gold, "stored gold failed state recomputation")
            if donor.state_hash not in contracts:
                contracts[donor.state_hash] = read_json(Path(p["paths"]["contract"]))
            require(gold == review.reference_answer(candidate, contracts[donor.state_hash], checks),
                    f"independent oracle mismatch: {op}")
            require(score_board_fluency(gold, gold, m)["correct"], "gold failed shared scorer")
            checks["shared_scorer_gold_checks"] += 1
            key = (donor.terrain_hash if op in PIP_OPERATIONS else donor.state_hash, op, review.compact(q))
            require(key not in presentations and messages[0]["content"] not in prompts, "duplicate question")
            presentations.add(key)
            prompts.add(messages[0]["content"])
            identities = {"state_id": m["state_id"], "state_sha256": donor.state_hash,
                          "terrain_sha256": donor.terrain_hash, **p, **p["source"]}
            for name, group in groups.items():
                if identities[name] is not None:
                    group[identities[name]].add(split)
        if split == "train":
            require(max(Counter(r["metadata"]["state_sha256"] for r in rows).values()) <= 4,
                    "training state cap exceeded")
            roster_cells = defaultdict(Counter)
            for row in rows:
                m = row["metadata"]
                roster_cells[m["operation"], m["selection_stratum"]][tuple(m["queried_roster_positions"])] += 1
            for (op, cell), counts in roster_cells.items():
                require(max(counts.values()) - min(counts.values()) <= 1,
                        f"training roster imbalance: {op}/{cell}")
                if any(len(pos) == 1 for pos in counts):
                    require(set(counts) == {(i,) for i in range(4)}, f"missing queried roster position: {op}/{cell}")
        else:
            for op in PIP_OPERATIONS:
                require(len({r["metadata"]["provenance"]["board_map_sha256"] for r in rows
                             if r["metadata"]["operation"] == op}) == 5, "heldout pip rows need all five maps")
    for name, group in groups.items():
        require(all(len(splits) == 1 for splits in group.values()), f"cross-split {name} leakage")
    require(files["validation_eval"] == validation_subset(files["validation"]), "fixed validation subset changed")
    require(Counter(r["metadata"]["operation"] for r in files["validation_eval"]) == quotas("validation_eval"),
            "validation eval operation quota")
    first = Counter(r["metadata"]["operation"] for r in files["train"][:1024])
    require(set(first) == set(OPERATIONS) and max(first.values()) - min(first.values()) <= 1,
            "first 1024 operation coverage is not near-uniform")
    heldout_graphs = {split: graph_case_coverage(sources[split]) for split in ("validation", "test")}
    return {"checks": dict(checks), "split_disjoint_keys": list(groups), "review_overlap": 0,
            "heldout_graph_coverage": heldout_graphs,
            "profiles": {split: profile(rows) for split, rows in files.items()},
            "first_1024": exposure(files["train"][:1024]), "all_train": exposure(files["train"])}


def build_dataset(output: Path = OUTPUT, *, seed: int = SEED, dry_run: bool = False) -> dict:
    output = output.resolve()
    require(output.parent.is_dir(), "output parent must already exist")
    require(not output.exists(), "output already exists; validate it instead of overwriting")
    exclusions = review_exclusions()
    sources, source_audit = load_sources(DEFAULT_ROOT)
    donors, eligibility = eligible_sources(sources, exclusions)
    files, selection = {}, {}
    for i, split in enumerate(SPLITS):
        selected, selection[split] = select(donors[split], split, seed + i)
        files[split] = make_rows(selected, split)
    files["validation_eval"] = validation_subset(files["validation"])
    validation = validate_rows(files, sources, exclusions)
    code = [Path(__file__), Path(review.__file__), ROOT / "sft/board_fluency_scoring.py",
            ROOT / "sft/spatial_tasks.py"]
    metadata = {
        "schema": SCHEMA, "corpus_status": "admitted_sft", "review_only": False,
        "admitted_for_training": True, "training_split": "train", "seed": seed, "counts": COUNTS,
        "families": review.FAMILIES, "source_audit": source_audit, "review_exclusions": exclusions,
        "eligibility": eligibility, "selection": selection, "validation": validation,
        "token_inventory_source": pin(INVENTORY), "build_code": [pin(path) for path in code],
        "training_plan": {
            "source_checkpoint": "/runs/catan-vision-sft/spatial-continuation-20260912-r01/checkpoints/checkpoint-128",
            "approved_language_lora_rank": 16, "optimizer_steps": 128, "effective_batch_size": 8,
            "planned_presentations": 1024, "corpus_presentations": 3200,
            "full_corpus_epoch_steps_at_batch_8": 400, "planned_fraction_of_corpus": 0.32,
            "order": "20-operation round-robin; deterministic shuffle within each operation. "
                     "First 1024 coverage assumes sequential single-pass consumption; launcher must enforce it.",
            "token_exposure": "See validation.first_1024 and validation.all_train. Atlas token occurrences "
                              "are exact; full native token exposure is measured by checkpoint-tokenizer preflight.",
        },
        "scope": [
            "Only the 20 approved reviewed operations; original nonempty admitted states and original splits.",
            "All 200 reviewed state IDs, canonical state hashes, and visible-board hashes excluded.",
            "Board-wide pip questions deduplicate on terrain, ignoring pieces, robber and ports.",
            "Full settlement conjunction and Longest Road remain transfer-only; strategy deferred to RL.",
            "Reserved color diagnostics contribute zero rows. No static atlas rehearsal.",
            "Heldout cycles/alternate routes and effective blockers have no source support; see absent strata.",
            "Admission certifies material state and source provenance, not independently reconstructed histories.",
            "Rows on a state/game/map are correlated; row count is not an independent sample count.",
        ],
    }
    summary = {"valid": True, "dry_run": dry_run, "corpus_status": "admitted_sft", "counts": COUNTS,
               "output_dir": str(output), "eligibility": eligibility,
               "profiles": validation["profiles"], "checks": validation["checks"],
               "first_1024": validation["first_1024"]}
    if dry_run:
        return summary
    output.mkdir()
    for name, rows in files.items():
        (output / f"{name}.jsonl").write_text("".join(review.compact(row) + "\n" for row in rows))
    # A byte-for-byte copy of the successful trainer inventory, including its schema.
    (output / "trainable_tokens.json").write_bytes(INVENTORY.read_bytes())
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    names = [f"{name}.jsonl" for name in files] + ["trainable_tokens.json", "metadata.json"]
    manifest = {"schema": SCHEMA, "corpus_status": "admitted_sft", "counts": COUNTS,
                "files": {name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size,
                                  **({"rows": COUNTS[Path(name).stem]} if name.endswith(".jsonl") else {})}
                          for name in names}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return summary


def validate_dataset(path: Path) -> dict:
    """Read-only launcher gate: hashes/counts/splits/exclusions/source joins/all golds.

    Accept a corpus directory or its train.jsonl. Raises on any discrepancy; a
    successful return explicitly admits only train.jsonl for training.
    """
    output = path.resolve()
    if output.is_file():
        require(output.name == "train.jsonl", "only train.jsonl is an admitted training input")
        output = output.parent
    manifest = read_json(output / "manifest.json")
    require(manifest["schema"] == SCHEMA and manifest["corpus_status"] == "admitted_sft", "not an admitted corpus")
    names = {f"{name}.jsonl" for name in COUNTS} | {"trainable_tokens.json", "metadata.json"}
    require(set(manifest["files"]) == names and manifest["counts"] == COUNTS, "manifest files/counts mismatch")
    for name, info in manifest["files"].items():
        asset = output / name
        require(asset.stat().st_size == info["bytes"] and file_sha256(asset) == info["sha256"],
                f"artifact bytes changed: {name}")
        if name.endswith(".jsonl"):
            require(info["rows"] == COUNTS[Path(name).stem], f"manifest row count: {name}")
    metadata = read_json(output / "metadata.json")
    require(metadata["schema"] == SCHEMA and metadata["corpus_status"] == "admitted_sft"
            and metadata["admitted_for_training"] is True and metadata["review_only"] is False
            and metadata["training_split"] == "train" and metadata["counts"] == COUNTS, "corpus admission changed")
    for info in metadata["build_code"] + [metadata["token_inventory_source"]]:
        require(file_sha256(Path(info["path"])) == info["sha256"], f"pinned input changed: {info['path']}")
    require((output / "trainable_tokens.json").read_bytes() == INVENTORY.read_bytes(), "inventory must be exact copy")
    exclusions = review_exclusions()
    require(exclusions == metadata["review_exclusions"], "review exclusions changed")
    sources, source_audit = load_sources(DEFAULT_ROOT)
    require(source_audit == metadata["source_audit"], "source admission changed")
    files = {name: read_jsonl(output / f"{name}.jsonl") for name in COUNTS}
    validation = validate_rows(files, sources, exclusions)
    require(json.loads(json.dumps(validation)) == metadata["validation"], "coverage/exposure/gold audit changed")
    return {"valid": True, "schema": SCHEMA, "corpus_status": "admitted_sft", "counts": COUNTS,
            "admitted_for_training": True, "train_jsonl": str(output / "train.jsonl"),
            "manifest_sha256": file_sha256(output / "manifest.json"),
            "files": manifest["files"], "review_overlap": 0, "checks": validation["checks"],
            "profiles": validation["profiles"], "first_1024": validation["first_1024"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    result = (validate_dataset(args.output_dir) if args.validate else
              build_dataset(args.output_dir, seed=args.seed, dry_run=args.dry_run))
    # Aggregate facts only; source records never enter terminal/chat output.
    print(json.dumps({key: result[key] for key in ("valid", "counts", "checks")}, indent=2, sort_keys=True))
    print(json.dumps({split: {key: p[key] for key in
                             ("unique_states", "unique_maps", "max_presentations_per_state", "by_stratum")}
                      for split, p in result["profiles"].items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
