from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import TypeAlias, cast

from data_pipeline.board_recognition.sources import canonical_sha256, file_sha256
from evals.catan_board_bench.tokens import atlas_tokens
from sft.board.symbolic_board_tasks import (
    STATIC_TASKS,
    TRAIN_TASKS,
    TRANSFER_TASKS,
    score_symbolic_task,
    strict_json,
    symbolic_answer,
    symbolic_prompt,
    symbolic_task_role,
)
from sft.board.symbolic_board_tasks._types import StatePayload
from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_float, as_list, as_str

from ._components import directional_pair_manifest, known_direction_exposure
from ._sources import (
    DEFAULT_OUTPUT,
    EVAL_QUOTAS,
    SPLITS,
    TOKEN_PATTERN,
    TRAIN_QUOTAS,
    _check,
    _hash,
    load_sources,
    read_json,
    read_jsonl,
)
from ._transfer import graph_case_coverage, transfer_projection, transfer_rows, transfer_weights
from ._types import QueryDict, SourceRecord

# (task, mode, roster position, polarity, source density) crosstab cell.
ProfileKey: TypeAlias = tuple[str, str, int | None, str, str]


def validate_row_declarations(row: JsonDict, split: str) -> None:
    """Every task/family/split/role declaration must agree; redundant tags are not authority."""
    _check(set(row) == {"schema", "id", "row_id", "task_type", "training_family", "task_role",
                        "split", "messages", "metadata"}, "unexpected model record keys")
    _check(row["schema"] == "catan_symbolic_board_row/v2", "wrong row schema")
    task, metadata = row["task_type"], row["metadata"]
    _check(isinstance(task, str) and isinstance(metadata, dict), "invalid task declarations")
    role = symbolic_task_role(cast("str", task), split)
    for key, value in (("task_type", task), ("training_family", task), ("split", split), ("task_role", role)):
        _check(row.get(key) == cast("JsonDict", metadata).get(key) == value,
               f"task declaration mismatch: {key}")


def component_profile(rows: list[JsonDict]) -> JsonLikeDict:
    """Record model-visible queried roster positions alongside conditional source diversity."""
    cells: defaultdict[ProfileKey, list[JsonDict]] = defaultdict(list)
    families: defaultdict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        m, task = as_dict(row["metadata"]), as_str(row["task_type"])
        if task in STATIC_TASKS:
            continue
        messages = [as_dict(entry) for entry in as_list(row["messages"])]
        prompt, response = as_str(messages[0]["content"]), as_str(messages[1]["content"])
        roster_match = re.search(r"Participants: ([A-Z_ ]+)\.\nBoard:", prompt)
        roster = cast("re.Match[str]", roster_match).group(1).split()
        question = prompt.rsplit("\n", 1)[-1]
        color_match = re.search(r"(?:List existing |List all existing |List nodes with a |For |existing )([A-Z_]+)(?: road| incident road| building| settlement| city|,)", question)
        position = roster.index(color_match.group(1)) if color_match else None
        mode = as_str(as_dict(m["sampling_cell"])["mode"])
        if task == "symbolic_local_constraint":
            mode = ("has_owned_incident_road" if "incident road" in question else
                    "no_adjacent_building" if "edge-adjacent" in question else "empty")
        if task in ("symbolic_reachable", "symbolic_shortest_route"):
            polarity = as_str(as_dict(m["sampling_cell"])["polarity"])
        else:
            polarity = "negative" if response in ("NONE", "no") else "positive"
        p = as_dict(m["provenance"])
        cells[task, mode, position, polarity, as_str(p["density_bin"])].append(p)
        families[task].append(p)
    return {
        "rendered_roster_crosstabs": [dict(task_type=t, mode=mode, roster_position=position,
                                           polarity=polarity, density=density, rows=len(ps),
                                           unique_states=len({p["state_id"] for p in ps}))
                                      for (t, mode, position, polarity, density), ps in cells.items()],
        "families": {t: {"rows": len(ps), "unique_states": len({p["state_id"] for p in ps}),
                          "unique_maps": len({p["board_map_sha256"] for p in ps}),
                          "by_density": dict(Counter(as_str(p["density_bin"]) for p in ps)),
                          "by_source_kind": dict(
                              Counter(as_str(as_dict(p["source"])["kind"]) for p in ps))}
                     for t, ps in families.items()},
    }


def validate_component_balance(rows: list[JsonDict]) -> None:
    """Production gate on rendered roster/predicate/label cross-tabs, not cached labels."""
    tables: defaultdict[tuple[str, str, int | None], Counter[str]] = defaultdict(Counter)
    crosstabs = cast("list[JsonDict]", component_profile(rows)["rendered_roster_crosstabs"])
    for cell in crosstabs:
        if cell["task_type"] in ("symbolic_local_constraint", "symbolic_owned_incident_roads",
                                  "symbolic_owned_nodes", "symbolic_owned_roads"):
            key = (as_str(cell["task_type"]), as_str(cell["mode"]),
                   cast("int | None", cell["roster_position"]))
            tables[key][as_str(cell["polarity"])] += cast("int", cell["rows"])
    for key, counts in tables.items():
        _check(counts["positive"] == counts["negative"] > 0, f"rendered roster label shortcut: {key}")


def validate_rows(files: dict[str, list[JsonDict]], pairs: dict[str, list[list[str]]],
                  sources: dict[str, list[SourceRecord]] | None = None,
                  exposure: JsonLikeDict | None = None) -> JsonLikeDict:
    """Re-render prompts, recompute all answers, check split/pair/source boundaries."""
    ids: set[object] = set()
    pair_sets: dict[str, set[tuple[str, ...]]] = {}
    coverage: dict[str, list[str]] = {}
    query_coverage: dict[str, list[str]] = {}
    provenance_groups: dict[str, dict[object, str]] = {
        key: {} for key in ("trajectory_id", "board_map_sha256")}
    for split, values in pairs.items():
        pair_sets[split] = {tuple(v) for v in values}
        _check(len(pair_sets[split]) == len(values), "duplicate directional pair")
    for a, b in combinations(pair_sets, 2):
        _check(not pair_sets[a] & pair_sets[b], "directional pair leakage")
    for split, rows in files.items():
        base_split = split.removeprefix("transfer_")
        tasks = TRANSFER_TASKS if split.startswith("transfer_") else TRAIN_TASKS
        tokens: set[str] = set()
        queried: set[str] = set()
        projections: set[str] = set()
        source_index = ({as_str(r["provenance"]["state_id"]): r for r in sources[base_split]}
                        if sources else None)
        for position, row in enumerate(rows):
            validate_row_declarations(row, split)
            _check(row["row_id"] == row["id"] and row["id"] not in ids, "duplicate row id")
            ids.add(row["id"])
            task, metadata = as_str(row["task_type"]), as_dict(row["metadata"])
            _check(task in tasks and task == metadata["task_type"], "task split leakage")
            _check(metadata["split"] == split and metadata["row_position"] == position, "row ordering/split error")
            messages = [as_dict(entry) for entry in as_list(row["messages"])]
            _check(len(messages) == 2 and [m["role"] for m in messages] == ["user", "assistant"]
                   and all(set(m) == {"role", "content"} and isinstance(m["content"], str) for m in messages),
                   "rows require user/answer text only")
            target = as_dict(metadata["target"])
            _check(messages[0]["content"] == symbolic_prompt(task, target), "prompt/target mismatch or leaked input")
            score = score_symbolic_task("deliberately untrusted cache",
                                        as_str(messages[1]["content"]), metadata)
            _check(score is not None and bool(score["correct"]), "answer failed recomputing scorer")
            checked = cast("JsonDict", score)
            _check(metadata["query_sha256"] == canonical_sha256(target["query"]), "query digest mismatch")
            if task in ("symbolic_direction", "symbolic_direction_choice"):
                q = as_dict(target["query"])
                pair = tuple(sorted((as_str(q["a"]), as_str(q["b"]))))
                _check(pair in pair_sets[base_split] and metadata["directional_pair"] == list(pair), "wrong direction partition")
                if exposure is not None:
                    known = as_dict(exposure["pair_sources"]).get(" ".join(pair), [])
                    _check(metadata.get("known_training_exposure") == known, "historical exposure annotation mismatch")
                    _check(base_split == "train" or not known, "known training pair entered heldout")
            if task not in STATIC_TASKS:
                p = as_dict(metadata["provenance"])
                _check(p["split"] == base_split, "source split leakage")
                for key in provenance_groups:
                    value = as_dict(p["source"])[key] if key == "trajectory_id" else p[key]
                    previous = provenance_groups[key].setdefault(value, base_split)
                    _check(previous == base_split, f"dynamic {key} leakage")
                if source_index is not None:
                    original = source_index[as_str(p["state_id"])]
                    _check(original["state"] == target["state"] and original["provenance"] == p,
                           "model state/provenance differs from real source")
                if task in TRAIN_TASKS:
                    slot = as_dict(metadata["sampling_cell"])
                    q = as_dict(target["query"])
                    if "color" in q:
                        index = cast("int", slot["position"])
                        _check(type(slot["position"]) is int and 0 <= index < 4 and
                               q["color"] == as_list(as_dict(target["state"])["colors"])[index],
                               "sampling roster position mismatch")
                    if slot["polarity"] in ("positive", "negative"):
                        positive = checked["expected_normalized"] not in ("no", "NONE")
                        _check(positive == (slot["polarity"] == "positive"), "sampling polarity mismatch")
            if task in TRANSFER_TASKS:
                projection = transfer_projection(
                    task, cast("StatePayload", target["state"]),
                    cast("QueryDict", target["query"]))
                _check(projection == metadata.get("task_projection_sha256") and projection not in projections,
                       "duplicate or incorrect task-relevant transfer projection")
                projections.add(projection)
                q = as_dict(target["query"])
                mode = "all"
                if task == "symbolic_settlement_locations":
                    near = q["near"]
                    mode = (as_str(q["phase"]) + "/"
                            + (as_str(as_dict(near)["kind"]) if near else "all"))
                if task in ("symbolic_longest_leaders", "symbolic_longest_lengths"):
                    answer = symbolic_answer("symbolic_longest_lengths", cast("JsonDict", {
                        "state": target["state"], "query": {}}))
                    lengths = as_dict(strict_json(answer))
                    positive = max(as_float(v) for v in lengths.values()) > 0
                else:
                    positive = checked["expected_normalized"] != "NONE"
                _check(metadata["evaluation_mode"] == mode and
                       metadata["polarity"] == ("positive" if positive else "negative"), "transfer evaluation group mismatch")
            tokens.update(TOKEN_PATTERN.findall(
                as_str(messages[0]["content"]) + " " + as_str(messages[1]["content"])))
            queried.update(TOKEN_PATTERN.findall(json.dumps(target["query"])))
        coverage[split], query_coverage[split] = sorted(tokens), sorted(queried)
        if split.startswith("transfer_"):
            for row, weights in zip(rows, transfer_weights(rows), strict=True):
                _check(all(as_dict(row["metadata"]).get(k) == v for k, v in weights.items()),
                       "invalid transfer weights")
        else:
            validate_component_balance(rows)
    _check(set(coverage["train"]) == set(atlas_tokens()), "missing or noncanonical training tokens")
    _check(set(query_coverage["train"]) == set(atlas_tokens()), "all 154 tokens must be active training queries")
    return {"tokens": coverage, "active_query_tokens": query_coverage}


def preserved_v1_artifacts() -> JsonLikeDict:
    """Verify historical output bytes only; historical code hashes intentionally stay historical."""
    root = DEFAULT_OUTPUT.with_name("symbolic_board_v1")
    if not root.exists():
        return {"status": "not_present"}
    manifest = read_json(root / "manifest.json")
    for entry in as_dict(manifest["files"]).values():
        info = as_dict(entry)
        _check(file_sha256(Path(as_str(info["path"]))) == info["sha256"],
               "historical v1 artifact changed")
    return {"status": "preserved", "manifest": _hash(root / "manifest.json"),
            "files": manifest["files"]}


def validate_dataset(output_dir: Path = DEFAULT_OUTPUT) -> JsonLikeDict:
    """Validate bytes, current pinned sources, real-state joins, prompts and all labels."""
    output = output_dir.resolve()
    manifest = read_json(output / "manifest.json")
    _check(manifest["schema"] == "catan_symbolic_board_manifest/v2", "this validator requires v2 output")
    for entry in as_dict(manifest["files"]).values():
        info = as_dict(entry)
        path = Path(as_str(info["path"]))
        _check(path.parent == output and path.is_file(), "manifest output path mismatch")
        _check(file_sha256(path) == info["sha256"], f"generated file changed: {path}")
    metadata = read_json(output / "metadata.json")
    _check(metadata["preserved_v1"] == preserved_v1_artifacts(), "historical v1 receipt changed")
    for entry in as_dict(manifest["source_hashes"]).values():
        info = as_dict(entry)
        _check(file_sha256(Path(as_str(info["path"]))) == info["sha256"],
               f"pinned source changed: {info['path']}")
    sources, audit = load_sources(
        Path(as_str(as_dict(metadata["source_audit"])["source_root"])))
    _check(audit == metadata["source_audit"], "source admission audit changed")
    files = {name: read_jsonl(Path(as_str(as_dict(entry)["path"])))
             for name, entry in as_dict(manifest["files"]).items()
             if "rows" in as_dict(entry)}
    _check({s: len(rows) for s, rows in files.items()} == manifest["counts"] == metadata["counts"], "output count mismatch")
    pairs = read_json(output / "directional_pairs.json")
    exposure = known_direction_exposure()
    _check(exposure == metadata["historical_exposure_audit"], "known historical exposure changed")
    exposed_pairs = [key.split() for key in as_dict(exposure["pair_sources"])]
    splits = cast("dict[str, list[list[str]]]", pairs["splits"])
    _check(pairs["known_training_pairs"] == exposed_pairs and
           splits == directional_pair_manifest(cast("int", metadata["seed"]), exposed_pairs),
           "pair assignment changed")
    _check(validate_rows(files, splits, sources, exposure) == metadata["coverage"],
           "coverage changed")
    for split, rows in files.items():
        _check([r["id"] for r in rows] == as_dict(metadata["row_ids"])[split], "row order changed")
        if split in SPLITS[:3]:
            _check(dict(Counter(r["task_type"] for r in rows)) == (TRAIN_QUOTAS if split == "train" else EVAL_QUOTAS),
                   "family quota drift")
            _check(component_profile(rows) == as_dict(metadata["component_profiles"])[split],
                   "conditional coverage changed")
    for i, split in enumerate(("validation", "test")):
        name = "transfer_" + split
        regenerated, report = transfer_rows(sources[split], name,
                                            cast("int", metadata["seed"]) + i + 10)
        for index, row in enumerate(regenerated):
            as_dict(row["metadata"])["row_position"] = index
        _check(regenerated == files[name]
               and report == as_dict(metadata["transfer_selection"])[name],
               "transfer population/selection/weights changed")
    _check(graph_case_coverage(sources["color_diagnostic"]) ==
           as_dict(as_dict(metadata["reserved_color_diagnostic_coverage"])["graph_cases"]),
           "reserved diagnostics changed")
    return {"valid": True, "counts": metadata["counts"],
            "source_exclusions": len(cast("list[object]", audit["exclusions"]))}
