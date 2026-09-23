"""Source-row verification, case selection, and comparison-row building."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import cast

from data_pipeline.board_recognition.sources import canonical_sha256, file_sha256
from sft.board.coordinate_comparison._constants import (
    ATOM_INSTRUCTION,
    COMPARISON_FIELDS,
    COMPLETE_TEST_TASKS,
    DEFAULT_SOURCE,
    GEOMETRY_CONVENTION,
    INCIDENCE_RELATIONS,
    OPERATION_AREA,
    PAIR_QUOTAS,
    REPRESENTATIONS,
    SCHEMA,
    VERSION,
    SourceCase,
)
from sft.board.coordinate_comparison._mapping import (
    _compact,
    _mapping,
    _require,
    _same,
    _static_fact_sha256,
    _unique_object,
    mapping_sha256,
    project_text,
)
from sft.board.symbolic_board_tasks import (
    STATIC_TASKS,
    symbolic_answer,
    symbolic_prompt,
    symbolic_task_role,
)
from sft.json_types import JsonDict, JsonLikeDict, JsonList, as_dict, as_int, as_str


def comparison_prompt(task: str, target: JsonDict, representation: str) -> str:
    """Original question plus a static inventory or the original complete state."""
    prompt = symbolic_prompt(task, target)
    _require(representation in REPRESENTATIONS, "invalid representation")
    header = ATOM_INSTRUCTION
    if representation == "coordinates":
        prompt = prompt.replace("Use the fixed learned Catan atlas. ",
                                "Use the supplied integer-coordinate Catan layout. ", 1)
        header = GEOMETRY_CONVENTION + "\n" + header
    if target["state"] is None:
        header += "\nEntity inventory: " + " ".join(_mapping())
    return project_text(header + "\n" + prompt, representation)


def _sampling_cell(task: str, target: JsonDict, gold: str) -> JsonLikeDict:
    q, state = as_dict(target["query"]), as_dict(target["state"])
    mode = q["piece"] if task == "symbolic_owned_nodes" else (
        as_str(q["token"])[1] if task == "symbolic_piece_owner" else "all")
    colors = cast("list[str]", state["colors"])
    return {"mode": mode,
            "position": colors.index(as_str(q["color"])) if "color" in q else None,
            "polarity": "negative" if gold == "NONE" else "positive"}


@lru_cache(maxsize=1024)
def _source_semantics(metadata_json: str) -> tuple[str, str]:
    metadata = as_dict(json.loads(metadata_json))
    task = as_str(metadata["task_type"])
    target = as_dict(metadata["target"])
    split = as_str(metadata["split"])
    _require(task in PAIR_QUOTAS and split in {"test", "validation"}, "invalid source task/split")
    _require(metadata["training_family"] == task
             and metadata["task_role"] == symbolic_task_role(task, split), "source task role mismatch")
    gold = symbolic_answer(task, target)
    _same(metadata["query_sha256"], canonical_sha256(target["query"]), "source query hash")
    if task in STATIC_TASKS:
        _require("provenance" not in metadata, "static source has dynamic provenance")
    else:
        provenance = as_dict(metadata["provenance"])
        _require(provenance["split"] == as_dict(provenance["source"])["split"] == split,
                 "source provenance split mismatch")
        for key in ("state_id", "board_map_sha256", "board_fact_sha256"):
            _require(bool(isinstance(provenance[key], str) and provenance[key]),
                     "missing source provenance")
        _same(metadata["sampling_cell"], _sampling_cell(task, target, gold), "source sampling cell")
    if task in ("symbolic_direction", "symbolic_direction_choice"):
        q = as_dict(target["query"])
        _same(metadata["directional_pair"], sorted((as_str(q["a"]), as_str(q["b"]))),
              "source directional pair")
        _same(metadata["known_training_exposure"], [], "heldout directional exposure")
    return symbolic_prompt(task, target), gold


def _source_row(metadata: JsonDict, source_id: str) -> JsonDict:
    prompt, gold = _source_semantics(_compact(metadata))
    return {
        "schema": "catan_symbolic_board_row/v2", "id": source_id, "row_id": source_id,
        **{key: metadata[key] for key in ("task_type", "training_family", "task_role", "split")},
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": gold}],
        "metadata": metadata,
    }


def load_source_rows(root: Path = DEFAULT_SOURCE) -> tuple[list[SourceCase], JsonLikeDict]:
    """Verify both source split files against their manifest, retaining raw-line receipts."""
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = as_dict(json.loads(manifest_path.read_text(), object_pairs_hook=_unique_object))
    _require(manifest["schema"] == "catan_symbolic_board_manifest/v2", "invalid source manifest")
    sources: list[SourceCase] = []
    hashes = {"manifest.json": file_sha256(manifest_path)}
    for split in ("test", "validation"):
        path = root / f"{split}.jsonl"
        digest = file_sha256(path)
        declaration = as_dict(as_dict(manifest["files"])[split])
        _require(digest == declaration["sha256"], f"source file hash mismatch: {split}")
        raw_lines = path.read_bytes().splitlines(keepends=True)
        _require(len(raw_lines) == declaration["rows"] == as_dict(manifest["counts"])[split] == 480,
                 f"source row quota mismatch: {split}")
        hashes[path.name] = digest
        seen: set[object] = set()
        for line, raw in enumerate(raw_lines, 1):
            row = as_dict(json.loads(raw, object_pairs_hook=_unique_object))
            _require(row["id"] not in seen, "duplicate source ID")
            seen.add(row["id"])
            meta = as_dict(row["metadata"])
            _require(row["split"] == meta["split"] == split
                     and meta["row_position"] == line - 1, "source split/position mismatch")
            if row["task_type"] not in PAIR_QUOTAS:
                continue
            _require(not COMPARISON_FIELDS.intersection(meta), "reserved source metadata keys")
            _same(row, _source_row(meta, as_str(row["id"])), "source prompt/gold/declarations")
            messages = [as_dict(entry) for entry in cast("JsonList", row["messages"])]
            receipt = cast("JsonDict", {
                "id": row["id"], "split": split, "schema": row["schema"], "file": path.name,
                "file_sha256": digest, "line": line,
                "line_sha256": hashlib.sha256(raw).hexdigest(),
                "row_sha256": canonical_sha256(row),
                "metadata_sha256": canonical_sha256(meta),
                "metadata_keys": sorted(meta),
                "prompt_sha256": canonical_sha256(messages[0]["content"]),
                "answer_sha256": canonical_sha256(messages[1]["content"]),
            })
            sources.append(SourceCase(row=row, receipt=receipt))
    audit: JsonLikeDict = {"root": str(root), "sha256": hashes,
                           "original_source_hashes": copy.deepcopy(manifest["source_hashes"])}
    return sources, audit


def _required_cells(task: str) -> list[JsonLikeDict]:
    modes = ("building", "settlement", "city") if task == "symbolic_owned_nodes" else ("all",)
    return [dict(mode=mode, position=position, polarity=polarity)
            for position in range(4) for mode in modes for polarity in ("positive", "negative")]


def _case_state_id(case: SourceCase) -> object:
    provenance = as_dict(case["row"]["metadata"]).get("provenance", {})
    return as_dict(provenance).get("state_id")


def select_source_cases(sources: list[SourceCase]) -> list[SourceCase]:
    """Deterministic exact quotas; least-used source state, then source ID breaks ties."""
    selected: list[SourceCase] = []
    uses: Counter[object] = Counter()
    test = [s for s in sources if s["row"]["split"] == "test"]

    def take(pool: list[SourceCase], count: int, label: str, *, complete: bool = False) -> None:
        _require(len(pool) == count if complete else len(pool) >= count, f"missing source quota: {label}")
        pool = list(pool)
        for _ in range(count):
            chosen = min(pool, key=lambda s: (uses[_case_state_id(s)],
                                              as_str(s["row"]["id"])))
            pool.remove(chosen)
            selected.append(chosen)
            state_id = _case_state_id(chosen)
            if state_id is not None:
                uses[state_id] += 1

    for task in COMPLETE_TEST_TASKS:
        take([s for s in test if s["row"]["task_type"] == task], 32, task, complete=True)
    for relation in INCIDENCE_RELATIONS:
        if relation == "P->N":
            pool = [s for s in sources if s["row"]["split"] == "validation"
                    and s["row"]["task_type"] == "symbolic_incidence"
                    and as_dict(as_dict(s["row"]["metadata"])["target"])["query"] in (
                        {"token": "<P03>", "family": "N"}, {"token": "<P08>", "family": "N"})]
            take(pool, 2, relation, complete=True)
        else:
            pool = [s for s in test if s["row"]["task_type"] == "symbolic_incidence"
                    and _relation_key(as_dict(s["row"]["metadata"])) == relation]
            take(pool, 2, relation)
    for task in ("symbolic_owned_nodes", "symbolic_owned_roads", "symbolic_owned_incident_roads"):
        for cell in _required_cells(task):
            pool = [s for s in test if s["row"]["task_type"] == task
                    and as_dict(s["row"]["metadata"])["sampling_cell"] == cell]
            take(pool, 1 if task == "symbolic_owned_nodes" else 2, f"{task}/{cell}")
    _require(len(selected) == len({s["row"]["id"] for s in selected}) == 200, "source selection IDs/quota")
    # Preserve original physical order within each source split.
    return sorted(selected, key=lambda s: (s["row"]["split"] != "test",
                                           as_int(s["receipt"]["line"])))


def _relation_key(metadata: JsonDict) -> str:
    q = as_dict(as_dict(metadata["target"])["query"])
    return as_str(q["token"])[1] + "->" + as_str(q["family"])


def build_comparison_rows(sources: list[SourceCase]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for chosen in select_source_cases(sources):
        source, receipt = chosen["row"], chosen["receipt"]
        task = as_str(source["task_type"])
        target = as_dict(as_dict(source["metadata"])["target"])
        canonical_gold = symbolic_answer(task, target)
        for representation in REPRESENTATIONS:
            row = copy.deepcopy(source)
            metadata = as_dict(row["metadata"])
            pair_id = f"{VERSION}/{source['id']}"
            answer = project_text(canonical_gold, representation)
            comparison_fields: JsonDict = {
                "schema": SCHEMA, "representation": representation, "pair_id": pair_id,
                "area": OPERATION_AREA[task], "operation": task,
                "family": OPERATION_AREA[task],
                "source": copy.deepcopy(receipt), "source_id": source["id"],
                "source_split": source["split"],
                "mapping_sha256": mapping_sha256(),
                "canonical_query_sha256": canonical_sha256(target["query"]),
                "canonical_fact_sha256": (_static_fact_sha256() if target["state"] is None
                                          else canonical_sha256(target["state"])),
                "canonical_answer": canonical_gold, "answer": answer,
                "comparison_position": len(rows),
            }
            metadata.update(comparison_fields)
            messages: JsonList = [
                {"role": "user", "content": comparison_prompt(task, target, representation)},
                {"role": "assistant", "content": answer},
            ]
            row.update(schema=SCHEMA, id=f"{pair_id}/{representation}",
                       row_id=f"{pair_id}/{representation}", messages=messages)
            rows.append(row)
    return rows
