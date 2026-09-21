"""Matched atlas-ID/integer-coordinate inference diagnostics, without trainer imports.

Only ``messages[0]`` is model input. Canonical targets and source receipts stay in
metadata; the crosswalk is a separate artifact. This is a representation-usability
comparison on an atlas-trained checkpoint, not an equal-training-budget experiment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from data_pipeline.board_recognition.sources import canonical_sha256, file_sha256
from sft.board_fluency_scoring import _strip_transport
from sft.symbolic_board_tasks import (
    STATIC_TASKS, atlas_geometry, score_symbolic_task, symbolic_answer,
    symbolic_prompt, symbolic_task_role,
)


SCHEMA = "catan_coordinate_comparison/v1"
VERSION = "coordinate_comparison_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "artifacts/generated/sft/symbolic_board_v2"
REPRESENTATIONS = ("atlas", "coordinates")
PAIR_QUOTAS = {
    "symbolic_direction": 32, "symbolic_direction_choice": 32,
    "symbolic_neighbors": 32, "symbolic_incidence": 16,
    "symbolic_piece_owner": 32, "symbolic_owned_nodes": 24,
    "symbolic_owned_roads": 16, "symbolic_owned_incident_roads": 16,
}
OPERATION_AREA = {
    "symbolic_direction": "direction", "symbolic_direction_choice": "direction",
    "symbolic_neighbors": "adjacency", "symbolic_incidence": "incidence",
    "symbolic_piece_owner": "ownership", "symbolic_owned_nodes": "ownership",
    "symbolic_owned_roads": "ownership", "symbolic_owned_incident_roads": "ownership",
}
INCIDENCE_RELATIONS = ("N->T", "N->E", "N->P", "T->N", "T->E", "E->N", "E->T", "P->N")
COMPLETE_TEST_TASKS = (
    "symbolic_direction", "symbolic_direction_choice", "symbolic_neighbors", "symbolic_piece_owner",
)
ATLAS_ATOM = re.compile(r"<[NTEP][0-9_]+>")
INTEGER = r"(?:0|-?[1-9][0-9]*)"
COORDINATE_ATOM = re.compile(rf"[NTEP]\({INTEGER},{INTEGER}\)")
GEOMETRY_CONVENTION = (
    "Coordinates use a scaled integer layout: x increases right and y increases down. "
    "N(x,y) is a node; T(x,y) is a land tile center; E(x,y) is a road-edge midpoint; "
    "P(x,y) is the midpoint of a port's two attached nodes. "
    "A tile's six corner offsets are (0,-4),(2,-2),(2,2),(0,4),(-2,2),(-2,-2). "
    "Road edges join consecutive tile corners; their coordinates are the arithmetic "
    "midpoints of their endpoint nodes. Ports attach to the two endpoints of the "
    "coastal edge at their midpoint. Entity types distinguish coincident points. "
    "Use decimal integers without leading zeros, plus signs, or negative zero."
)
ATOM_INSTRUCTION = "Return entity identifiers as whitespace-free typed atoms exactly as shown."
COMPARISON_FIELDS = frozenset({
    "schema", "representation", "pair_id", "area", "operation", "family", "source",
    "source_id", "source_split", "mapping_sha256", "canonical_query_sha256",
    "canonical_fact_sha256", "canonical_answer", "answer", "comparison_position",
})
SOURCE_ROW_FIELDS = frozenset({
    "schema", "id", "row_id", "task_type", "training_family", "task_role", "split", "messages", "metadata",
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _same(actual: object, expected: object, label: str) -> None:
    _require(_compact(actual) == _compact(expected), f"{label} mismatch")


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line, object_pairs_hook=_unique_object) for line in handle if line.strip()]


@lru_cache(maxsize=1)
def _geometry() -> dict:
    return atlas_geometry()


@lru_cache(maxsize=1)
def _mapping() -> dict[str, str]:
    atlas = _geometry()
    base = atlas["positions"]
    points = {token: (2 * x, 2 * y) for token, (x, y) in base.items()}
    for edge, (a, b) in atlas["edges"].items():
        points[edge] = tuple(base[a][axis] + base[b][axis] for axis in (0, 1))
    for port in atlas["raw"]["ports"]:
        a, b = (f"<N{nid:02d}>" for nid in port["attached_nodes"])
        points[port["token"]] = tuple(base[a][axis] + base[b][axis] for axis in (0, 1))
    # The inventory matches the complete dynamic board's entity order.
    order = [t for family in "TNEP" for t in sorted(atlas["tokens"]) if t[1] == family]
    mapping = {t: f"{t[1]}({points[t][0]},{points[t][1]})" for t in order}
    _require(len(mapping) == len(set(mapping.values())) == 154, "coordinate mapping is not bijective")
    _require(all(COORDINATE_ATOM.fullmatch(v) for v in mapping.values()), "noninteger coordinate mapping")
    return mapping


def coordinate_mapping() -> dict[str, str]:
    """Detached 154-entry atlas-to-coordinate mapping in model inventory order."""
    return dict(_mapping())


@lru_cache(maxsize=1)
def _inverse_mapping() -> dict[str, str]:
    return {point: token for token, point in _mapping().items()}


def mapping_artifact() -> dict:
    """Crosswalk for offline inspection, never included in a model prompt."""
    return {
        "schema": SCHEMA, "kind": "mapping", "scale": 2,
        "convention": GEOMETRY_CONVENTION, "inventory_order": list(_mapping()),
        "atlas_to_coordinates": coordinate_mapping(),
        "coordinates_to_atlas": dict(_inverse_mapping()),
        "added_tokenizer_tokens": [],
    }


@lru_cache(maxsize=1)
def mapping_sha256() -> str:
    # Hash the payload; never include this digest in its own input.
    return canonical_sha256(mapping_artifact())


@lru_cache(maxsize=1)
def _static_fact_sha256() -> str:
    atlas = _geometry()
    facts = {key: {t: sorted(v) for t, v in atlas[key].items()}
             for key in ("graph", "touching", "tile_neighbors")}
    facts.update(positions=atlas["positions"], edges=atlas["edges"])
    return canonical_sha256(facts)


def project_text(text: str, representation: str) -> str:
    """Replace entity atoms only, retaining record order and all other facts."""
    _require(representation in REPRESENTATIONS and isinstance(text, str), "invalid text projection")
    if representation == "atlas":
        return text

    def replace(match: re.Match) -> str:
        _require(match[0] in _mapping(), f"unknown atlas atom: {match[0]}")
        return _mapping()[match[0]]

    return ATLAS_ATOM.sub(replace, text)


def comparison_prompt(task: str, target: dict, representation: str) -> str:
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


def _sampling_cell(task: str, target: dict, gold: str) -> dict:
    q, state = target["query"], target["state"]
    mode = q["piece"] if task == "symbolic_owned_nodes" else (
        q["token"][1] if task == "symbolic_piece_owner" else "all")
    return {"mode": mode, "position": state["colors"].index(q["color"]) if "color" in q else None,
            "polarity": "negative" if gold == "NONE" else "positive"}


@lru_cache(maxsize=1024)
def _source_semantics(metadata_json: str) -> tuple[str, str]:
    metadata = json.loads(metadata_json)
    task, target, split = metadata["task_type"], metadata["target"], metadata["split"]
    _require(task in PAIR_QUOTAS and split in {"test", "validation"}, "invalid source task/split")
    _require(metadata["training_family"] == task
             and metadata["task_role"] == symbolic_task_role(task, split), "source task role mismatch")
    gold = symbolic_answer(task, target)
    _same(metadata["query_sha256"], canonical_sha256(target["query"]), "source query hash")
    if task in STATIC_TASKS:
        _require("provenance" not in metadata, "static source has dynamic provenance")
    else:
        provenance = metadata["provenance"]
        _require(provenance["split"] == provenance["source"]["split"] == split,
                 "source provenance split mismatch")
        for key in ("state_id", "board_map_sha256", "board_fact_sha256"):
            _require(isinstance(provenance[key], str) and bool(provenance[key]), "missing source provenance")
        _same(metadata["sampling_cell"], _sampling_cell(task, target, gold), "source sampling cell")
    if task in ("symbolic_direction", "symbolic_direction_choice"):
        q = target["query"]
        _same(metadata["directional_pair"], sorted((q["a"], q["b"])), "source directional pair")
        _same(metadata["known_training_exposure"], [], "heldout directional exposure")
    return symbolic_prompt(task, target), gold


def _source_row(metadata: dict, source_id: str) -> dict:
    prompt, gold = _source_semantics(_compact(metadata))
    return {
        "schema": "catan_symbolic_board_row/v2", "id": source_id, "row_id": source_id,
        **{key: metadata[key] for key in ("task_type", "training_family", "task_role", "split")},
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": gold}],
        "metadata": metadata,
    }


def load_source_rows(root: Path = DEFAULT_SOURCE) -> tuple[list[dict], dict]:
    """Verify both source split files against their manifest, retaining raw-line receipts."""
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(), object_pairs_hook=_unique_object)
    _require(manifest["schema"] == "catan_symbolic_board_manifest/v2", "invalid source manifest")
    sources, hashes = [], {"manifest.json": file_sha256(manifest_path)}
    for split in ("test", "validation"):
        path = root / f"{split}.jsonl"
        digest = file_sha256(path)
        declaration = manifest["files"][split]
        _require(digest == declaration["sha256"], f"source file hash mismatch: {split}")
        raw_lines = path.read_bytes().splitlines(keepends=True)
        _require(len(raw_lines) == declaration["rows"] == manifest["counts"][split] == 480,
                 f"source row quota mismatch: {split}")
        hashes[path.name] = digest
        seen = set()
        for line, raw in enumerate(raw_lines, 1):
            row = json.loads(raw, object_pairs_hook=_unique_object)
            _require(row["id"] not in seen, "duplicate source ID")
            seen.add(row["id"])
            _require(row["split"] == row["metadata"]["split"] == split
                     and row["metadata"]["row_position"] == line - 1, "source split/position mismatch")
            if row["task_type"] not in PAIR_QUOTAS:
                continue
            _require(not COMPARISON_FIELDS.intersection(row["metadata"]), "reserved source metadata keys")
            _same(row, _source_row(row["metadata"], row["id"]), "source prompt/gold/declarations")
            sources.append({"row": row, "receipt": {
                "id": row["id"], "split": split, "schema": row["schema"], "file": path.name,
                "file_sha256": digest, "line": line,
                "line_sha256": hashlib.sha256(raw).hexdigest(),
                "row_sha256": canonical_sha256(row),
                "metadata_sha256": canonical_sha256(row["metadata"]),
                "metadata_keys": sorted(row["metadata"]),
                "prompt_sha256": canonical_sha256(row["messages"][0]["content"]),
                "answer_sha256": canonical_sha256(row["messages"][1]["content"]),
            }})
    return sources, {"root": str(root), "sha256": hashes,
                     "original_source_hashes": copy.deepcopy(manifest["source_hashes"])}


def _required_cells(task: str) -> list[dict]:
    modes = ("building", "settlement", "city") if task == "symbolic_owned_nodes" else ("all",)
    return [dict(mode=mode, position=position, polarity=polarity)
            for position in range(4) for mode in modes for polarity in ("positive", "negative")]


def select_source_cases(sources: list[dict]) -> list[dict]:
    """Deterministic exact quotas; least-used source state, then source ID breaks ties."""
    selected, uses = [], Counter()
    test = [s for s in sources if s["row"]["split"] == "test"]

    def take(pool: list[dict], count: int, label: str, *, complete: bool = False) -> None:
        _require(len(pool) == count if complete else len(pool) >= count, f"missing source quota: {label}")
        pool = list(pool)
        for _ in range(count):
            chosen = min(pool, key=lambda s: (
                uses[s["row"]["metadata"].get("provenance", {}).get("state_id")], s["row"]["id"]))
            pool.remove(chosen)
            selected.append(chosen)
            state_id = chosen["row"]["metadata"].get("provenance", {}).get("state_id")
            if state_id is not None:
                uses[state_id] += 1

    for task in COMPLETE_TEST_TASKS:
        take([s for s in test if s["row"]["task_type"] == task], 32, task, complete=True)
    for relation in INCIDENCE_RELATIONS:
        if relation == "P->N":
            pool = [s for s in sources if s["row"]["split"] == "validation"
                    and s["row"]["task_type"] == "symbolic_incidence"
                    and s["row"]["metadata"]["target"]["query"] in (
                        {"token": "<P03>", "family": "N"}, {"token": "<P08>", "family": "N"})]
            take(pool, 2, relation, complete=True)
        else:
            pool = [s for s in test if s["row"]["task_type"] == "symbolic_incidence"
                    and _relation_key(s["row"]["metadata"]) == relation]
            take(pool, 2, relation)
    for task in ("symbolic_owned_nodes", "symbolic_owned_roads", "symbolic_owned_incident_roads"):
        for cell in _required_cells(task):
            pool = [s for s in test if s["row"]["task_type"] == task
                    and s["row"]["metadata"]["sampling_cell"] == cell]
            take(pool, 1 if task == "symbolic_owned_nodes" else 2, f"{task}/{cell}")
    _require(len(selected) == len({s["row"]["id"] for s in selected}) == 200, "source selection IDs/quota")
    # Preserve original physical order within each source split.
    return sorted(selected, key=lambda s: (s["row"]["split"] != "test", s["receipt"]["line"]))


def _relation_key(metadata: dict) -> str:
    q = metadata["target"]["query"]
    return q["token"][1] + "->" + q["family"]


def build_comparison_rows(sources: list[dict]) -> list[dict]:
    rows = []
    for chosen in select_source_cases(sources):
        source, receipt = chosen["row"], chosen["receipt"]
        task, target = source["task_type"], source["metadata"]["target"]
        canonical_gold = symbolic_answer(task, target)
        for representation in REPRESENTATIONS:
            row = copy.deepcopy(source)
            metadata = row["metadata"]
            pair_id = f"{VERSION}/{source['id']}"
            answer = project_text(canonical_gold, representation)
            metadata.update(
                schema=SCHEMA, representation=representation, pair_id=pair_id,
                area=OPERATION_AREA[task], operation=task, family=OPERATION_AREA[task],
                source=copy.deepcopy(receipt), source_id=source["id"], source_split=source["split"],
                mapping_sha256=mapping_sha256(), canonical_query_sha256=canonical_sha256(target["query"]),
                canonical_fact_sha256=(_static_fact_sha256() if target["state"] is None
                                       else canonical_sha256(target["state"])),
                canonical_answer=canonical_gold, answer=answer, comparison_position=len(rows),
            )
            row.update(schema=SCHEMA, id=f"{pair_id}/{representation}", row_id=f"{pair_id}/{representation}",
                       messages=[{"role": "user", "content": comparison_prompt(task, target, representation)},
                                 {"role": "assistant", "content": answer}])
            rows.append(row)
    return rows


def validate_comparison_metadata(metadata: dict) -> dict:
    """Fail closed on contracts; return reconstructed, hash-bound original source row.

    Extra evaluator annotations (e.g. input_mode/eval_variant) are permitted. Only
    the original metadata_keys are used to recompute the source receipt.
    """
    _require(isinstance(metadata, dict), "metadata must be a dictionary")
    try:
        _require(metadata["schema"] == SCHEMA and COMPARISON_FIELDS <= metadata.keys(), "invalid comparison schema/fields")
        task, representation, source = metadata["task_type"], metadata["representation"], metadata["source"]
        _require(task in PAIR_QUOTAS and representation in REPRESENTATIONS, "invalid comparison task/representation")
        _require(metadata["operation"] == task
                 and metadata["area"] == metadata["family"] == OPERATION_AREA[task], "comparison area/operation mismatch")
        _require(metadata["mapping_sha256"] == mapping_sha256(), "mapping hash mismatch")
        keys = source["metadata_keys"]
        _require(isinstance(keys, list) and all(isinstance(k, str) for k in keys)
                 and keys == sorted(set(keys)) and not COMPARISON_FIELDS.intersection(keys), "invalid source metadata keys")
        original_metadata = {key: metadata[key] for key in keys}
        _same(canonical_sha256(original_metadata), source["metadata_sha256"], "source metadata/provenance hash")
        original = _source_row(original_metadata, source["id"])
        _same(canonical_sha256(original), source["row_sha256"], "source row hash")
        _require(source["schema"] == original["schema"] and source["split"] == metadata["source_split"] == metadata["split"]
                 and source["id"] == metadata["source_id"], "source identity/split mismatch")
        _require(re.fullmatch(rf"symbolic_board_v2/{source['split']}/{task}/[0-9]{{5}}", source["id"]) is not None,
                 "invalid original source ID")
        _require(source["file"] == f"{source['split']}.jsonl"
                 and isinstance(source["file_sha256"], str) and re.fullmatch("[0-9a-f]{64}", source["file_sha256"]),
                 "invalid source file/hash")
        _require(isinstance(source["line_sha256"], str) and re.fullmatch("[0-9a-f]{64}", source["line_sha256"]),
                 "invalid source line hash")
        _require(type(source["line"]) is int and source["line"] == original_metadata["row_position"] + 1
                 and 1 <= source["line"] <= 480, "source position mismatch")
        _require(metadata["pair_id"] == f"{VERSION}/{source['id']}", "pair identity mismatch")
        _require(type(metadata["comparison_position"]) is int and 0 <= metadata["comparison_position"] < 400,
                 "invalid comparison position")
        _require(REPRESENTATIONS[metadata["comparison_position"] % 2] == representation,
                 "comparison position/representation mismatch")
        target = metadata["target"]
        gold = original["messages"][1]["content"]
        _same(metadata["canonical_query_sha256"], canonical_sha256(target["query"]), "canonical query hash")
        _same(metadata["canonical_fact_sha256"], _static_fact_sha256() if target["state"] is None
              else canonical_sha256(target["state"]), "canonical fact hash")
        _same(source["prompt_sha256"], canonical_sha256(original["messages"][0]["content"]), "source prompt hash")
        _same(source["answer_sha256"], canonical_sha256(gold), "source answer hash")
        _require(metadata["canonical_answer"] == gold
                 and metadata["answer"] == project_text(gold, representation), "gold projection mismatch")
        return original
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed comparison metadata: {exc}") from exc


def _parse_response(text: str, metadata: dict) -> str:
    task, q = metadata["task_type"], metadata["target"]["query"]
    if task == "symbolic_direction":
        _require(text in {"yes", "no"}, "expected yes or no")
        return text
    if text == "NONE":
        _require(task != "symbolic_direction_choice", "choice requires an offered entity")
        return text
    atoms = text.split()
    _require(bool(atoms) and len(atoms) == len(set(atoms)), "empty or duplicate answer")
    if task == "symbolic_piece_owner":
        _require(len(atoms) == 1 and atoms[0] in metadata["target"]["state"]["colors"], "expected one participant color")
        return atoms[0]
    family = (q["a"][1] if task == "symbolic_direction_choice" else
              q["token"][1] if task == "symbolic_neighbors" else
              q["family"] if task == "symbolic_incidence" else
              "N" if task == "symbolic_owned_nodes" else "E")
    if metadata["representation"] == "coordinates":
        _require(all(COORDINATE_ATOM.fullmatch(a) and a in _inverse_mapping() for a in atoms),
                 "expected known integer-coordinate atoms only")
        atoms = [_inverse_mapping()[a] for a in atoms]
    else:
        _require(all(a in _mapping() for a in atoms), "expected known atlas atoms only")
    _require(all(a[1] == family for a in atoms), "wrong entity family")
    if task == "symbolic_direction_choice":
        _require(len(atoms) == 1 and atoms[0] in q["choices"], "expected one offered entity")
    return " ".join(sorted(atoms))


def score_coordinate_comparison(expected: str, response: str, metadata: dict) -> dict | None:
    """Exact-schema dispatch, strict syntax, then the existing recomputing oracle.

    Bad metadata/gold raises ValueError. Bad predictions receive no credit. Only
    known trailing transport suffixes and surrounding whitespace are ignored.
    """
    _require(isinstance(metadata, dict), "metadata must be a dictionary")
    if metadata.get("schema") != SCHEMA:
        return None
    original = validate_comparison_metadata(metadata)
    _require(isinstance(expected, str) and expected == metadata["answer"], "expected gold projection mismatch")
    normalized, canonical, error = response if isinstance(response, str) else None, None, None
    correct, valid = False, False
    try:
        _require(isinstance(response, str), "response must be text")
        normalized = _strip_transport(response)
        canonical = _parse_response(normalized, metadata)
        valid = True
        symbolic = score_symbolic_task(metadata["canonical_answer"], canonical, original["metadata"])
        correct = symbolic["correct"]
        canonical = symbolic["response_normalized"]
    except ValueError as exc:
        error = str(exc)
    return {
        "scoring": SCHEMA, "symbolic_scoring": metadata["task_type"],
        "correct": bool(correct), "format_valid": valid, "format_error": error,
        "expected": expected, "expected_normalized": metadata["answer"],
        "canonical_expected_normalized": metadata["canonical_answer"],
        "response_raw": response if isinstance(response, str) else None,
        "raw_response_normalized": normalized,
        "response_normalized": project_text(canonical, metadata["representation"]) if valid else normalized,
        "canonical_response_normalized": canonical,
    }


def _pair_identity(metadata: dict) -> dict:
    keys = set(metadata["source"]["metadata_keys"]) | COMPARISON_FIELDS
    return {key: metadata[key] for key in sorted(keys - {"representation", "answer", "comparison_position"})}


def validate_comparison_rows(rows: list[dict]) -> dict:
    """Recompute every prompt/gold/source receipt and enforce the complete 400-row panel."""
    _require(isinstance(rows, list) and len(rows) == 400, "comparison requires exactly 400 rows")
    ids, pairs, file_hashes = [], {}, {}
    for position, row in enumerate(rows):
        _require(isinstance(row, dict) and row.keys() == SOURCE_ROW_FIELDS, "invalid comparison row keys")
        metadata = row["metadata"]
        original = validate_comparison_metadata(metadata)
        _require(metadata.keys() == set(metadata["source"]["metadata_keys"]) | COMPARISON_FIELDS,
                 "unexpected dataset metadata fields")
        _require(row["schema"] == SCHEMA, "row schema mismatch")
        for field in ("task_type", "training_family", "task_role", "split"):
            _same(row[field], original[field], f"row {field}")
        representation, pair_id = metadata["representation"], metadata["pair_id"]
        _require(metadata["comparison_position"] == position and representation == REPRESENTATIONS[position % 2],
                 "comparison position/alternating order mismatch")
        _require(row["id"] == row["row_id"] == f"{pair_id}/{representation}" and row["id"] not in ids,
                 "duplicate or mismatched row ID")
        ids.append(row["id"])
        expected_messages = [
            {"role": "user", "content": comparison_prompt(row["task_type"], metadata["target"], representation)},
            {"role": "assistant", "content": metadata["answer"]},
        ]
        _same(row["messages"], expected_messages, "rendered prompt/gold projection")
        score = score_coordinate_comparison(metadata["answer"], metadata["answer"], metadata)
        _require(score["correct"] and score["format_valid"], "gold roundtrip failed")
        if position % 2:
            _require(rows[position - 1]["metadata"]["pair_id"] == pair_id, "nonadjacent pair")
            _same(_pair_identity(rows[position - 1]["metadata"]), _pair_identity(metadata), "paired source/provenance")
        else:
            _require(pair_id not in pairs, "duplicate pair ID")
            pairs[pair_id] = metadata
        source = metadata["source"]
        _require(file_hashes.setdefault(source["file"], source["file_sha256"]) == source["file_sha256"],
                 "inconsistent source file hashes")
    canonical = list(pairs.values())
    _same(dict(Counter(m["operation"] for m in canonical)), PAIR_QUOTAS, "operation quotas")
    for task in COMPLETE_TEST_TASKS:
        _same(sorted(m["source_id"] for m in canonical if m["task_type"] == task),
              [f"symbolic_board_v2/test/{task}/{i:05d}" for i in range(32)], "complete test source IDs")
    incidence = [m for m in canonical if m["task_type"] == "symbolic_incidence"]
    _same(dict(Counter(_relation_key(m) for m in incidence)), dict.fromkeys(INCIDENCE_RELATIONS, 2), "incidence quotas")
    validation = [m for m in canonical if m["split"] == "validation"]
    _require(len(validation) == 2 and all(m["task_type"] == "symbolic_incidence" for m in validation),
             "validation source exception quota")
    _same(sorted(m["target"]["query"]["token"] for m in validation), ["<P03>", "<P08>"], "validation port queries")
    _require(all(_relation_key(m) == "P->N" for m in validation), "invalid validation relation")
    for task in ("symbolic_owned_nodes", "symbolic_owned_roads", "symbolic_owned_incident_roads"):
        actual = Counter(_compact(m["sampling_cell"]) for m in canonical if m["task_type"] == task)
        expected = {_compact(cell): 1 if task == "symbolic_owned_nodes" else 2 for cell in _required_cells(task)}
        _same(dict(actual), expected, f"{task} sampling quotas")
    states = Counter(m["provenance"]["state_id"] for m in canonical if "provenance" in m)
    return {
        "schema": SCHEMA, "valid": True, "rows": len(rows), "pairs": len(pairs), "ids": ids,
        "pair_ids": list(pairs), "mapping_sha256": mapping_sha256(), "mapping_entities": 154,
        "by_family": dict(Counter(r["metadata"]["family"] for r in rows)),
        "by_operation": dict(Counter(r["metadata"]["operation"] for r in rows)),
        "by_representation": dict(Counter(r["metadata"]["representation"] for r in rows)),
        "pair_quotas": dict(PAIR_QUOTAS), "pairs_by_split": dict(Counter(m["split"] for m in canonical)),
        "incidence_relations": dict(Counter(_relation_key(m) for m in incidence)),
        "source_file_sha256": file_hashes, "unique_dynamic_states": len(states),
        "max_pairs_per_dynamic_state": max(states.values(), default=0),
        "gold_roundtrip_rows": len(rows), "row_order": "adjacent atlas, coordinates per source case",
    }


def _summarize_pairs(pairs: dict) -> dict:
    outcomes = {name: [] for name in ("atlas_only", "coordinates_only", "both", "neither")}
    by_representation = {rep: {"rows": len(pairs), "correct": 0, "format_valid": 0,
                               "format_invalid_ids": []} for rep in REPRESENTATIONS}
    for pair_id, arms in pairs.items():
        a, c = (arms[rep]["score"]["correct"] for rep in REPRESENTATIONS)
        outcome = "both" if a and c else "atlas_only" if a else "coordinates_only" if c else "neither"
        outcomes[outcome].append(pair_id)
        for rep, item in arms.items():
            report, score = by_representation[rep], item["score"]
            report["correct"] += int(score["correct"])
            report["format_valid"] += int(score["format_valid"])
            if not score["format_valid"]:
                report["format_invalid_ids"].append(item["id"])
    for report in by_representation.values():
        report["accuracy"] = report["correct"] / len(pairs) if pairs else None
        report["format_invalid"] = len(pairs) - report["format_valid"]
        report["format_valid_rate"] = report["format_valid"] / len(pairs) if pairs else None
    return {
        "pairs": len(pairs), "rows": 2 * len(pairs), "pair_ids": list(pairs),
        **{name: len(values) for name, values in outcomes.items()},
        "outcome_pair_ids": outcomes, "by_representation": by_representation,
        "coordinates_minus_atlas_accuracy": ((len(outcomes["coordinates_only"]) - len(outcomes["atlas_only"])) / len(pairs)
                                              if pairs else None),
    }


def paired_summary(records: list[dict]) -> dict:
    """Require complete pairs (subsets allowed), rescore raw outputs, retain outcome IDs."""
    _require(isinstance(records, list), "records must be a list")
    pairs, seen = {}, set()
    for record in records:
        _require(isinstance(record, dict) and {"id", "metadata", "expected", "response"} <= record.keys(),
                 "incomplete comparison record")
        metadata = record["metadata"]
        score = score_coordinate_comparison(record["expected"], record["response"], metadata)
        _require(score is not None, "noncomparison record in paired summary")
        pair_id, representation = metadata["pair_id"], metadata["representation"]
        _require(record["id"] == f"{pair_id}/{representation}" and record["id"] not in seen, "duplicate/mismatched record ID")
        seen.add(record["id"])
        arms = pairs.setdefault(pair_id, {})
        _require(representation not in arms, "duplicate pair arm")
        arms[representation] = {"id": record["id"], "metadata": metadata, "score": score}
    by_task, by_area = defaultdict(dict), defaultdict(dict)
    for pair_id, arms in pairs.items():
        _require(set(arms) == set(REPRESENTATIONS), f"incomplete pair: {pair_id}")
        atlas, coordinates = (arms[rep]["metadata"] for rep in REPRESENTATIONS)
        _same(_pair_identity(atlas), _pair_identity(coordinates), "paired source/provenance")
        _require(coordinates["comparison_position"] == atlas["comparison_position"] + 1,
                 "paired original positions mismatch")
        by_task[atlas["task_type"]][pair_id] = arms
        by_area[atlas["area"]][pair_id] = arms
    return {
        "schema": SCHEMA, "rescored_from_raw": True, **_summarize_pairs(pairs),
        "by_task": {task: _summarize_pairs(group) for task, group in sorted(by_task.items())},
        "by_area": {area: _summarize_pairs(group) for area, group in sorted(by_area.items())},
    }
