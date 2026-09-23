"""Response scoring, row validation, and paired representation summaries."""

from __future__ import annotations

import re
from collections import Counter
from typing import cast

from data_pipeline.board_recognition.sources import canonical_sha256
from sft.board.board_fluency_scoring import _strip_transport
from sft.board.coordinate_comparison._constants import (
    COMPARISON_FIELDS,
    COMPLETE_TEST_TASKS,
    COORDINATE_ATOM,
    INCIDENCE_RELATIONS,
    OPERATION_AREA,
    PAIR_QUOTAS,
    REPRESENTATIONS,
    SCHEMA,
    SOURCE_ROW_FIELDS,
    VERSION,
)
from sft.board.coordinate_comparison._mapping import (
    _compact,
    _inverse_mapping,
    _mapping,
    _require,
    _same,
    _static_fact_sha256,
    mapping_sha256,
    project_text,
)
from sft.board.coordinate_comparison._rows import (
    _relation_key,
    _required_cells,
    _source_row,
    comparison_prompt,
)
from sft.board.symbolic_board_tasks import score_symbolic_task
from sft.json_types import (
    JsonDict,
    JsonLikeDict,
    JsonList,
    as_bool,
    as_dict,
    as_list,
    as_str,
)


def validate_comparison_metadata(metadata: JsonDict) -> JsonDict:
    """Fail closed on contracts; return reconstructed, hash-bound original source row.

    Extra evaluator annotations (e.g. input_mode/eval_variant) are permitted. Only
    the original metadata_keys are used to recompute the source receipt.
    """
    _require(isinstance(metadata, dict), "metadata must be a dictionary")
    try:
        _require(metadata["schema"] == SCHEMA and COMPARISON_FIELDS <= metadata.keys(), "invalid comparison schema/fields")
        task = as_str(metadata["task_type"])
        representation = as_str(metadata["representation"])
        source = as_dict(metadata["source"])
        _require(task in PAIR_QUOTAS and representation in REPRESENTATIONS, "invalid comparison task/representation")
        _require(metadata["operation"] == task
                 and metadata["area"] == metadata["family"] == OPERATION_AREA[task], "comparison area/operation mismatch")
        _require(metadata["mapping_sha256"] == mapping_sha256(), "mapping hash mismatch")
        keys = cast("list[str]", source["metadata_keys"])
        _require(isinstance(keys, list) and all(isinstance(k, str) for k in keys)
                 and keys == sorted(set(keys))
                 and not COMPARISON_FIELDS.intersection(keys), "invalid source metadata keys")
        original_metadata = {key: metadata[key] for key in keys}
        _same(canonical_sha256(original_metadata), source["metadata_sha256"], "source metadata/provenance hash")
        original = _source_row(original_metadata, as_str(source["id"]))
        _same(canonical_sha256(original), source["row_sha256"], "source row hash")
        _require(source["schema"] == original["schema"] and source["split"] == metadata["source_split"] == metadata["split"]
                 and source["id"] == metadata["source_id"], "source identity/split mismatch")
        _require(re.fullmatch(rf"symbolic_board_v2/{source['split']}/{task}/[0-9]{{5}}",
                              as_str(source["id"])) is not None,
                 "invalid original source ID")
        _require(source["file"] == f"{source['split']}.jsonl"
                 and isinstance(source["file_sha256"], str)
                 and re.fullmatch("[0-9a-f]{64}", source["file_sha256"]) is not None,
                 "invalid source file/hash")
        _require(isinstance(source["line_sha256"], str)
                 and re.fullmatch("[0-9a-f]{64}", source["line_sha256"]) is not None,
                 "invalid source line hash")
        line_number = cast("int", source["line"])
        position_index = cast("int", metadata["comparison_position"])
        _require(type(source["line"]) is int
                 and line_number == cast("int", original_metadata["row_position"]) + 1
                 and 1 <= line_number <= 480, "source position mismatch")
        _require(metadata["pair_id"] == f"{VERSION}/{source['id']}", "pair identity mismatch")
        _require(type(metadata["comparison_position"]) is int and 0 <= position_index < 400,
                 "invalid comparison position")
        _require(REPRESENTATIONS[position_index % 2] == representation,
                 "comparison position/representation mismatch")
        target = as_dict(metadata["target"])
        gold = as_str(as_dict(as_list(original["messages"])[1])["content"])
        _same(metadata["canonical_query_sha256"], canonical_sha256(target["query"]), "canonical query hash")
        _same(metadata["canonical_fact_sha256"], _static_fact_sha256() if target["state"] is None
              else canonical_sha256(target["state"]), "canonical fact hash")
        _same(source["prompt_sha256"],
              canonical_sha256(as_dict(as_list(original["messages"])[0])["content"]),
              "source prompt hash")
        _same(source["answer_sha256"], canonical_sha256(gold), "source answer hash")
        _require(metadata["canonical_answer"] == gold
                 and metadata["answer"] == project_text(gold, representation), "gold projection mismatch")
        return original
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed comparison metadata: {exc}") from exc


def _parse_response(text: str, metadata: JsonDict) -> str:
    target = as_dict(metadata["target"])
    task, q = as_str(metadata["task_type"]), as_dict(target["query"])
    if task == "symbolic_direction":
        _require(text in {"yes", "no"}, "expected yes or no")
        return text
    if text == "NONE":
        _require(task != "symbolic_direction_choice", "choice requires an offered entity")
        return text
    atoms = text.split()
    _require(bool(atoms) and len(atoms) == len(set(atoms)), "empty or duplicate answer")
    if task == "symbolic_piece_owner":
        _require(len(atoms) == 1
                 and atoms[0] in as_list(as_dict(target["state"])["colors"]),
                 "expected one participant color")
        return atoms[0]
    family = (as_str(q["a"])[1] if task == "symbolic_direction_choice" else
              as_str(q["token"])[1] if task == "symbolic_neighbors" else
              as_str(q["family"]) if task == "symbolic_incidence" else
              "N" if task == "symbolic_owned_nodes" else "E")
    if metadata["representation"] == "coordinates":
        _require(all(COORDINATE_ATOM.fullmatch(a) and a in _inverse_mapping() for a in atoms),
                 "expected known integer-coordinate atoms only")
        atoms = [_inverse_mapping()[a] for a in atoms]
    else:
        _require(all(a in _mapping() for a in atoms), "expected known atlas atoms only")
    _require(all(a[1] == family for a in atoms), "wrong entity family")
    if task == "symbolic_direction_choice":
        _require(len(atoms) == 1 and atoms[0] in as_list(q["choices"]),
                 "expected one offered entity")
    return " ".join(sorted(atoms))


def score_coordinate_comparison(expected: str, response: str,
                                metadata: JsonDict) -> JsonLikeDict | None:
    """Exact-schema dispatch, strict syntax, then the existing recomputing oracle.

    Bad metadata/gold raises ValueError. Bad predictions receive no credit. Only
    known trailing transport suffixes and surrounding whitespace are ignored.
    """
    _require(isinstance(metadata, dict), "metadata must be a dictionary")
    if metadata.get("schema") != SCHEMA:
        return None
    original = validate_comparison_metadata(metadata)
    _require(isinstance(expected, str) and expected == metadata["answer"], "expected gold projection mismatch")
    normalized: str | None = response if isinstance(response, str) else None
    canonical: str | None = None
    error: str | None = None
    correct, valid = False, False
    try:
        _require(isinstance(response, str), "response must be text")
        normalized = _strip_transport(response)
        canonical = _parse_response(normalized, metadata)
        valid = True
        symbolic = as_dict(score_symbolic_task(as_str(metadata["canonical_answer"]),
                                               canonical, as_dict(original["metadata"])))
        correct = as_bool(symbolic["correct"])
        canonical = as_str(symbolic["response_normalized"])
    except ValueError as exc:
        error = str(exc)
    return {
        "scoring": SCHEMA, "symbolic_scoring": metadata["task_type"],
        "correct": bool(correct), "format_valid": valid, "format_error": error,
        "expected": expected, "expected_normalized": metadata["answer"],
        "canonical_expected_normalized": metadata["canonical_answer"],
        "response_raw": response if isinstance(response, str) else None,
        "raw_response_normalized": normalized,
        "response_normalized": (project_text(cast("str", canonical),
                                             as_str(metadata["representation"]))
                                if valid else normalized),
        "canonical_response_normalized": canonical,
    }


def _pair_identity(metadata: JsonDict) -> JsonDict:
    keys = {as_str(k) for k in as_list(as_dict(metadata["source"])["metadata_keys"])}
    keys |= COMPARISON_FIELDS
    return {key: metadata[key] for key in sorted(keys - {"representation", "answer", "comparison_position"})}


def validate_comparison_rows(rows: list[JsonDict]) -> JsonLikeDict:
    """Recompute every prompt/gold/source receipt and enforce the complete 400-row panel."""
    _require(isinstance(rows, list) and len(rows) == 400, "comparison requires exactly 400 rows")
    ids: list[str] = []
    pairs: dict[str, JsonDict] = {}
    file_hashes: dict[str, str] = {}
    for position, row in enumerate(rows):
        _require(isinstance(row, dict) and row.keys() == SOURCE_ROW_FIELDS, "invalid comparison row keys")
        metadata = as_dict(row["metadata"])
        original = validate_comparison_metadata(metadata)
        _require(metadata.keys() == {as_str(k) for k in
                                     as_list(as_dict(metadata["source"])["metadata_keys"])}
                 | COMPARISON_FIELDS, "unexpected dataset metadata fields")
        _require(row["schema"] == SCHEMA, "row schema mismatch")
        for field in ("task_type", "training_family", "task_role", "split"):
            _same(row[field], original[field], f"row {field}")
        representation = as_str(metadata["representation"])
        pair_id = as_str(metadata["pair_id"])
        _require(metadata["comparison_position"] == position and representation == REPRESENTATIONS[position % 2],
                 "comparison position/alternating order mismatch")
        _require(row["id"] == row["row_id"] == f"{pair_id}/{representation}" and row["id"] not in ids,
                 "duplicate or mismatched row ID")
        ids.append(as_str(row["id"]))
        expected_messages: JsonList = [
            {"role": "user", "content": comparison_prompt(as_str(row["task_type"]),
                                                          as_dict(metadata["target"]),
                                                          representation)},
            {"role": "assistant", "content": metadata["answer"]},
        ]
        _same(row["messages"], expected_messages, "rendered prompt/gold projection")
        answer = as_str(metadata["answer"])
        score = as_dict(score_coordinate_comparison(answer, answer, metadata))
        _require(bool(score["correct"] and score["format_valid"]), "gold roundtrip failed")
        if position % 2:
            previous = as_dict(rows[position - 1]["metadata"])
            _require(previous["pair_id"] == pair_id, "nonadjacent pair")
            _same(_pair_identity(previous), _pair_identity(metadata), "paired source/provenance")
        else:
            _require(pair_id not in pairs, "duplicate pair ID")
            pairs[pair_id] = metadata
        source = as_dict(metadata["source"])
        file_name, file_hash = as_str(source["file"]), as_str(source["file_sha256"])
        _require(file_hashes.setdefault(file_name, file_hash) == file_hash,
                 "inconsistent source file hashes")
    canonical = list(pairs.values())
    _same(dict(Counter(m["operation"] for m in canonical)), PAIR_QUOTAS, "operation quotas")
    for task in COMPLETE_TEST_TASKS:
        _same(sorted(as_str(m["source_id"]) for m in canonical if m["task_type"] == task),
              [f"symbolic_board_v2/test/{task}/{i:05d}" for i in range(32)], "complete test source IDs")
    incidence = [m for m in canonical if m["task_type"] == "symbolic_incidence"]
    _same(dict(Counter(_relation_key(m) for m in incidence)), dict.fromkeys(INCIDENCE_RELATIONS, 2), "incidence quotas")
    validation = [m for m in canonical if m["split"] == "validation"]
    _require(len(validation) == 2 and all(m["task_type"] == "symbolic_incidence" for m in validation),
             "validation source exception quota")
    _same(sorted(as_str(as_dict(as_dict(m["target"])["query"])["token"]) for m in validation),
          ["<P03>", "<P08>"], "validation port queries")
    _require(all(_relation_key(m) == "P->N" for m in validation), "invalid validation relation")
    for task in ("symbolic_owned_nodes", "symbolic_owned_roads", "symbolic_owned_incident_roads"):
        actual = Counter(_compact(m["sampling_cell"]) for m in canonical if m["task_type"] == task)
        expected = {_compact(cell): 1 if task == "symbolic_owned_nodes" else 2 for cell in _required_cells(task)}
        _same(dict(actual), expected, f"{task} sampling quotas")
    states = Counter(as_dict(m["provenance"])["state_id"] for m in canonical if "provenance" in m)
    report: JsonLikeDict = {
        "schema": SCHEMA, "valid": True, "rows": len(rows), "pairs": len(pairs), "ids": ids,
        "pair_ids": list(pairs), "mapping_sha256": mapping_sha256(), "mapping_entities": 154,
        "by_family": dict(Counter(as_str(as_dict(r["metadata"])["family"]) for r in rows)),
        "by_operation": dict(Counter(as_str(as_dict(r["metadata"])["operation"]) for r in rows)),
        "by_representation": dict(
            Counter(as_str(as_dict(r["metadata"])["representation"]) for r in rows)),
        "pair_quotas": dict(PAIR_QUOTAS),
        "pairs_by_split": dict(Counter(as_str(m["split"]) for m in canonical)),
        "incidence_relations": dict(Counter(_relation_key(m) for m in incidence)),
        "source_file_sha256": file_hashes, "unique_dynamic_states": len(states),
        "max_pairs_per_dynamic_state": max(states.values(), default=0),
        "gold_roundtrip_rows": len(rows), "row_order": "adjacent atlas, coordinates per source case",
    }
    return report
