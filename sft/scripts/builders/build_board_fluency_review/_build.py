"""Build the bounded, text-only 200-example board-fluency human review batch.

This is a review schema, not an admitted trainer task schema. Only the first
MAX_SOURCE_ROWS physical source lines are read; the full source hash is merely
reported from its manifest. Source states and original provenance stay intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from typing import cast

from data_pipeline.board_recognition.sources import (
    canonical_sha256,
    file_sha256,
    repository_relative,
)
from evals.catan_board_bench.annotations import contract_to_render_state
from sft.board.symbolic_board_tasks import atlas_geometry, decode_state
from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_list, as_str, loads_json

from ._audit import audit_contract, reference_answer, validate_render
from ._candidates import select
from ._facts import Facts, richness
from ._prompt import answer, prompt, question
from ._sources import (
    BUILD_DEPENDENCIES,
    FAMILIES,
    MAX_SOURCE_ROWS,
    OPERATION_FAMILY,
    OUTPUT,
    OWN_FILES,
    SCHEMA,
    SEED,
    VERSION,
    Donor,
    Message,
    PreviewRow,
    QueryDict,
    ReviewRow,
    SourceRef,
    compact,
    load_prefix,
    require,
)


def build(donors: list[Donor], source_audit: JsonLikeDict, seed: int) -> dict[str, bytes]:
    atlas = atlas_geometry()
    selected, selection = select(donors, atlas, seed)
    rows: list[ReviewRow] = []
    preview_rows: list[PreviewRow] = []
    boards: dict[str, JsonDict] = {}
    contracts: dict[str, JsonDict] = {}
    checks: Counter[str] = Counter()
    op_indices: Counter[str] = Counter()
    for candidate in selected:
        donor, op, q = candidate.donor, candidate.operation, candidate.query
        family = OPERATION_FAMILY[op]
        if donor.state_hash not in contracts:
            contracts[donor.state_hash] = audit_contract(donor)
            checks["original_contract_sha_state_source_map_verified"] += 1
            checks["original_training_donor_answer_recomputed"] += 1
        contract = contracts[donor.state_hash]
        facts = Facts(decode_state(donor.state), atlas)
        gold, text = answer(facts, op, q), question(op, q)
        require(gold == reference_answer(candidate, contract, checks),
                f"answer crosscheck failed: {op} / source line {donor.line} / {q}")
        require(canonical_sha256(donor.state) == donor.state_hash, "donor state was mutated")
        state_id = as_str(donor.provenance["state_id"])
        if state_id not in boards:
            # JSON round trip turns the RenderState TypedDict into the same plain JSON object.
            boards[state_id] = as_dict(loads_json(json.dumps(contract_to_render_state(contract))))
            validate_render(contract, boards[state_id], donor.state)
            checks["exact_render_state_crosschecks"] += 1
        row_id = f"{VERSION}/{family}/{op}/{op_indices[op]:03d}"
        op_indices[op] += 1
        source = SourceRef(
            row_id=as_str(donor.row["row_id"]), line=donor.line,
            density=as_str(donor.provenance["density_bin"]),
            map_id=as_str(donor.provenance["board_map_sha256"]),
            source_kind=as_str(as_dict(donor.provenance["source"])["kind"]),
        )
        model_prompt = prompt(donor.state, text)
        query: QueryDict = {"operation": op, **q}
        metadata: JsonLikeDict = {
            "review_only": True, "admitted_for_training": False,
            "class": "board_fluency", "family": family, "operation": op,
            "state_id": state_id, "question": text, "answer": gold,
            "target": {"state": donor.state, "query": query},
            "source": cast("JsonLikeDict", source), "provenance": donor.provenance,
            "state_sha256": donor.state_hash, "query_sha256": canonical_sha256(query),
            "donor_physical_line_sha256": donor.line_sha256,
            "donor_declarations": {key: donor.row[key] for key in
                                   ("schema", "split", "task_role", "task_type", "training_family")},
            "selection_stratum": candidate.stratum,
            "hypothetical_query_only": op in (
                "road_removal_connectivity", "settlement_upgrade_production", "robber_move_production"),
        }
        rows.append(ReviewRow(
            schema=SCHEMA, id=row_id, row_id=row_id,
            messages=[Message(role="user", content=model_prompt),
                      Message(role="assistant", content=gold)],
            metadata=metadata,
        ))
        preview_rows.append(PreviewRow(
            row_id=row_id, state_id=state_id, family=family, operation=op,
            question=text, answer=gold, prompt=model_prompt, source=source,
        ))

    counts_family = dict(Counter(r["family"] for r in preview_rows))
    counts_operation = dict(Counter(r["operation"] for r in preview_rows))
    require(len(rows) == 200 and counts_family == dict.fromkeys(FAMILIES, 40),
            "hard family count assertion failed: expected exactly 40 in each of five families")
    require(counts_operation == dict.fromkeys(OPERATION_FAMILY, 10), "operation counts must all equal 10")
    require(len({r["row_id"] for r in rows}) == 200, "duplicate stable row IDs")
    require(len({(as_dict(r["metadata"])["state_sha256"],
                  compact(as_dict(as_dict(r["metadata"])["target"])["query"]))
                 for r in rows}) == 200, "duplicate canonical state/query padding")
    require(len({r["messages"][0]["content"] for r in rows}) == 200, "duplicate model prompts")
    for row, view in zip(rows, preview_rows, strict=True):
        m = as_dict(row["metadata"])
        require([msg["role"] for msg in row["messages"]] == ["user", "assistant"], "message roles")
        require(all(set(msg) == {"role", "content"} and isinstance(msg["content"], str)
                    for msg in row["messages"]), "model input must be text-only")
        require(view["answer"] == m["answer"] == row["messages"][1]["content"], "preview gold mismatch")
        target = as_dict(m["target"])
        require(view["prompt"] == row["messages"][0]["content"]
                == prompt(as_dict(target["state"]), question(as_str(m["operation"]),
                    cast("QueryDict", {key: value
                                       for key, value in as_dict(target["query"]).items()
                                       if key != "operation"}))), "prompt/target/preview mismatch")
        require(set(view) == {"row_id", "state_id", "family", "operation", "question",
                              "answer", "prompt", "source"}, "unexpected shared UI row schema")
        require(view["state_id"] in boards and 1 <= view["source"]["line"] <= MAX_SOURCE_ROWS,
                "preview/source join failed")
    checks.update({"validated_review_rows": len(rows), "validated_preview_rows": len(preview_rows),
                   "duplicate_state_query_pairs": 0, "duplicate_prompts": 0,
                   "family_count_assertions": 5, "operation_count_assertions": 20})
    states = Counter(c.donor.state_hash for c in selected)
    donor_uses = Counter(c.donor.row["row_id"] for c in selected)
    summary: JsonLikeDict = {
        "row_count": len(rows), "counts_by_family": counts_family,
        "counts_by_operation": counts_operation, "unique_state_count": len(states),
        "unique_donor_row_count": len(donor_uses), "reused_donor_presentations": len(rows) - len(donor_uses),
        "max_presentations_per_state": max(states.values()),
        "unique_map_count": len({c.donor.provenance["board_map_sha256"] for c in selected}),
        "unique_trajectory_count": len({as_str(as_dict(c.donor.provenance["source"])["trajectory_id"])
                                        for c in selected}),
        "counts_by_density": dict(Counter(r["source"]["density"] for r in preview_rows)),
        "counts_by_source_kind": dict(Counter(r["source"]["source_kind"] for r in preview_rows)),
        "seed": seed, "review_only": True, "admitted_for_training": False,
        "source_prefix_physical_lines": source_audit["physical_lines_read"],
    }
    require(len(states) == selection["maximum_distinct_states_for_required_cells"],
            "selection failed to maximize distinct source states")
    observations: JsonLikeDict = {
        "prefix": richness(donors, atlas),
        "selected": richness([c.donor for c in selected], atlas),
        "selected_cells": {op: dict(Counter(c.stratum for c in selected if c.operation == op))
                           for op in OPERATION_FAMILY},
        "unsupported_required_cells": [],
        "observed_but_unselected_cells": [
            s for s in (as_dict(e) for e in as_list(selection["support_by_cell"]))
            if not s["selected"]],
        "scope_limits": [
            "Only the first 400 physical source rows were examined; richness is not a full-corpus claim.",
            "Source contracts certify material state, not an independently reconstructed game history.",
            "Road-removal examples query the removed edge's endpoints, contrasting bridges and actual alternate routes.",
            "Production ignores bank shortages; upgrades explicitly ignore cost and piece supply.",
            "Full settlement-rule conjunction/legal placement and Longest Road optimization remain transfer-only.",
            "Strategic policy is outside this deterministic board-fluency review and belongs to RL.",
            "Source states/maps/trajectories are correlated; row count is not an independent-sample claim.",
        ],
    }
    preview: JsonLikeDict = {
        "schema": SCHEMA, "title": "Symbolic board fluency — review v1",
        "metadata": summary, "rows": cast("list[JsonLikeDict]", preview_rows),
        "boards": boards,
    }
    payloads: dict[str, bytes] = {
        "review.jsonl": ("".join(compact(row) + "\n" for row in rows)).encode(),
        "preview.json": (json.dumps(preview, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    }
    build_metadata: JsonLikeDict = {
        "schema": SCHEMA, "title": preview["title"], **summary,
        "source_audit": source_audit, "selection": selection, "observed_richness": observations,
        "validation": {"status": "passed", **dict(checks)},
        "build_code": {repository_relative(path): file_sha256(path) for path in BUILD_DEPENDENCIES},
        "artifacts": {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                      for name, payload in payloads.items()},
        "source_verification_scope": {
            "all_prefix_row_training_declarations": True,
            "selected_original_contract_sha256": True,
            "selected_contract_equals_original_target_state": True,
            "selected_original_provenance_preserved": True,
            "selected_map_and_visible_fact_hashes": True,
            "full_source_file_sha256": False,
            "original_label_files_rehashed": False,
            "full_source_loader_invoked": False,
            "model_input_contains_images_or_render_state": False,
            "successor_states_persisted_as_source": False,
        },
    }
    payloads["metadata.json"] = (
        json.dumps(build_metadata, indent=2, sort_keys=True) + "\n").encode()
    return payloads


def publish(payloads: dict[str, bytes], *, rewrite: bool, verify_only: bool) -> None:
    require(set(payloads) == OWN_FILES, "unexpected output files")
    require(OUTPUT.parent.is_dir() and not OUTPUT.is_symlink(), "invalid output parent/directory")
    if verify_only:
        require(OUTPUT.is_dir(), "no existing batch to verify")
    elif OUTPUT.exists():
        require(rewrite, "output directory already exists; use --verify-only or explicit --rewrite-own-artifacts")
        require(OUTPUT.is_dir(), "output path is not a directory")
        require({p.name for p in OUTPUT.iterdir()} <= OWN_FILES, "refusing to overwrite unowned artifacts")
    else:
        OUTPUT.mkdir()  # Fail on races; no silent replacement of another build.
    for name, payload in payloads.items():
        path = OUTPUT / name
        require(not path.is_symlink(), f"refusing artifact symlink: {path}")
        expected_hash = hashlib.sha256(payload).hexdigest()
        if not verify_only:
            # Explicit rewrite touches only these three owned generated files.
            with path.open("wb" if rewrite else "xb") as handle:
                handle.write(payload)
        require(file_sha256(path) == expected_hash, f"artifact rebuild/hash mismatch: {name}")
    metadata = as_dict(json.loads((OUTPUT / "metadata.json").read_text()))
    preview = as_dict(json.loads((OUTPUT / "preview.json").read_text()))
    with (OUTPUT / "review.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    require(len(rows) == len(as_list(preview["rows"])) == metadata["row_count"] == 200,
            "written artifacts failed row-count round trip")
    require(len(as_dict(preview["boards"])) == metadata["unique_state_count"],
            "written preview board count")
    print(json.dumps({
        "status": "verified_byte_identical_rebuild" if verify_only else "built_and_validated",
        "artifacts": {name: repository_relative(OUTPUT / name) for name in sorted(OWN_FILES)},
        "row_count": metadata["row_count"], "counts_by_family": metadata["counts_by_family"],
        "counts_by_operation": metadata["counts_by_operation"],
        "unique_state_count": metadata["unique_state_count"],
        "unique_donor_row_count": metadata["unique_donor_row_count"],
        "unique_map_count": metadata["unique_map_count"],
        "unique_trajectory_count": metadata["unique_trajectory_count"],
        "reused_donor_presentations": metadata["reused_donor_presentations"],
        "counts_by_density": metadata["counts_by_density"],
        "counts_by_source_kind": metadata["counts_by_source_kind"],
        "source_prefix_physical_lines": metadata["source_prefix_physical_lines"],
        "full_corpus_hash_verified": as_dict(
            metadata["source_audit"])["full_corpus_hash_verified"],
        "validation": metadata["validation"],
    }, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-prefix", action="store_true",
                        help="Print bounded source richness without writing artifacts")
    parser.add_argument("--seed", type=int, default=SEED)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true",
                      help="Rebuild in memory and compare all three artifact hashes without writes")
    mode.add_argument("--rewrite-own-artifacts", action="store_true",
                      help="Explicitly rewrite only this builder's three artifacts (also repairs partial writes)")
    args = parser.parse_args()
    if not args.inspect_prefix and OUTPUT.exists() and not (args.verify_only or args.rewrite_own_artifacts):
        parser.error("output directory already exists; use --verify-only or explicit --rewrite-own-artifacts")
    donors, audit = load_prefix()
    if args.inspect_prefix:
        print(json.dumps({"source": audit, "richness": richness(donors, atlas_geometry()),
                          "example_original_provenance": donors[0].provenance},
                         indent=2, sort_keys=True))
        return 0
    payloads = build(donors, audit, args.seed)
    publish(payloads, rewrite=args.rewrite_own_artifacts, verify_only=args.verify_only)
    return 0
