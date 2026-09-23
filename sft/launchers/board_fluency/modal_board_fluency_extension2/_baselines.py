from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_list, as_str, loads_json
from sft.launchers.board_fluency import modal_board_fluency_extension as ext1
from sft.launchers.board_fluency import modal_board_fluency_sft as original
from sft.launchers.board_fluency.modal_board_fluency_extension._types import as_number, at
from sft.launchers.board_fluency.modal_board_fluency_sft import (
    BASE,
    check_manifest,
    evaluator,
    rows_at,
    shared,
)
from sft.scripts.train.train_trl_catan_vision import sha256_file

from ._config import (
    BUDGET_CLAIMS,
    CLAIM_KEY,
    COST_R06_KEY,
    COST_R06_SHA256,
    DATA_DIR,
    LIMITS,
    LOCAL_PARENT,
    LOCAL_PARENT_SHA256,
    LOCAL_R06,
    LOCAL_ROOT,
    PARENT,
    PARENT_ROOT,
    PARENT_RUN,
    PARENT_SHA256,
    POLICY,
    PRIOR_R06_USD,
    PRIOR_USD,
    R01_ADDENDUM_SHA256,
    R01_RECORDED_USD,
    R01_SUBTOTAL_KEY,
    R01_TOTAL_KEY,
    R06_ROOT,
    R06_RUN,
    SCHEMA,
    TEACHER_SHA256,
)
from ._planning import budget_plan, configuration, read_parent, source_hashes


def suffix_identity(path: Path, inputs: JsonDict) -> tuple[bytes, JsonLikeDict]:
    """Slice raw lines, never serialize or regenerate the original examples."""
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    if (len(lines) != 3200 or any(not line.strip() for line in lines)
            or hashlib.sha256(payload).hexdigest() != at(inputs, "files", "train.jsonl", "sha256")):
        raise ValueError("original raw training lines changed")
    rows = [as_dict(loads_json(line)) for line in lines]
    if [row["id"] for row in rows] != at(inputs, "panels", "train", "ids"):
        raise ValueError("original ordered training IDs changed")
    for index, row in enumerate(rows):
        metadata = as_dict(row["metadata"])
        canonical = hashlib.sha256(json.dumps(at(metadata, "target", "state"), sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
        if metadata["state_sha256"] != canonical or metadata["row_position"] != index:
            raise ValueError("canonical state hash or original row position differs")

    def contract(items: list[JsonDict]) -> JsonDict:
        return {**original.row_contract(items),
                "ordered_state_sha256": [at(row, "metadata", "state_sha256") for row in items]}

    consumed_ids = {row["id"] for row in rows[:2048]}
    consumed_states = {at(row, "metadata", "state_sha256") for row in rows[:2048]}
    if len(consumed_ids) != 2048 or len(as_list(at(inputs, "panels", "train", "ids"))) != 3200:
        raise ValueError("already-consumed rows 1-2048 identity differs")
    fresh_rows, repeat_rows = rows[2048:3200], rows[0:896]
    if len(fresh_rows) != 1152 or len(repeat_rows) != 896:
        raise ValueError("fresh/repeat partition differs")
    fresh, repeat = contract(fresh_rows), contract(repeat_rows)
    if set(as_list(fresh["ids"])) & consumed_ids:
        raise ValueError("fresh part repeats already-consumed example IDs 1-2048")
    if set(as_list(fresh["state_sha256"])) & consumed_states:
        raise ValueError("fresh part repeats already-consumed canonical states 1-2048")
    fresh_bytes = b"".join(lines[2048:3200])
    repeat_bytes = b"".join(lines[0:896])
    suffix = fresh_bytes + repeat_bytes
    suffix_lines = suffix.splitlines(keepends=True)
    if len(suffix_lines) != 2048 or suffix_lines[:1152] != lines[2048:3200]:
        raise ValueError("fresh part bytes differ from original rows 2049-3200")
    if suffix_lines[1152:] != lines[0:896]:
        raise ValueError("repeat part is not byte-identical to original rows 1-896")
    suffix_rows = [as_dict(loads_json(line)) for line in suffix_lines]
    if ([row["id"] for row in suffix_rows[:1152]] != [row["id"] for row in fresh_rows]
            or [row["id"] for row in suffix_rows[1152:]] != [row["id"] for row in repeat_rows]):
        raise ValueError("suffix fresh/repeat IDs differ from original rows")
    consumed = contract(suffix_rows)
    return suffix, {"sha256": hashlib.sha256(suffix).hexdigest(), "bytes": len(suffix),
                    "rows": 2048, "fresh_rows": 1152, "repeat_rows": 896,
                    "fresh_start_row_zero_based": 2048, "repeat_start_row_zero_based": 0,
                    "source_sha256": at(inputs, "files", "train.jsonl", "sha256"),
                    "fresh": fresh, "repeat": repeat, "consumed": consumed,
                    "already_consumed_rows_one_based_inclusive": [1, 2048],
                    "fresh_rows_one_based_inclusive": [2049, 3200],
                    "repeat_rows_one_based_inclusive": [1, 896],
                    "cumulative_presentations": 4096, "cumulative_unique_examples": 3200,
                    "cumulative_corpus_epoch": 1.0,
                    "unique_examples_basis": "row IDs; fresh part ID- and canonical-state-disjoint from rows 1-2048; repeat part byte- and ID-identical to rows 1-896"}


def retained_baselines(parent: dict[str, JsonDict], root: Path, data: Path, review: Path) -> JsonDict:
    """Strictly rescore the complete r01 post panels against unchanged gold/meta/IDs."""
    inputs = as_dict(parent["launch"]["inputs"])
    baselines = as_dict(at(parent["posteval"], "result", "panels"))
    if set(baselines) != {"review", "validation_eval"}:
        raise ValueError("both complete r01 post panels are required")
    for panel, (count, correct) in {"review": (200, 57), "validation_eval": (190, 96)}.items():
        saved = as_dict(baselines[panel])
        output = root / "posteval" / panel
        check_manifest(output, as_dict(saved["files"]))
        records = rows_at(output / "records.jsonl")
        gold = rows_at(review if panel == "review" else data / "validation_eval.jsonl")
        by_id = {row["id"]: row for row in gold}
        if (saved["rows"] != count or saved["correct"] != correct
                or saved["output_dir"] != PARENT_ROOT + f"/posteval/{panel}"
                or len(records) != count
                or Counter(row.get("id") for row in records) != Counter(as_list(at(inputs, "panels", panel, "ids")))):
            raise ValueError(f"{panel}: incomplete/duplicate/wrong retained baseline IDs")
        for record in records:
            row = by_id[record["id"]]
            metadata = {**evaluator.evaluation_metadata(row, image_variant="original"), "input_mode": "text"}
            if (record["expected"] != evaluator.expected_text(row) or record["metadata"] != metadata
                    or not isinstance(record["response"], str) or record.get("candidate_score") is not None
                    or record["score"] != evaluator.score_response(
                        as_str(record["expected"]), record["response"], metadata=metadata)
                    or as_dict(record["score"]).get("scoring") != metadata["schema"]
                    or type(as_dict(record["score"]).get("correct")) is not bool):
                raise ValueError(f"{panel}: retained prediction/gold/metadata/strict score differs")
        summary = shared.read_json(output / "summary.json")
        conditions = {"adapter_dir": PARENT, "model_id": BASE, "model_revision": shared.MODEL_REVISION,
                      "eval_jsonl": DATA_DIR + f"/{panel}.jsonl", "input_mode": "text", "bits": 16,
                      "max_sequence_length": 4096, "batch_size": 16, "max_new_tokens": 512,
                      "long_max_new_tokens": 512, "candidate_scoring": False,
                      "reasoning_enabled": False, "image_variant": "original", "truncation": False}
        recomputed = evaluator.summarize(records)
        conditions.update({key: value for key, value in recomputed.items() if key != "generated_at"})
        if (any(summary.get(key) != value for key, value in conditions.items())
                or recomputed["correct"] != correct or saved["exact_accuracy"] != correct / count
                or at(summary, "precision", "preserve_visual_fp32") is not True):
            raise ValueError(f"{panel}: retained inference conditions/summary disagree")
    return baselines


def inspect_original(
    parent: dict[str, JsonDict], root: Path, data: Path, review: Path, teacher: Path,
) -> tuple[bytes, JsonLikeDict, JsonDict]:
    inputs = as_dict(parent["launch"]["inputs"])
    if original.inspect_data(data, review) != inputs:
        raise ValueError("original data/inventory/review manifest changed")
    if sha256_file(teacher) != TEACHER_SHA256:
        raise ValueError("teacher120 hash differs")
    teacher_rows = rows_at(teacher)
    validation = {row["id"]: row for row in rows_at(data / "validation_eval.jsonl")}
    if teacher_rows != [validation[row_id] for row_id in as_list(inputs["teacher_ids"])]:
        raise ValueError("teacher120 is not the exact original fixed panel")
    suffix, identity = suffix_identity(data / "train.jsonl", inputs)
    return suffix, identity, retained_baselines(parent, root, data, review)


def teacher_path_for(remote: bool) -> Path:
    return Path(R06_ROOT) / "prepare/teacher120.jsonl" if remote else LOCAL_R06 / "prepare/teacher120.jsonl"


def build_plan(run_name: str, budget_usd: float = 21) -> JsonDict:
    parent = read_parent(LOCAL_PARENT)
    config = configuration(run_name, as_dict(parent["launch"]["config"]))
    if (LOCAL_ROOT / run_name).exists():
        raise FileExistsError("local run name already used")
    ext1.verify_hashes(LOCAL_PARENT, LOCAL_PARENT_SHA256)
    if shared.read_json(LOCAL_PARENT / "analysis.json")["checks_passed"] is not True:
        raise ValueError("parent offline analysis was not successful")
    _, suffix, baselines = inspect_original(parent, LOCAL_PARENT, original.DATASET, original.REVIEW,
                                           LOCAL_R06 / "prepare/teacher120.jsonl")
    plan = {"schema": SCHEMA, "run_name": run_name,
            "created_at": shared.now(), "reservation_id": uuid.uuid4().hex,
            "root": str(Path(config.output_dir).parent), "data_dir": DATA_DIR,
            "parent": PARENT, "parent_root": PARENT_ROOT, "base": BASE,
            "model_revision": shared.MODEL_REVISION, "profile": "icebear5h",
            "hf_secret_name": "huggingface-secret-2", "config": asdict(config),
            "inputs": parent["launch"]["inputs"], "parent_source_sha256": parent["launch"]["source_sha256"],
            "source_sha256": source_hashes(), "parent_receipts_sha256": PARENT_SHA256,
            "local_parent_receipts_sha256": LOCAL_PARENT_SHA256, "suffix": suffix, "saved_baselines": baselines,
            "budget_claim": {"dictionary": BUDGET_CLAIMS, "key": CLAIM_KEY},
            "cost_lineage": {"r06_sha256": COST_R06_SHA256, "r06_total_key": COST_R06_KEY,
                             "r06_receipt_utf8": (LOCAL_R06 / "cost_estimate.json").read_text(),
                             "r01_sha256": R01_ADDENDUM_SHA256, "r01_total_key": R01_TOTAL_KEY,
                             "r01_subtotal_key": R01_SUBTOTAL_KEY,
                             "r01_receipt_utf8": (LOCAL_PARENT / "cost_addendum.json").read_text(),
                             "carry_forward_usd_decimal": str(PRIOR_USD)},
            "budget": budget_plan(budget_usd), "limits": LIMITS, "policy": POLICY}
    return as_dict(ext1.normalized(plan))


def verify_plan(plan: JsonDict) -> None:
    """Repeat offline admission locally; the same guards use old mounts on the server."""
    if ext1.normalized(plan) != plan:
        raise ValueError("plan must round-trip through strict JSON unchanged")
    remote = os.environ.get("MODAL_IS_REMOTE") == "1"
    root = Path(PARENT_ROOT) if remote else LOCAL_PARENT
    parent = read_parent(root)
    config = configuration(as_str(plan["run_name"]), as_dict(parent["launch"]["config"]))
    expected = {"schema": SCHEMA, "root": str(Path(config.output_dir).parent),
                "data_dir": DATA_DIR, "parent": PARENT, "parent_root": PARENT_ROOT, "base": BASE,
                "model_revision": shared.MODEL_REVISION, "profile": "icebear5h",
                "hf_secret_name": "huggingface-secret-2", "config": asdict(config),
                "inputs": parent["launch"]["inputs"], "parent_source_sha256": parent["launch"]["source_sha256"],
                "source_sha256": source_hashes(), "parent_receipts_sha256": PARENT_SHA256,
                "local_parent_receipts_sha256": LOCAL_PARENT_SHA256, "limits": LIMITS, "policy": POLICY,
                "budget_claim": {"dictionary": BUDGET_CLAIMS, "key": CLAIM_KEY},
                "budget": budget_plan(as_number(at(plan, "budget", "approved_usd")))}
    if any(plan[key] != value for key, value in expected.items()):
        raise ValueError("plan configuration/identity/source/budget/resource caps differ")
    if at(plan, "budget_claim", "key") == ext1.COST_SHA256:
        raise ValueError("r02 must use a NEW budget-lineage claim key, not r01's claim key")
    reservation_id = as_str(plan["reservation_id"])
    if len(reservation_id) != 32 or uuid.UUID(hex=reservation_id).hex != reservation_id:
        raise ValueError("invalid immutable reservation identity")
    cost = as_dict(plan["cost_lineage"])
    r06_utf8, r01_utf8 = as_str(cost["r06_receipt_utf8"]), as_str(cost["r01_receipt_utf8"])
    if (cost["r06_sha256"] != COST_R06_SHA256 or cost["r06_total_key"] != COST_R06_KEY
            or hashlib.sha256(r06_utf8.encode()).hexdigest() != COST_R06_SHA256
            or cost["r01_sha256"] != R01_ADDENDUM_SHA256 or cost["r01_total_key"] != R01_TOTAL_KEY
            or cost["r01_subtotal_key"] != R01_SUBTOTAL_KEY
            or hashlib.sha256(r01_utf8.encode()).hexdigest() != R01_ADDENDUM_SHA256
            or cost["carry_forward_usd_decimal"] != str(PRIOR_USD)):
        raise ValueError("parent cost lineage differs")
    r06 = json.loads(r06_utf8, parse_float=Decimal)
    r01 = json.loads(r01_utf8, parse_float=Decimal)
    if (r06["run_name"] != R06_RUN or r06["totals"][COST_R06_KEY] != PRIOR_R06_USD):
        raise ValueError("r06 cost carry-forward is not the pinned allowance-inclusive total")
    if (r01["run_name"] != PARENT_RUN or r01[R01_TOTAL_KEY] != PRIOR_USD
            or r01[R01_SUBTOTAL_KEY] != R01_RECORDED_USD
            or r01["prior_allowance_inclusive_usd"] != PRIOR_R06_USD):
        raise ValueError("r01 cost carry-forward is not the pinned recorded/total lineage")
    if PRIOR_R06_USD + R01_RECORDED_USD != PRIOR_USD:
        raise ValueError("carry-forward arithmetic differs")
    for key in ("h200_per_second", "cpu_core_per_second", "memory_gib_per_second", "gpu_stage_per_second"):
        planned = as_number(at(plan, "budget", key))
        if (not math.isclose(float(r06["rates"][key + "_usd"]), planned, rel_tol=0, abs_tol=1e-15)
                or not math.isclose(as_number(at(parent["launch"], "budget", key)), planned,
                                    rel_tol=0, abs_tol=1e-15)):
            raise ValueError("extension rates differ from the parent cost lineage")
    if not remote:
        ext1.verify_hashes(root, LOCAL_PARENT_SHA256)
    data, review = (Path(DATA_DIR), Path(DATA_DIR) / "review.jsonl") if remote else (original.DATASET, original.REVIEW)
    _, suffix, baselines = inspect_original(parent, root, data, review, teacher_path_for(remote))
    if plan["suffix"] != suffix or plan["saved_baselines"] != baselines:
        raise ValueError("suffix identity or complete retained baseline differs")
    original.require_dependencies()
