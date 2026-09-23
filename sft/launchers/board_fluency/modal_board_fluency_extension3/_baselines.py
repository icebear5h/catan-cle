from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from sft.json_types import JsonDict, as_dict, as_float, as_list, as_str, loads_json
from sft.launchers.board_fluency import modal_board_fluency_extension as ext1
from sft.launchers.board_fluency import modal_board_fluency_extension2 as ext2
from sft.launchers.board_fluency import modal_board_fluency_sft as original
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
    DATA_DIR,
    LIMITS,
    LOCAL_PARENT,
    LOCAL_PARENT_SHA256,
    LOCAL_R01,
    LOCAL_R06,
    LOCAL_ROOT,
    PARENT,
    PARENT_ROOT,
    PARENT_RUN,
    PARENT_SHA256,
    POLICY,
    PRIOR_R01_USD,
    PRIOR_USD,
    R01_ADDENDUM_SHA256,
    R01_RUN,
    R01_TOTAL_KEY,
    R02_ADDENDUM_SHA256,
    R02_PRIOR_KEY,
    R02_RECORDED_USD,
    R02_SUBTOTAL_KEY,
    R02_TOTAL_KEY,
    SCHEMA,
    TEACHER_SHA256,
)
from ._planning import budget_plan, configuration, read_parent, source_hashes
from ._types import counts, decimal_receipt, dig, json_dict, json_list, receipt_number


def suffix_identity(path: Path, inputs: JsonDict) -> tuple[bytes, JsonDict]:
    """Slice raw lines, never serialize or regenerate the original examples."""
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    if (len(lines) != 3200 or any(not line.strip() for line in lines)
            or hashlib.sha256(payload).hexdigest() != dig(inputs, "files", "train.jsonl", "sha256")):
        raise ValueError("original raw training lines changed")
    rows = [as_dict(loads_json(line)) for line in lines]
    train_ids = dig(inputs, "panels", "train", "ids")
    if [row["id"] for row in rows] != train_ids:
        raise ValueError("original ordered training IDs changed")
    for index, row in enumerate(rows):
        metadata = as_dict(row["metadata"])
        canonical = hashlib.sha256(json.dumps(dig(metadata, "target", "state"), sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
        if metadata["state_sha256"] != canonical or metadata["row_position"] != index:
            raise ValueError("canonical state hash or original row position differs")

    def contract(items: list[JsonDict]) -> JsonDict:
        return {**original.row_contract(items),
                "ordered_state_sha256": [dig(row, "metadata", "state_sha256") for row in items]}

    if len(set(as_list(train_ids))) != 3200:
        raise ValueError("original training ID set differs")
    epoch2_rows, epoch3_rows = rows[0:3200], rows[0:896]
    if len(epoch2_rows) != 3200 or len(epoch3_rows) != 896:
        raise ValueError("epoch2/epoch3 partition differs")
    epoch2, epoch3 = contract(epoch2_rows), contract(epoch3_rows)
    if set(as_list(epoch2["ids"])) != set(as_list(train_ids)):
        raise ValueError("epoch-2 part is not the full original corpus IDs")
    epoch2_bytes = b"".join(lines[0:3200])
    epoch3_bytes = b"".join(lines[0:896])
    if epoch2_bytes != payload:
        raise ValueError("epoch-2 part is not byte-identical to the full original corpus")
    suffix = epoch2_bytes + epoch3_bytes
    suffix_lines = suffix.splitlines(keepends=True)
    if len(suffix_lines) != 4096 or suffix_lines[:3200] != lines[0:3200]:
        raise ValueError("epoch-2 part bytes differ from the full original corpus")
    if suffix_lines[3200:] != lines[0:896]:
        raise ValueError("epoch-3 part is not byte-identical to original rows 1-896")
    suffix_rows = [as_dict(loads_json(line)) for line in suffix_lines]
    if ([row["id"] for row in suffix_rows[:3200]] != [row["id"] for row in epoch2_rows]
            or [row["id"] for row in suffix_rows[3200:]] != [row["id"] for row in epoch3_rows]):
        raise ValueError("suffix epoch2/epoch3 IDs differ from original rows")
    if len({row["id"] for row in suffix_rows}) != 3200:
        raise ValueError("cumulative unique examples differ from 3200")
    consumed_ids = [row["id"] for row in suffix_rows]
    if len(consumed_ids) != 4096 or len(set(consumed_ids)) != 3200:
        raise ValueError("suffix presentations/unique counts differ")
    consumed_operations = counts(epoch2["by_operation"]) + counts(epoch3["by_operation"])
    consumed_families = counts(epoch2["by_family"]) + counts(epoch3["by_family"])
    consumed: JsonDict = {
        "rows": 4096, "ids": consumed_ids, "unique_ids": json_list(sorted({as_str(i) for i in consumed_ids})),
        "unique_rows": 3200,
        "state_sha256": json_list(sorted({as_str(dig(row, "metadata", "state_sha256")) for row in suffix_rows})),
        "ordered_state_sha256": [dig(row, "metadata", "state_sha256") for row in suffix_rows],
        "by_operation": json_dict(consumed_operations), "by_family": json_dict(consumed_families)}
    return suffix, {"sha256": hashlib.sha256(suffix).hexdigest(), "bytes": len(suffix),
                    "rows": 4096, "epoch2_rows": 3200, "epoch3_rows": 896,
                    "epoch2_start_row_zero_based": 0, "epoch3_start_row_zero_based": 0,
                    "source_sha256": dig(inputs, "files", "train.jsonl", "sha256"),
                    "epoch2": epoch2, "epoch3": epoch3, "consumed": consumed,
                    "epoch2_rows_one_based_inclusive": [1, 3200],
                    "epoch3_rows_one_based_inclusive": [1, 896],
                    "cumulative_presentations": 8192, "cumulative_unique_examples": 3200,
                    "cumulative_corpus_epoch": 1.0,
                    "unique_examples_basis": "row IDs; epoch-2 part byte- and ID-identical to full original corpus (3200 rows), epoch-3 part byte- and ID-identical to rows 1-896; cumulative unique still 3200"}


def retained_baselines(parent: dict[str, JsonDict], root: Path, data: Path, review: Path) -> JsonDict:
    """Strictly rescore the complete r02 post panels against unchanged gold/meta/IDs."""
    inputs = as_dict(parent["launch"]["inputs"])
    baselines = as_dict(dig(parent["posteval"], "result", "panels"))
    if set(baselines) != {"review", "validation_eval"}:
        raise ValueError("both complete r02 post panels are required")
    for panel, (count, correct) in {"review": (200, 76), "validation_eval": (190, 116)}.items():
        saved = as_dict(baselines[panel])
        output = root / "posteval" / panel
        check_manifest(output, as_dict(saved["files"]))
        records = rows_at(output / "records.jsonl")
        gold = rows_at(review if panel == "review" else data / "validation_eval.jsonl")
        by_id = {row["id"]: row for row in gold}
        if (saved["rows"] != count or saved["correct"] != correct
                or saved["output_dir"] != PARENT_ROOT + f"/posteval/{panel}"
                or len(records) != count
                or Counter(row.get("id") for row in records) != Counter(as_list(dig(inputs, "panels", panel, "ids")))):
            raise ValueError(f"{panel}: incomplete/duplicate/wrong retained baseline IDs")
        for record in records:
            row = by_id[record["id"]]
            metadata: JsonDict = {**evaluator.evaluation_metadata(row, image_variant="original"), "input_mode": "text"}
            expected, response = evaluator.expected_text(row), record["response"]
            # record["expected"] == expected once the first clause passes, so scoring it is unchanged.
            if (record["expected"] != expected or record["metadata"] != metadata
                    or not isinstance(response, str) or record.get("candidate_score") is not None
                    or record["score"] != evaluator.score_response(expected, response, metadata=metadata)
                    or as_dict(record["score"]).get("scoring") != metadata["schema"]
                    or type(as_dict(record["score"]).get("correct")) is not bool):
                raise ValueError(f"{panel}: retained prediction/gold/metadata/strict score differs")
        summary = shared.read_json(output / "summary.json")
        conditions: JsonDict = {"adapter_dir": PARENT, "model_id": BASE, "model_revision": shared.MODEL_REVISION,
                      "eval_jsonl": DATA_DIR + f"/{panel}.jsonl", "input_mode": "text", "bits": 16,
                      "max_sequence_length": 4096, "batch_size": 16, "max_new_tokens": 512,
                      "long_max_new_tokens": 512, "candidate_scoring": False,
                      "reasoning_enabled": False, "image_variant": "original", "truncation": False}
        recomputed = evaluator.summarize(records)
        conditions.update({key: value for key, value in recomputed.items() if key != "generated_at"})
        if (any(summary.get(key) != value for key, value in conditions.items())
                or recomputed["correct"] != correct or saved["exact_accuracy"] != correct / count
                or dig(summary, "precision", "preserve_visual_fp32") is not True):
            raise ValueError(f"{panel}: retained inference conditions/summary disagree")
    return baselines


def inspect_original(parent: dict[str, JsonDict], root: Path, data: Path, review: Path,
                     teacher: Path) -> tuple[bytes, JsonDict, JsonDict]:
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


def build_plan(run_name: str, budget_usd: float = 31) -> JsonDict:
    parent = read_parent(LOCAL_PARENT)
    config = configuration(run_name, as_dict(parent["launch"]["config"]))
    if (LOCAL_ROOT / run_name).exists():
        raise FileExistsError("local run name already used")
    ext1.verify_hashes(LOCAL_PARENT, LOCAL_PARENT_SHA256)
    if shared.read_json(LOCAL_PARENT / "analysis.json")["checks_passed"] is not True:
        raise ValueError("parent offline analysis was not successful")
    _, suffix, baselines = inspect_original(parent, LOCAL_PARENT, original.DATASET, original.REVIEW,
                                           LOCAL_R06 / "prepare/teacher120.jsonl")
    plan: dict[str, object] = {"schema": SCHEMA, "run_name": run_name,
            "created_at": shared.now(), "reservation_id": uuid.uuid4().hex,
            "root": str(Path(config.output_dir).parent), "data_dir": DATA_DIR,
            "parent": PARENT, "parent_root": PARENT_ROOT, "base": BASE,
            "model_revision": shared.MODEL_REVISION, "profile": "icebear5h",
            "hf_secret_name": "huggingface-secret-2", "config": asdict(config),
            "inputs": parent["launch"]["inputs"], "parent_source_sha256": parent["launch"]["source_sha256"],
            "source_sha256": source_hashes(), "parent_receipts_sha256": PARENT_SHA256,
            "local_parent_receipts_sha256": LOCAL_PARENT_SHA256, "suffix": suffix, "saved_baselines": baselines,
            "budget_claim": {"dictionary": BUDGET_CLAIMS, "key": CLAIM_KEY},
            "cost_lineage": {"r01_sha256": R01_ADDENDUM_SHA256, "r01_total_key": R01_TOTAL_KEY,
                             "r01_receipt_utf8": (LOCAL_R01 / "cost_addendum.json").read_text(),
                             "r02_sha256": R02_ADDENDUM_SHA256, "r02_subtotal_key": R02_SUBTOTAL_KEY,
                             "r02_total_key": R02_TOTAL_KEY, "r02_prior_key": R02_PRIOR_KEY,
                             "r02_receipt_utf8": (LOCAL_PARENT / "cost_addendum.json").read_text(),
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
    expected: dict[str, object] = {"schema": SCHEMA, "root": str(Path(config.output_dir).parent),
                "data_dir": DATA_DIR, "parent": PARENT, "parent_root": PARENT_ROOT, "base": BASE,
                "model_revision": shared.MODEL_REVISION, "profile": "icebear5h",
                "hf_secret_name": "huggingface-secret-2", "config": asdict(config),
                "inputs": parent["launch"]["inputs"], "parent_source_sha256": parent["launch"]["source_sha256"],
                "source_sha256": source_hashes(), "parent_receipts_sha256": PARENT_SHA256,
                "local_parent_receipts_sha256": LOCAL_PARENT_SHA256, "limits": LIMITS, "policy": POLICY,
                "budget_claim": {"dictionary": BUDGET_CLAIMS, "key": CLAIM_KEY},
                "budget": budget_plan(as_float(dig(plan, "budget", "approved_usd")))}
    if any(plan[key] != value for key, value in expected.items()):
        raise ValueError("plan configuration/identity/source/budget/resource caps differ")
    if dig(plan, "budget_claim", "key") in (ext1.COST_SHA256, ext2.CLAIM_KEY):
        raise ValueError("r03 must use a NEW budget-lineage claim key, not r01/r02 claim keys")
    reservation_id = as_str(plan["reservation_id"])
    if len(reservation_id) != 32 or uuid.UUID(hex=reservation_id).hex != reservation_id:
        raise ValueError("invalid immutable reservation identity")
    cost = as_dict(plan["cost_lineage"])
    if (cost["r01_sha256"] != R01_ADDENDUM_SHA256 or cost["r01_total_key"] != R01_TOTAL_KEY
            or hashlib.sha256(as_str(cost["r01_receipt_utf8"]).encode()).hexdigest() != R01_ADDENDUM_SHA256
            or cost["r02_sha256"] != R02_ADDENDUM_SHA256 or cost["r02_subtotal_key"] != R02_SUBTOTAL_KEY
            or cost["r02_total_key"] != R02_TOTAL_KEY or cost["r02_prior_key"] != R02_PRIOR_KEY
            or hashlib.sha256(as_str(cost["r02_receipt_utf8"]).encode()).hexdigest() != R02_ADDENDUM_SHA256
            or cost["carry_forward_usd_decimal"] != str(PRIOR_USD)):
        raise ValueError("parent cost lineage differs")
    r01 = decimal_receipt(as_str(cost["r01_receipt_utf8"]))
    r02 = decimal_receipt(as_str(cost["r02_receipt_utf8"]))
    if (r01["run_name"] != R01_RUN or r01[R01_TOTAL_KEY] != PRIOR_R01_USD):
        raise ValueError("r01 cost carry-forward is not the pinned cumulative total")
    if (r02["run_name"] != PARENT_RUN or r02[R02_SUBTOTAL_KEY] != R02_RECORDED_USD
            or r02[R02_TOTAL_KEY] != PRIOR_USD or r02[R02_PRIOR_KEY] != PRIOR_R01_USD):
        raise ValueError("r02 cost carry-forward is not the pinned recorded/subtotal lineage")
    if PRIOR_R01_USD + R02_RECORDED_USD != PRIOR_USD:
        raise ValueError("carry-forward arithmetic differs")
    for key in ("h200_per_second", "cpu_core_per_second", "memory_gib_per_second", "gpu_stage_per_second"):
        if not math.isclose(as_float(dig(parent["launch"], "budget", key)), as_float(dig(plan, "budget", key)),
                            rel_tol=0, abs_tol=1e-15):
            raise ValueError("extension rates differ from the parent cost lineage")
    if not math.isclose(receipt_number(r02, "rates", "gpu_stage_per_second_usd"),
                        as_float(dig(plan, "budget", "gpu_stage_per_second")),
                        rel_tol=0, abs_tol=1e-15):
        raise ValueError("extension GPU rate differs from r02 recorded rates")
    if not remote:
        ext1.verify_hashes(root, LOCAL_PARENT_SHA256)
    data, review = (Path(DATA_DIR), Path(DATA_DIR) / "review.jsonl") if remote else (original.DATASET, original.REVIEW)
    _, suffix, baselines = inspect_original(parent, root, data, review, ext2.teacher_path_for(remote))
    if plan["suffix"] != suffix or plan["saved_baselines"] != baselines:
        raise ValueError("suffix identity or complete retained baseline differs")
    original.require_dependencies()
