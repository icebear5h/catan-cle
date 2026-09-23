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
    COST_KEY,
    COST_SHA256,
    DATA_DIR,
    LIMITS,
    LOCAL_PARENT,
    LOCAL_PARENT_SHA256,
    LOCAL_ROOT,
    PARENT,
    PARENT_ROOT,
    PARENT_RUN,
    PARENT_SHA256,
    POLICY,
    PRIOR_USD,
)
from ._planning import (
    budget_plan,
    configuration,
    normalized,
    read_parent,
    source_hashes,
    verify_hashes,
)
from ._types import as_number, at


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

    prefix, consumed = contract(rows[:1024]), contract(rows[1024:2048])
    suffix = b"".join(lines[1024:])
    shared_states = ({as_str(state) for state in as_list(prefix["state_sha256"])}
                     & {as_str(at(row, "metadata", "state_sha256")) for row in rows[1024:]})
    if (len(set(as_list(at(inputs, "panels", "train", "ids")))) != 3200
            or set(as_list(prefix["ids"])) & {row["id"] for row in rows[1024:]} or shared_states):
        raise ValueError("suffix repeats parent-prefix example IDs or canonical states")
    return suffix, {"sha256": hashlib.sha256(suffix).hexdigest(), "bytes": len(suffix),
                    "rows": 2176, "original_start_row_zero_based": 1024,
                    "source_sha256": at(inputs, "files", "train.jsonl", "sha256"),
                    "parent_prefix": prefix, "consumed": consumed,
                    "parent_prefix_id_overlap": [],
                    "shared_parent_prefix_state_sha256": sorted(shared_states),
                    "unique_examples_basis": "row IDs; entire suffix also canonical-state-disjoint from parent prefix"}


def retained_baselines(parent: dict[str, JsonDict], root: Path, data: Path, review: Path) -> JsonDict:
    """Strictly rescore the complete r06 post panels against unchanged gold/meta/IDs."""
    inputs = as_dict(parent["launch"]["inputs"])
    baselines = as_dict(at(parent["posteval"], "result", "panels"))
    if set(baselines) != {"review", "validation_eval"}:
        raise ValueError("both complete r06 post panels are required")
    for panel, (count, correct) in {"review": (200, 58), "validation_eval": (190, 87)}.items():
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
    parent: dict[str, JsonDict], root: Path, data: Path, review: Path,
) -> tuple[bytes, JsonLikeDict, JsonDict]:
    inputs = as_dict(parent["launch"]["inputs"])
    if original.inspect_data(data, review) != inputs:
        raise ValueError("original data/inventory/review manifest changed")
    if root == Path(PARENT_ROOT):
        if sha256_file(data / "launch.json") != PARENT_SHA256["launch.json"]:
            raise ValueError("old uploaded data launch receipt changed")
    teacher = rows_at(root / "prepare/teacher120.jsonl")
    validation = {row["id"]: row for row in rows_at(data / "validation_eval.jsonl")}
    if teacher != [validation[row_id] for row_id in as_list(inputs["teacher_ids"])]:
        raise ValueError("teacher120 is not the exact original fixed panel")
    suffix, identity = suffix_identity(data / "train.jsonl", inputs)
    return suffix, identity, retained_baselines(parent, root, data, review)


def build_plan(run_name: str, budget_usd: float = 15) -> JsonDict:
    parent = read_parent(LOCAL_PARENT)
    config = configuration(run_name, as_dict(parent["launch"]["config"]))
    if (LOCAL_ROOT / run_name).exists():
        raise FileExistsError("local run name already used")
    verify_hashes(LOCAL_PARENT, LOCAL_PARENT_SHA256)
    if shared.read_json(LOCAL_PARENT / "analysis.json")["checks_passed"] is not True:
        raise ValueError("parent offline analysis was not successful")
    _, suffix, baselines = inspect_original(parent, LOCAL_PARENT, original.DATASET, original.REVIEW)
    plan = {"schema": "catan_board_fluency_extension_launch/v1", "run_name": run_name,
            "created_at": shared.now(), "reservation_id": uuid.uuid4().hex,
            "root": str(Path(config.output_dir).parent), "data_dir": DATA_DIR,
            "parent": PARENT, "parent_root": PARENT_ROOT, "base": BASE,
            "model_revision": shared.MODEL_REVISION, "profile": "icebear5h",
            "hf_secret_name": "huggingface-secret-2", "config": asdict(config),
            "inputs": parent["launch"]["inputs"], "parent_source_sha256": parent["launch"]["source_sha256"],
            "source_sha256": source_hashes(), "parent_receipts_sha256": PARENT_SHA256,
            "local_parent_receipts_sha256": LOCAL_PARENT_SHA256, "suffix": suffix, "saved_baselines": baselines,
             "budget_claim": {"dictionary": BUDGET_CLAIMS, "key": COST_SHA256},
             "cost_lineage": {"sha256": COST_SHA256, "total_key": COST_KEY,
                             "receipt_utf8": (LOCAL_PARENT / "cost_estimate.json").read_text()},
            "budget": budget_plan(budget_usd), "limits": LIMITS, "policy": POLICY}
    return as_dict(normalized(plan))


def verify_plan(plan: JsonDict) -> None:
    """Repeat offline admission locally; the same guards use old mounts on the server."""
    if normalized(plan) != plan:
        raise ValueError("plan must round-trip through strict JSON unchanged")
    remote = os.environ.get("MODAL_IS_REMOTE") == "1"
    root = Path(PARENT_ROOT) if remote else LOCAL_PARENT
    parent = read_parent(root)
    config = configuration(as_str(plan["run_name"]), as_dict(parent["launch"]["config"]))
    expected = {"schema": "catan_board_fluency_extension_launch/v1", "root": str(Path(config.output_dir).parent),
                "data_dir": DATA_DIR, "parent": PARENT, "parent_root": PARENT_ROOT, "base": BASE,
                "model_revision": shared.MODEL_REVISION, "profile": "icebear5h",
                "hf_secret_name": "huggingface-secret-2", "config": asdict(config),
                "inputs": parent["launch"]["inputs"], "parent_source_sha256": parent["launch"]["source_sha256"],
                "source_sha256": source_hashes(), "parent_receipts_sha256": PARENT_SHA256,
                 "local_parent_receipts_sha256": LOCAL_PARENT_SHA256, "limits": LIMITS, "policy": POLICY,
                 "budget_claim": {"dictionary": BUDGET_CLAIMS, "key": COST_SHA256},
                "budget": budget_plan(as_number(at(plan, "budget", "approved_usd")))}
    if any(plan[key] != value for key, value in expected.items()):
        raise ValueError("plan configuration/identity/source/budget/resource caps differ")
    reservation_id = as_str(plan["reservation_id"])
    if len(reservation_id) != 32 or uuid.UUID(hex=reservation_id).hex != reservation_id:
        raise ValueError("invalid immutable reservation identity")
    cost = as_dict(plan["cost_lineage"])
    receipt_utf8 = as_str(cost["receipt_utf8"])
    if (cost["sha256"] != COST_SHA256 or cost["total_key"] != COST_KEY
            or hashlib.sha256(receipt_utf8.encode()).hexdigest() != COST_SHA256):
        raise ValueError("parent cost lineage differs")
    receipt = json.loads(receipt_utf8, parse_float=Decimal)
    if (receipt["run_name"] != PARENT_RUN or receipt["totals"][COST_KEY] != PRIOR_USD
            or receipt["provenance"]["launch_sha256"] != shared.digest(parent["launch"])):
        raise ValueError("parent cost carry-forward is not the pinned allowance-inclusive total")
    for key in ("h200_per_second", "cpu_core_per_second", "memory_gib_per_second", "gpu_stage_per_second"):
        planned = as_number(at(plan, "budget", key))
        if (not math.isclose(float(receipt["rates"][key + "_usd"]), planned, rel_tol=0, abs_tol=1e-15)
                or not math.isclose(as_number(at(parent["launch"], "budget", key)), planned,
                                    rel_tol=0, abs_tol=1e-15)):
            raise ValueError("extension rates differ from the parent cost lineage")
    if not remote:
        verify_hashes(root, LOCAL_PARENT_SHA256)
    data, review = (Path(DATA_DIR), Path(DATA_DIR) / "review.jsonl") if remote else (original.DATASET, original.REVIEW)
    _, suffix, baselines = inspect_original(parent, root, data, review)
    if plan["suffix"] != suffix or plan["saved_baselines"] != baselines:
        raise ValueError("suffix identity or complete retained baseline differs")
    original.require_dependencies()
