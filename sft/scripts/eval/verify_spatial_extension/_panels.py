"""panels."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_list, as_str
from sft.launchers.spatial.modal_spatial_continuation import (
    NEW_PANEL_TASKS,
    OLD_PANELS,
    PANEL_BUDGETS,
    check_identity,
    dataset_identity,
    digest,
    evaluation_conditions,
    read_json,
)
from sft.launchers.spatial.modal_spatial_extension import (
    CHECKPOINT_STEPS,
)
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.eval import verify_spatial_extension as verify

from ._base import CHECKPOINT_FILES, STATE_PATH
from ._receipts import expected_checkpoint, full_history, local_path, require


def verify_training(run: Path, launch: JsonDict, result: JsonDict,
                    audit: JsonDict) -> list[JsonDict]:
    worker = as_dict(as_dict(as_dict(result["stages"])["training"])["result"])
    require(worker["status"] == "completed" and worker["config"] == launch["config"]
            and worker["source_sha256"] == launch["source_sha256"]
            and worker["checkpoint"] == audit["checkpoint"] == expected_checkpoint(launch)
            and worker["checkpoint_audit"] == audit, "training/checkpoint audit differs")
    require(audit["visual_dtypes"] == {"F32": 333}, "checkpoint must contain 333 FP32 tensors")
    hashes = as_dict(audit["files_sha256"])
    require(set(hashes) == set(CHECKPOINT_FILES)
            and all(isinstance(value, str) and len(value) == 64
                    and all(c in "0123456789abcdef" for c in value) for value in hashes.values()),
            "incomplete checkpoint file hashes")
    for key in ("tokens", "token_ids"):
        require(as_dict(audit["semantic_tokens"])[key]
                == as_dict(as_dict(launch["parent_audit"])["semantic_tokens"])[key],
                f"checkpoint atlas {key} differs from parent")
    raw_history = worker.get("teacher_forced_history")
    history = None if raw_history is None else [as_dict(row) for row in as_list(raw_history)]
    state_path = local_path(run, STATE_PATH)
    if state_path.exists():
        require(verify.sha256_file(state_path) == hashes["trainer_state.json"], "trainer_state hash differs")
        state = read_json(state_path)
        require(state["global_step"] == as_dict(launch["config"])["max_steps"],
                "trainer_state step differs")
        saved_history = [row for row in (as_dict(item) for item in as_list(state["log_history"]))
                         if "eval_loss" in row]
        require(history is None or history == saved_history, "embedded history differs from trainer_state")
        history = saved_history
    require(history is not None and full_history(history),
            "all eight teacher-forced evaluations required")
    history = cast("list[JsonDict]", history)
    require(all(isinstance(row.get("eval_loss"), (int, float))
                and not isinstance(row["eval_loss"], bool)
                and math.isfinite(cast("float", row["eval_loss"]))
                for row in history), "non-finite or missing teacher-forced loss")
    if "checkpoints" in worker:
        checkpoints = as_dict(worker["checkpoints"])
        require(set(checkpoints) == {str(step) for step in CHECKPOINT_STEPS},
                "training checkpoint schedule differs")
        output_dir = as_dict(launch["config"])["output_dir"]
        for step, entry in checkpoints.items():
            require(as_dict(entry)["checkpoint"]
                    == f"{output_dir}/checkpoints/checkpoint-{step}",
                    "training checkpoint path differs")
        require(as_dict(checkpoints["256"])["trainer_state_sha256"] == hashes["trainer_state.json"],
                "final training checkpoint hash differs")
    return history


def verify_panel(run: Path, launch: JsonDict, post: JsonDict, label: str) -> JsonLikeDict:
    panel = as_dict(as_dict(launch["panels"])[label])
    saved = as_dict(as_dict(post["panels"])[label])
    baseline = as_dict(as_dict(as_dict(launch["saved_baselines"])[label])["summary"])
    source = (OLD_PANELS[label] if label in OLD_PANELS
              else as_dict(as_dict(as_dict(launch["dataset_inputs"])["new_panels"])[
                  NEW_PANEL_TASKS[label]]))
    identity = dataset_identity(as_str(source["eval_jsonl"]), as_str(source["image_root"]))
    require(identity == panel["local_identity"], f"{label}: local input bytes/pixels/order differ")
    check_identity(identity, as_dict(panel["identity"]))
    require(saved["identity"] == panel["identity"], f"{label}: uploaded input identity differs")
    require(saved["conditions"] == evaluation_conditions(label)
            and saved["scorer_sha256"] == digest(launch["source_sha256"]),
            f"{label}: conditions/scorer differ")
    require(saved["records_path"] == f"/runs/catan-vision-sft/pipelines/{run.name}/post/{label}/records.jsonl",
            f"{label}: records path differs")
    path = local_path(run, f"post/{label}-records.jsonl")
    records_hash = verify.sha256_file(path)
    require(records_hash == saved["records_sha256"], f"{label}: records hash differs")
    records = [as_dict(row) for _, row in evaluator.iter_jsonl(path)]
    summary = read_json(local_path(run, f"post/{label}-summary.json"))
    require(summary == saved["summary"], f"{label}: summary receipt differs")
    require(baseline["eval_source_sha256"] == identity["sha256"]
            and baseline["eval_jsonl"] == panel["eval_jsonl"]
            and baseline["image_root"] == panel["image_root"] and bool(baseline["eval_set_id"]),
            f"{label}: pinned parent input metadata differs")
    audit = as_dict(post["checkpoint_audit"])
    evidence = as_dict(summary["adapter_evidence"])
    visual = as_dict(evidence["visual_state"])
    require(evidence["adapter_loaded"] is True and evidence["adapter_dir"] == post["checkpoint"]
            and visual["loaded"] is True and visual["tensors"] == visual["expected_tensors"] == 333
            and visual["path"] == f"{post['checkpoint']}/visual_model.safetensors"
            and visual["sha256"] == as_dict(audit["files_sha256"])["visual_model.safetensors"],
            f"{label}: checkpoint visual/hash identity differs")
    require(evidence["visual_precision"] == {
        "base_load_dtype": "torch.bfloat16", "loaded_dtypes": {"torch.float32": 333},
        "promoted_before_restore": True, "source_dtypes": {"F32": 333}},
        f"{label}: visual precision differs")
    require(as_dict(summary["precision"])["preserve_visual_fp32"] is True
            and summary["reasoning_enabled"] is summary["candidate_scoring"] is False,
            f"{label}: precision/generation flags differ")
    for key in ("tokens", "token_ids"):
        require(as_dict(evidence["semantic_tokens"])[key]
                == as_dict(audit["semantic_tokens"])[key],
                f"{label}: loaded atlas {key} differ")
    rows = [as_dict(row) for _, row in evaluator.iter_jsonl(Path(as_str(source["eval_jsonl"])))]
    # Generation stably partitions the frozen input into short, then long answers.
    ordered = sorted(rows, key=evaluator.is_long_answer)
    count, batch, budget = PANEL_BUDGETS[label]
    require(len(records) == len(rows) == count, f"{label}: incomplete record count")
    for index, (row, record) in enumerate(zip(ordered, records, strict=True), 1):
        require(record["id"] == (row.get("id") or row["row_id"]) and record["index"] == index,
                f"{label}: record order/ID differs at {index}")
        metadata = evaluator.evaluation_metadata(row, image_variant="original")
        metadata.update(eval_set_id=baseline["eval_set_id"], eval_source_sha256=identity["sha256"])
        require(record["metadata"] == metadata and record["expected"] == evaluator.expected_text(row),
                f"{label}: target/metadata differs at {index}")
        require(isinstance(record["response"], str) and record.get("candidate_score") is None,
                f"{label}: missing response or unexpected candidate scoring at {index}")
        score = evaluator.score_response(as_str(record["expected"]), as_str(record["response"]),
                                         metadata=metadata)
        require(score == record["score"], f"{label}: saved score differs at {index}")
        record["score"] = score
    computed = evaluator.summarize(records)
    expected: JsonDict = {**computed, "generated_at": summary["generated_at"],
                "model_id": as_dict(launch["config"])["model_id"],
                "adapter_dir": post["checkpoint"],
                "eval_jsonl": panel["eval_jsonl"], "image_root": panel["image_root"],
                "token_inventory": as_dict(launch["config"])["token_inventory"],
                "bits": 16, "batch_size": batch, "max_new_tokens": budget,
                "long_max_new_tokens": budget, "adapter_evidence": evidence,
                "reasoning_enabled": False, "image_variant": "original", "occlusion_margin": 0.03,
                "candidate_scoring": False, "rows_skipped_without_spatial_target": 0,
                "eval_set_id": baseline["eval_set_id"], "eval_source_sha256": identity["sha256"],
                "precision": {"preserve_visual_fp32": True,
                              "generation_autocast": {"device_type": "cuda", "dtype": "torch.bfloat16"}}}
    differences = sorted(key for key in summary.keys() | expected.keys()
                         if key not in summary or key not in expected or summary[key] != expected[key])
    require(not differences, f"{label}: summary aggregates/settings differ: {', '.join(differences)}")
    return {"rows": count, "correct": computed["correct"], "exact_accuracy": computed["exact_accuracy"],
            "records_sha256": records_hash, "input_identity": identity}
