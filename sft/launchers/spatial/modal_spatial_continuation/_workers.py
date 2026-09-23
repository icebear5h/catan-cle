"""Bounded remote training, evaluation and panel comparison."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from sft.json_types import JsonDict, as_dict, opt_str
from sft.launchers._json import at, at_dict, at_str
from sft.launchers.full_board.modal_full_board_pilot import (
    SNAPSHOT,
    app,
)
from sft.launchers.spatial import modal_spatial_continuation as launcher
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train.train_trl_catan_vision import (
    sha256_file,
    write_json_atomic,
)

from ._base import CONTINUATION_GPU_OPTIONS, NEW_LABELS, PANEL_BUDGETS
from ._plan import check_identity, dataset_identity, digest, now
from ._runtime import evaluation_conditions, panel_args, reload_volumes, verify_runtime


@app.function(**CONTINUATION_GPU_OPTIONS)
def train_bounded(plan: JsonDict, audit: JsonDict) -> JsonDict:
    reload_volumes()
    verify_runtime(plan)
    config = at_dict(plan, "config")
    train_jsonl = at_str(config, "train_jsonl")
    check_identity(dataset_identity(train_jsonl, opt_str(config["image_root"])), audit["train_identity"])
    if sha256_file(Path(train_jsonl)) != at(audit, "train_identity", "sha256"):
        raise ValueError("uploaded train JSONL bytes changed after CPU preflight")
    if dataset_identity(at_str(config, "eval_jsonl"), opt_str(config["eval_image_root"])) != audit["teacher_eval_identity"]:
        raise ValueError("teacher-forced evaluation changed after CPU preflight")
    if launcher.checkpoint_audit(Path(launcher.PARENT_CHECKPOINT), at_dict(plan, "parent_config"), parent=True) != audit["checkpoint"]:
        raise ValueError("parent checkpoint changed after CPU preflight")
    # Run the existing worker IN this bounded container, not as an unobserved child.
    try:
        return launcher.pilot_train.local(config)
    finally:
        launcher.sft_runs.commit()


@app.function(**CONTINUATION_GPU_OPTIONS)
def evaluate_bounded(plan: JsonDict, stage: str, audit: JsonDict) -> JsonDict:
    if stage not in ("pre", "post"):
        raise ValueError("only one pre and one post evaluation are permitted")
    reload_volumes()
    verify_runtime(plan)
    checkpoint = launcher.PARENT_CHECKPOINT if stage == "pre" else str(Path(at_str(plan, "config", "output_dir")) / "checkpoints/checkpoint-128")
    checkpoint_report = launcher.checkpoint_audit(Path(checkpoint), at_dict(plan, "parent_config" if stage == "pre" else "config"), parent=stage == "pre")
    if stage == "pre" and checkpoint_report != audit["checkpoint"]:
        raise ValueError("parent changed after preflight")
    output = launcher.RUN_ROOT / "pipelines" / at_str(plan, "run_name") / stage
    output.mkdir(parents=True, exist_ok=False)
    panels: JsonDict = {}
    result: JsonDict = {"status": "running", "checkpoint": checkpoint, "checkpoint_audit": checkpoint_report,
                        "started_at": now(), "panels": panels, "source_sha256": plan["source_sha256"]}
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        torch.set_num_threads(16)
        model, processor, evidence = evaluator.load_model(
            model_id=SNAPSHOT, adapter_dir=checkpoint, bits=16, disable_flash_attn2=True,
            token_inventory=at_str(plan, "config", "token_inventory"), preserve_visual_fp32=True)
        for label in (NEW_LABELS if stage == "pre" else PANEL_BUDGETS):
            panel = at_dict(plan, "panels", label)
            eval_jsonl = at_str(panel, "eval_jsonl")
            identity = dataset_identity(eval_jsonl, opt_str(panel["image_root"]))
            check_identity(identity, at(audit, "panels", label))
            if identity["sha256"] != at(audit, "panels", label, "sha256"):
                raise ValueError("uploaded eval bytes changed after preflight")
            summary = evaluator.run_eval_job(model=model, processor=processor, adapter_evidence=evidence,
                args=panel_args(plan, checkpoint, label), eval_jsonl=eval_jsonl,
                image_variant="original", output_dir=output / label)
            if summary["rows"] != PANEL_BUDGETS[label][0] or summary["attempted"] != summary["rows"]:
                raise ValueError("evaluation returned incomplete panel")
            records_path = output / label / "records.jsonl"
            panels[label] = {"summary": summary, "identity": identity,
                                      "records_path": str(records_path), "records_sha256": sha256_file(records_path),
                                      "scorer_sha256": digest(plan["source_sha256"]),
                                      "conditions": evaluation_conditions(label)}
            write_json_atomic(output / "result.json", result)
            launcher.sft_runs.commit()
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["ended_at"] = now()
        write_json_atomic(output / "result.json", result)
        launcher.sft_runs.commit()
    return result


def compare_panels(before: JsonDict, after: JsonDict) -> JsonDict:
    if set(before) != set(PANEL_BUDGETS) or set(after) != set(before):
        raise ValueError("comparison requires all six matched panels")
    comparison: JsonDict = {}
    for label in PANEL_BUDGETS:
        pre, post = at_dict(before, label), at_dict(after, label)
        check_identity(post["identity"], pre["identity"])
        if (not pre.get("scorer_sha256")
                or pre["scorer_sha256"] != post.get("scorer_sha256")
                or pre.get("conditions") != evaluation_conditions(label)
                or post.get("conditions") != evaluation_conditions(label)):
            raise ValueError("pre/post scorer version or generation conditions differ")
        a, b = at_dict(pre, "summary"), at_dict(post, "summary")
        if a["rows"] != b["rows"] or b["rows"] != PANEL_BUDGETS[label][0]:
            raise ValueError("pre/post panel counts differ")
        comparison[label] = {"rows": b["rows"], "correct_before": a["correct"], "correct_after": b["correct"],
                             "accuracy_before": a["exact_accuracy"], "accuracy_after": b["exact_accuracy"]}
    a, b = at_dict(before, "fullboard", "summary", "full_board"), at_dict(after, "fullboard", "summary", "full_board")
    as_dict(comparison["fullboard"]).update({f"{metric}_{when}": result[metric]
        for when, result in (("before", a), ("after", b))
        for metric in ("board_exact", "occupied_layout_macro_accuracy", "by_layout")})
    return comparison
