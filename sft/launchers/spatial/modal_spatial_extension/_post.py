"""Post-training evaluation across every panel."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from sft.json_types import JsonDict, as_dict, as_str, opt_str
from sft.launchers._json import at_dict, at_str
from sft.launchers.spatial import modal_spatial_extension as extension
from sft.launchers.spatial.modal_spatial_continuation import (
    PANEL_BUDGETS,
    SNAPSHOT,
    checkpoint_audit,
    dataset_identity,
    digest,
    evaluation_conditions,
    evaluator,
    now,
    panel_args,
    reload_volumes,
)
from sft.scripts.train.train_trl_catan_vision import (
    sha256_file,
    write_json_atomic,
)

from ._base import POST_GPU_OPTIONS, app
from ._plan import verify_runtime


@app.function(**POST_GPU_OPTIONS)
def evaluate_bounded(plan: JsonDict, audit: JsonDict, training: JsonDict) -> JsonDict:
    reload_volumes()
    verify_runtime(plan)
    config = at_dict(plan, "config")
    checkpoint = str(Path(at_str(config, "output_dir")) / "checkpoints/checkpoint-256")
    if training["status"] != "completed" or training["checkpoint"] != checkpoint:
        raise ValueError("post evaluation requires completed checkpoint-256 training")
    checkpoint_report = checkpoint_audit(Path(checkpoint), config, parent=False, expected_step=256,
                                         expected_audit=as_dict(training["checkpoint_audit"]))
    output = extension.RUN_ROOT / "pipelines" / at_str(plan, "run_name") / "post"
    output.mkdir(parents=True, exist_ok=False)
    panels: JsonDict = {}
    result: JsonDict = {"status": "running", "checkpoint": checkpoint, "checkpoint_audit": checkpoint_report,
                        "started_at": now(), "panels": panels, "source_sha256": plan["source_sha256"]}
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        torch.set_num_threads(16)
        model, processor, evidence = evaluator.load_model(
            model_id=SNAPSHOT, adapter_dir=checkpoint, bits=16, disable_flash_attn2=True,
            token_inventory=at_str(config, "token_inventory"), preserve_visual_fp32=True)
        for label, panel_value in at_dict(plan, "panels").items():
            panel = as_dict(panel_value)
            eval_jsonl = as_str(panel["eval_jsonl"])
            identity = dataset_identity(eval_jsonl, opt_str(panel["image_root"]))
            if identity != at_dict(audit, "panels")[label] or identity != panel["identity"]:
                raise ValueError("uploaded panel bytes/pixels changed after CPU preflight")
            summary = evaluator.run_eval_job(model=model, processor=processor, adapter_evidence=evidence,
                args=panel_args(plan, checkpoint, label), eval_jsonl=eval_jsonl,
                image_variant="original", output_dir=output / label)
            if summary["rows"] != PANEL_BUDGETS[label][0] or summary["attempted"] != summary["rows"]:
                raise ValueError("evaluation returned incomplete panel")
            records = output / label / "records.jsonl"
            panels[label] = {"summary": summary, "identity": identity,
                "records_path": str(records), "records_sha256": sha256_file(records),
                "scorer_sha256": digest(plan["source_sha256"]), "conditions": evaluation_conditions(label)}
            write_json_atomic(output / "result.json", result)
            extension.sft_runs.commit()
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["ended_at"] = now()
        write_json_atomic(output / "result.json", result)
        extension.sft_runs.commit()
    return result
