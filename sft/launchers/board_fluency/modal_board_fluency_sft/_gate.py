from __future__ import annotations

import gc
import shutil
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import save_file

from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_str
from sft.launchers._train_config import train_config
from sft.launchers.board_fluency import modal_board_fluency_eval as shared
from sft.launchers.modal_catan_vision_sft import sft_runs
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train import train_trl_catan_vision as trainer
from sft.scripts.train.train_trl_catan_vision import sha256_file, write_json_atomic

from ._callbacks import GateCallback
from ._config import BASE, EVALUATOR_FILES, PARENT, TRAINER_FILES
from ._data import progress, rows_at
from ._planning import check_manifest
from ._probes import (
    compare_probes,
    load_eval,
    probe,
    probe_metadata,
    visual_digest,
    visual_file_digest,
)


def prepared_for(plan: JsonDict) -> JsonDict:
    receipt = shared.read_json(Path(as_str(plan["root"])) / "prepare/result.json")
    if receipt["status"] != "completed" or receipt["launch_sha256"] != shared.digest(plan):
        raise ValueError("CPU preparation did not complete for this launch")
    prepared = as_dict(receipt["result"])
    plan_config = as_dict(plan["config"])
    check_manifest(Path(PARENT), as_dict(prepared["parent_files"]))
    check_manifest(Path(as_str(plan_config["initial_bundle"])),
                   as_dict(prepared["expanded_files"]))
    check_manifest(Path(BASE), as_dict(prepared["base_files"]))
    if sha256_file(Path(as_str(plan_config["eval_jsonl"]))) != prepared["teacher_sha256"]:
        raise ValueError("fixed teacher-forced panel changed")
    return prepared


def retained_prevalidation(plan: JsonDict, prepared: JsonDict) -> JsonLikeDict:
    """Reuse verified completed predictions without claiming a complete baseline."""
    reference = as_dict(plan["prevalidation_reference"])
    root = Path(as_str(reference["root"]))
    previous = shared.read_json(root / "launch.json")
    previous_hashes = as_dict(previous["source_sha256"])
    plan_hashes = as_dict(plan["source_sha256"])
    if previous["parent"] != PARENT or previous["base"] != BASE:
        raise ValueError("retained baseline checkpoint differs")
    if (as_dict(as_dict(previous["inputs"])["files"])["validation_eval.jsonl"]
            != as_dict(as_dict(plan["inputs"])["files"])["validation_eval.jsonl"]):
        raise ValueError("retained baseline input differs")
    for name in ("sft/board/board_fluency_scoring.py",):
        if previous_hashes[name] != plan_hashes[name]:
            raise ValueError(f"retained baseline inference/scoring code changed: {name}")
    # The evaluator is a package now; old receipts pin the single pre-split file while new
    # receipts pin every module. Any key or hash difference fails closed instead of KeyError.
    previous_evaluator = {k: v for k, v in previous_hashes.items()
                          if k.startswith("sft/scripts/eval/eval_qwen_vl_adapter")}
    if previous_evaluator != {k: plan_hashes[k] for k in EVALUATOR_FILES}:
        raise ValueError("retained baseline inference/scoring code changed: eval_qwen_vl_adapter")
    # The trainer is a package now; old receipts pin the single pre-split file while new
    # receipts pin every module. Any key or hash difference fails closed instead of KeyError.
    previous_trainer = {k: v for k, v in previous_hashes.items()
                        if k.startswith("sft/scripts/train/train_trl_catan_vision")}
    if previous_trainer != {k: plan_hashes[k] for k in TRAINER_FILES}:
        raise ValueError("retained baseline inference/scoring code changed: train_trl_catan_vision")
    old_preparation = shared.read_json(root / "prepare/result.json")
    if old_preparation["status"] != "completed":
        raise ValueError("retained baseline preparation was incomplete")
    old_files = as_dict(as_dict(old_preparation["result"])["expanded_files"])
    for name in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "chat_template.jinja", trainer.VISUAL_STATE_FILE):
        if old_files[name] != as_dict(prepared["expanded_files"])[name]:
            raise ValueError(f"retained baseline initialization changed: {name}")
    if shared.read_json(root / "gate/equivalence.json")["max_abs"] != 0.0:
        raise ValueError("retained baseline lacks exact parent parity evidence")
    source = root / "gate/pre-validation190/records.jsonl"
    if sha256_file(source) != reference["records_sha256"]:
        raise ValueError("retained raw predictions changed")
    records = rows_at(source)
    ids = [as_str(row["id"]) for row in records]
    expected = {as_str(row["id"]): row for row
                in rows_at(Path(as_str(plan["data_dir"])) / "validation_eval.jsonl")}
    if len(ids) != reference["completed_rows"] or len(set(ids)) != len(ids) or not set(ids) <= set(expected):
        raise ValueError("retained baseline ID coverage differs")
    for record in records:
        row = expected[as_str(record["id"])]
        metadata = evaluator.evaluation_metadata(row, image_variant="original")
        metadata["input_mode"] = "text"
        if (record["expected"] != evaluator.expected_text(row) or record["metadata"] != metadata
                or record["score"] != evaluator.score_response(
                    as_str(record["expected"]), as_str(record["response"]), metadata=metadata)):
            raise ValueError("retained baseline score/metadata differs")
    output = Path(as_str(plan["root"])) / "gate/pre-validation190"
    output.mkdir()
    shutil.copyfile(source, output / "records.jsonl")
    summary = {**evaluator.summarize(records), "status": "partial", "requested_rows": 190,
               "source": str(source), "source_sha256": reference["records_sha256"],
               "missing_ids": sorted(set(expected) - set(ids))}
    write_json_atomic(output / "summary.json", summary)
    sft_runs.commit()
    return {"correct": summary["correct"], "rows": len(ids), "ids": ids,
            "partial": True, "requested_rows": 190, "output_dir": str(output),
            "files": shared.file_manifest(output, ["records.jsonl", "summary.json"])}


def gate_work(plan: JsonDict, deadline: float) -> JsonLikeDict:
    prepared = prepared_for(plan)
    plan_config = as_dict(plan["config"])
    config = replace(train_config(plan_config), max_steps=2, save_steps=2, eval_steps=2,
                     eval_jsonl=None, output_dir=as_str(plan["root"]) + "/gate/training")
    probe_rows = rows_at(Path(as_str(plan_config["train_jsonl"])))[:4]
    model, tokenizer, evidence = load_eval(plan, PARENT)
    parent_probe = probe(model, tokenizer, probe_rows)
    parent_visual = visual_digest(model)
    del model, tokenizer, evidence
    gc.collect()
    torch.cuda.empty_cache()
    initial_bundle = config.initial_bundle
    if initial_bundle is None:
        raise ValueError("gate requires the expanded initial bundle")
    model, tokenizer, evidence = load_eval(plan, initial_bundle)
    expanded_probe = probe(model, tokenizer, probe_rows)
    save_file({"parent": parent_probe["logits"], "expanded": expanded_probe["logits"]},
              Path(as_str(plan["root"])) / "gate/parity_logits.safetensors")
    write_json_atomic(Path(as_str(plan["root"])) / "gate/parity_metadata.json", {
        "parent": probe_metadata(parent_probe),
        "expanded": probe_metadata(expanded_probe),
        "precision": "text inference: no autocast; BF16 base, FP32 LoRA",
    })
    sft_runs.commit()
    equivalence = compare_probes(parent_probe, expanded_probe)
    if visual_digest(model) != parent_visual:
        raise ValueError("expanded model changed the preserved visual tensor hash")
    write_json_atomic(Path(as_str(plan["root"])) / "gate/equivalence.json", equivalence)
    sft_runs.commit()
    baseline = retained_prevalidation(plan, prepared)
    del model, tokenizer, evidence, parent_probe
    gc.collect()
    torch.cuda.empty_cache()
    progress("gate: two real training steps in an isolated output directory")
    callback = GateCallback(config, deadline, probe_rows, expanded_probe)
    training = trainer.run_training(config, extra_callbacks=[callback])
    if training["status"] != "completed" or callback.saved_steps != [2] or callback.after is None:
        raise RuntimeError("two-step training gate did not complete")
    checkpoint = Path(config.output_dir) / "checkpoints/checkpoint-2"
    frozen_visual = visual_file_digest(Path(PARENT) / trainer.VISUAL_STATE_FILE)
    if visual_file_digest(checkpoint / trainer.VISUAL_STATE_FILE) != frozen_visual:
        raise ValueError("gate save changed frozen visual tensor values")
    model, tokenizer, evidence = load_eval(plan, str(checkpoint))
    reload_equivalence = compare_probes(callback.after, probe(model, tokenizer, probe_rows))
    if visual_digest(model) != parent_visual:
        raise ValueError("gate reload changed the visual tensors")
    check_manifest(Path(initial_bundle), as_dict(prepared["expanded_files"]))
    return {"equivalence": equivalence, "updates": callback.report,
             "reload_equivalence": reload_equivalence, "pre_validation190": baseline,
             "training": training, "gate_output_is_main_initialization": False,
             "frozen_visual_file_tensor_sha256": frozen_visual}
