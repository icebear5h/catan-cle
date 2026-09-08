"""Bounded fresh-Qwen pilot for image -> complete board state."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import modal

from sft.board_state_readout import select_validation, summarize_board_states
from sft.modal_catan_vision_sft import hf_cache, sft_data, sft_runs, training_image, upload_training_bundle
from sft.scripts.train_trl_catan_vision import TrainConfig, iter_jsonl, sha256_file, write_json_atomic

app = modal.App("catan-full-board-readout-pilot")
SNAPSHOT = "/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
ROOT = Path("artifacts/generated/board_recognition/replay_v1")
DATASET = ROOT / "full_board_readout_v1"
VOLUMES = {"/cache": hf_cache, "/data": sft_data, "/runs": sft_runs}
GPU_OPTIONS = dict(image=training_image, gpu="H200", cpu=(16.0, 16.0),
                   memory=(131072, 131072), startup_timeout=300, retries=0,
                   max_containers=1, scaledown_window=2, volumes=VOLUMES)


@app.function(image=training_image, cpu=(4.0, 4.0), memory=(8192, 8192),
              timeout=300, retries=0, volumes=VOLUMES)
def preflight(payload: dict) -> dict:
    from transformers import AutoTokenizer, AddedToken
    from sft.scripts.train_trl_catan_vision import inspect_jsonl_contract, load_token_inventory

    if not Path(SNAPSHOT).is_dir():
        raise FileNotFoundError("fresh base snapshot is not cached")
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    inventory = load_token_inventory(payload["token_inventory"])
    tokenizer.add_tokens([AddedToken(t, normalized=False, special=False) for t in inventory["atlas_tokens"]])
    report = inspect_jsonl_contract(payload["train_jsonl"], payload["image_root"], require_curriculum=False)
    lengths = [len(tokenizer.encode(r["messages"][-1]["content"], add_special_tokens=False))
               for _, r in iter_jsonl(Path(payload["train_jsonl"]))]
    report["answer_tokens"] = {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)}
    if max(lengths) >= 1280:
        raise ValueError("increase the generation allowance before launching")
    return report


@app.function(**GPU_OPTIONS, timeout=3600)
def train(payload: dict) -> dict:
    import os
    import torch
    from sft.scripts.train_trl_catan_vision import run_training

    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.set_num_threads(16)
    config = TrainConfig(**payload)
    if config.frozen_bundle or config.olora or config.resume_from_checkpoint or not 1 <= config.max_steps <= 128:
        raise ValueError("this full-board stage is bounded to at most 128 steps")
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    result = {"status": "running", "config": payload, "timeout_seconds": 3600}
    write_json_atomic(output / "pilot.json", result)
    sft_runs.commit()
    try:
        result["training"] = run_training(config)
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write_json_atomic(output / "pilot.json", result)
        sft_runs.commit()
    return result


@app.function(**GPU_OPTIONS, timeout=3600)
def evaluate(payload: dict, steps: int, output_tag: str = "", all_layout_controls: bool = False) -> dict:
    import torch
    from sft.scripts.eval_qwen_vl_adapter import load_model, run_eval_job
    from sft.scripts.train_trl_catan_vision import load_visual_state, resolve_wrapped_module

    torch.set_num_threads(16)
    config = TrainConfig(**payload)
    checkpoint = Path(config.output_dir) / "checkpoints" / f"checkpoint-{steps}"
    if output_tag and (not output_tag.replace("-", "").isalnum()):
        raise ValueError("invalid evaluation tag")
    output = Path(config.output_dir) / (f"generated-eval-{steps}" + ("-" + output_tag if output_tag else ""))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    started = time.monotonic()
    result = {"status": "running", "checkpoint": str(checkpoint), "variants": {}}
    try:
        model, processor, evidence = load_model(
            model_id=SNAPSHOT, adapter_dir=str(checkpoint), bits=16,
            disable_flash_attn2=True, token_inventory=config.token_inventory,
        )
        # Restore the FP32 checkpoint after the common evaluator's BF16 load.
        resolve_wrapped_module(model, "model.visual").float()
        evidence["visual_state"] = {"loaded": True, **load_visual_state(model, checkpoint)}
        args = argparse.Namespace(
            image_root=config.eval_image_root, limit=None, batch_size=4, long_batch_size=4,
            max_new_tokens=1280, long_max_new_tokens=1280, candidate_scoring=False,
            occlusion_margin=0.03, model_id=SNAPSHOT, adapter_dir=str(checkpoint), bits=16,
            token_inventory=config.token_inventory,
        )
        selected = [r for _, r in iter_jsonl(Path(config.eval_jsonl))]
        # Include real pieces in controls; layout representatives are empty boards.
        controls = output / "controls.jsonl"
        control_rows = [next(r for r in selected if r["density_bin"] == density)
                        for density in ("dense", "sparse", "setup", "empty")]
        control_rows.append(next(r for r in selected if r["layout_id"] != control_rows[0]["layout_id"]))
        if all_layout_controls:
            control_rows = select_validation(selected, count=len({r["layout_id"] for r in selected}))
            if any(r["density_bin"] != "dense" for r in control_rows):
                raise ValueError("control panel requires a dense board from each layout")
        controls.write_text("".join(json.dumps(r) + "\n" for r in control_rows))
        result["control_ids"] = [r["row_id"] for r in control_rows]
        for label, source, variant in (("heldout", config.eval_jsonl, "original"),
                                       ("blank", str(controls), "blank")):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                summary = run_eval_job(model=model, processor=processor, adapter_evidence=evidence,
                                      args=args, eval_jsonl=source, image_variant=variant,
                                      output_dir=output / label)
            result["variants"][label] = summary["full_board"]
            if label == "heldout":
                records = [r for _, r in iter_jsonl(output / label / "records.jsonl")]
                result["matched_control_originals"] = summarize_board_states([
                    r for r in records if r["id"] in result["control_ids"]
                ])
            write_json_atomic(output / "pilot_eval.json", result)
            sft_runs.commit()
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write_json_atomic(output / "pilot_eval.json", result)
        sft_runs.commit()
    return result



@app.function(image=training_image, cpu=(0.25, 0.25), memory=(2048, 2048),
              timeout=5400, retries=0, scaledown_window=2, volumes=VOLUMES)
def complete_pilot(payload: dict, training_call_id: str) -> dict:
    """Wait without a GPU, then run the fixed evaluation once training succeeds."""
    result = {"status": "waiting_for_training", "training_call_id": training_call_id}
    try:
        training = modal.FunctionCall.from_id(training_call_id).get(timeout=3900)
        if training["status"] != "completed":
            raise RuntimeError(f"training did not complete: {training['status']}")
        result["training"] = training
        result["evaluation"] = evaluate.remote(payload, payload["max_steps"])
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        sft_runs.reload()
        write_json_atomic(Path(payload["output_dir"]) / "completed_pilot.json", result)
        sft_runs.commit()
    return result


@app.local_entrypoint()
def main(execute: bool = False, steps: int = 128, run_name: str = "full-board-fresh-20260907",
         eval_receipt: str = "", follow_receipt: str = ""):
    if follow_receipt:
        receipt = json.loads(Path(follow_receipt).read_text())
        target = Path(follow_receipt).with_name("completion-launch.json")
        if target.exists():
            raise FileExistsError(target)
        if not execute:
            print(json.dumps({"wait_for": receipt["call_id"], "then": "evaluate saved checkpoint"}))
            return
        call = complete_pilot.spawn(receipt["config"], receipt["call_id"])
        write_json_atomic(target, {"call_id": call.object_id, "training_call_id": receipt["call_id"],
                                  "config": receipt["config"]})
        print(target.read_text())
        return
    if eval_receipt:
        receipt = json.loads(Path(eval_receipt).read_text())
        payload = receipt["config"]
        target = Path(eval_receipt).with_name("eval-launch.json")
        if target.exists():
            raise FileExistsError(target)
        if not execute:
            print(json.dumps({"evaluation_of": payload["output_dir"], "steps": payload["max_steps"]}))
            return
        call = evaluate.spawn(payload, payload["max_steps"])
        write_json_atomic(target, {"call_id": call.object_id, "config": payload})
        print(target.read_text())
        return
    if not 1 <= steps <= 128:
        raise ValueError("steps must be between 1 and 128")
    rows = [r for _, r in iter_jsonl(DATASET / "stage1/validation.jsonl")]
    selected = select_validation(rows)
    receipt_dir = Path("artifacts/runs/sft") / run_name
    if (receipt_dir / "launch.json").exists():
        raise FileExistsError(receipt_dir / "launch.json")
    if not execute:
        print(json.dumps({"fresh_base": SNAPSHOT, "steps": steps, "batch": 4, "accumulation": 2,
                          "train_boards": 1024, "eval_boards": len(selected), "timeout_seconds": 3600}))
        return
    with tempfile.TemporaryDirectory() as directory:
        sample = Path(directory) / "eval.jsonl"
        sample.write_text("".join(json.dumps(r) + "\n" for r in selected))
        paths = upload_training_bundle(
            DATASET / "stage1/train.jsonl", DATASET / "images",
            ROOT / "ms_swift_bidirectional_v1/trainable_tokens.json",
            remote_dir=f"catan-vision-sft/datasets/{run_name}", require_curriculum=False,
            eval_jsonl=sample, eval_image_root=DATASET / "images",
        )
    config = TrainConfig(
        train_jsonl=paths[0], image_root=paths[2], token_inventory=paths[3],
        eval_jsonl=paths[1], eval_image_root=paths[2], model_id=SNAPSHOT,
        output_dir=f"/runs/catan-vision-sft/{run_name}", max_steps=steps,
        per_device_train_batch_size=4, gradient_accumulation_steps=2,
        per_device_eval_batch_size=2, dataloader_num_workers=0,
        save_steps=min(32, steps), eval_steps=min(32, steps), save_total_limit=4,
        require_curriculum=False, token_init="vocab_gaussian", lora_rank=8, lora_alpha=16,
    )
    config.validate()
    audit = preflight.remote(asdict(config))
    print(json.dumps({"preflight": audit}), flush=True)
    call = train.spawn(asdict(config))
    write_json_atomic(receipt_dir / "launch.json", {"call_id": call.object_id,
        "config": asdict(config), "preflight": audit, "eval_ids": [r["row_id"] for r in selected],
        "trainer_sha256": sha256_file(Path("sft/scripts/train_trl_catan_vision.py")),
        "dataset_metadata": json.loads((DATASET / "metadata.json").read_text())})
    print((receipt_dir / "launch.json").read_text())
