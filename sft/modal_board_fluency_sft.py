"""Opt-in, single-run board-fluency continuation; dry runs never contact Modal.

MODAL_PROFILE=icebear5h CATAN_HF_SECRET_NAME=huggingface-secret-2 \
    .venv/bin/python -m sft.modal_board_fluency_sft --run-name NAME --budget-usd 15
Add --execute only after reviewing the local admission/configuration receipt.
The detached CPU coordinator owns preparation, gate, training and post-evaluation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import io
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import modal
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import TrainerCallback

from evals.catan_board_bench.tokens import atlas_tokens
from sft import modal_board_fluency_eval as shared
from sft.lora_expansion import expand_lora_bundle
from sft.modal_board_fluency_eval import VOLUMES, eval_image
from sft.modal_catan_vision_sft import HF_SECRET_NAME, sft_data, sft_runs
from sft.paths import PROJECT_ROOT
from sft.scripts import eval_qwen_vl_adapter as evaluator
from sft.scripts import train_trl_catan_vision as trainer
from sft.scripts.train_trl_catan_vision import (
    TrainConfig, _message_pair, encode_text_pair, load_token_inventory,
    pad_text_inputs, sha256_file, write_json_atomic,
)

from sft.scripts.build_board_fluency_dataset import validate_dataset


PARENT = "/runs/catan-vision-sft/spatial-continuation-20260912-r01/checkpoints/checkpoint-128"
BASE = ("/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/"
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
DATASET = PROJECT_ROOT / "artifacts/generated/sft/symbolic_board_fluency_sft_v1"
REVIEW = PROJECT_ROOT / shared.DEFAULT_REVIEW
REVIEW_SHA256 = "40469f58d060dd9a1d24354df42d959971172ba6e0de5e70b153cdda747e5b12"
DATA_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl", "validation_eval.jsonl",
              "trainable_tokens.json", "metadata.json", "manifest.json")
STAGE_SECONDS = {"prepare": 1200, "gate": 450, "train": 3300, "posteval": 1800}
STARTUP = 300
COORDINATOR_SECONDS = 9000
CONTROL_CPU = 1.0
# Includes r01/r02 CPU starts, r03 parity check and r04's capped baseline run.
PRIOR_ATTEMPTS_USD = 2.50
BASELINE_ROOT = "/runs/catan-vision-sft/board-fluency-sft-20260915-r04"
BASELINE_SHA256 = "2fc6ce78b4f1620d1b186a8e4052cf38d3cad4841250d1522703c2e9edccdd00"
ABSOLUTE_SECONDS = 8700
ATOL, RTOL = 0.002, 0.0002
ALLOWED = {"language_lora", "atlas_input_rows", "atlas_output_rows"}
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "PYTHONUNBUFFERED": "1"}
SOURCE_FILES = (
    "sft/modal_board_fluency_sft.py", "sft/modal_board_fluency_eval.py",
    "sft/lora_expansion.py", "sft/scripts/build_board_fluency_dataset.py",
    "sft/scripts/train_trl_catan_vision.py", "sft/scripts/eval_qwen_vl_adapter.py",
    "sft/board_fluency_scoring.py",
)
app = modal.App("catan-board-fluency-sft")
secret = modal.Secret.from_name("huggingface-secret-2", required_keys=["HF_TOKEN"])
COMMON = dict(image=eval_image, volumes=VOLUMES, secrets=[secret], startup_timeout=STARTUP,
              retries=0, max_containers=1, scaledown_window=2)
GPU = dict(**COMMON, gpu="H200", cpu=(16.0, 16.0), memory=(128 * 1024, 128 * 1024))


def progress(message: str) -> None:
    print(f"[board-fluency {shared.now()}] {message}", flush=True)


def budget_plan(budget_usd: float) -> dict:
    if not math.isfinite(budget_usd) or not 0 < budget_usd <= 15:
        raise ValueError("--budget-usd must be positive and at most the approved $15")
    gpu_rate = 0.001261 + 16 * 0.0000131 + 128 * 0.00000222
    gpu_seconds = sum(STAGE_SECONDS[s] + STARTUP for s in ("gate", "train", "posteval"))
    gpu_max = gpu_seconds * gpu_rate
    cpu_max = (STAGE_SECONDS["prepare"] + STARTUP) * (4 * 0.0000131 + 16 * 0.00000222)
    coordinator_max = (COORDINATOR_SECONDS + STARTUP) * (CONTROL_CPU * 0.0000131 + 2 * 0.00000222)
    # Includes idle scaledown, cancellation RPC/termination grace and reservation.
    reserve = 0.75
    upper = gpu_max + cpu_max + coordinator_max + reserve + PRIOR_ATTEMPTS_USD
    if upper > budget_usd:
        raise ValueError(f"bounded compute envelope ${upper:.6f} exceeds ${budget_usd:.2f}")
    return {"approved_usd": budget_usd, "rates_checked": "2026-09-15",
            "h200_per_second": 0.001261, "cpu_core_per_second": 0.0000131,
            "memory_gib_per_second": 0.00000222, "gpu_stage_per_second": gpu_rate,
            "gpu_seconds_including_startups": gpu_seconds, "gpu_upper_usd": gpu_max,
            "prepare_upper_usd": cpu_max, "coordinator_upper_usd": coordinator_max,
             "termination_and_control_reserve_usd": reserve, "compute_upper_usd": upper,
             "prior_attempts_allowance_usd": PRIOR_ATTEMPTS_USD,
            "excluded": "unrelated jobs, storage, image storage/build charges and subscriptions",
            "policy": "fixed checked rates; no automatic retry, extension or replacement stage"}


def configuration(run_name: str) -> TrainConfig:
    shared.validate_cli("", "", shared.MODEL_ID, shared.MODEL_REVISION, run_name, "", PARENT)
    root = f"/runs/catan-vision-sft/{run_name}"
    if root == str(Path(PARENT).parents[1]):
        raise ValueError("run name collides with the preserved parent")
    data = f"/data/board-fluency-sft/{run_name}"
    return TrainConfig(
        train_jsonl=data + "/train.jsonl", token_inventory=data + "/trainable_tokens.json",
        output_dir=root + "/training", eval_jsonl=root + "/prepare/teacher120.jsonl",
        model_id=BASE, profile="vision_tokens_lora", input_mode="text",
        max_sequence_length=4096, initial_bundle=root + "/expanded-r16", token_init="keep",
        max_steps=128, per_device_train_batch_size=4, gradient_accumulation_steps=2,
        per_device_eval_batch_size=4, learning_rate=1e-4, language_lora_learning_rate=5e-5,
        lora_rank=16, lora_alpha=32, lora_dropout=0.05, save_steps=32, eval_steps=32,
        save_total_limit=4, dataloader_num_workers=0, seed=44, require_curriculum=False,
        resume_from_checkpoint=None, publish_to_hub=False,
    )


def require_dependencies() -> None:
    callback = inspect.signature(trainer.run_training).parameters.get("extra_callbacks")
    if callback is None or callback.kind != inspect.Parameter.KEYWORD_ONLY:
        raise RuntimeError("run_training(config, *, extra_callbacks=None) is required")


def source_hashes() -> dict:
    return {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES}


def rows_at(path: Path) -> list[dict]:
    return [row for _, row in evaluator.iter_jsonl(path)]


def row_contract(rows: list[dict]) -> dict:
    ids, operations, families, states = [], Counter(), Counter(), set()
    for line, row in enumerate(rows, 1):
        row_id = row.get("id") or row.get("row_id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"missing row ID at line {line}")
        _, gold = _message_pair(row, line_number=line, input_mode="text")
        metadata = evaluator.evaluation_metadata(row, image_variant="original")
        score = evaluator.score_response(gold, gold, metadata=metadata)
        if score.get("correct") is not True or score.get("scoring") != metadata.get("schema"):
            raise ValueError(f"stored gold did not use the fluency schema scorer: {row_id}")
        state = row.get("metadata", {}).get("state_sha256")
        if not isinstance(state, str) or len(state) != 64:
            raise ValueError(f"missing canonical state identity: {row_id}")
        states.add(state)
        ids.append(row_id)
        operations[metadata["operation"]] += 1
        families[metadata["family"]] += 1
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("empty split or duplicate row IDs")
    return {"rows": len(rows), "ids": ids, "state_sha256": sorted(states),
            "by_operation": dict(operations), "by_family": dict(families)}


def teacher_ids(rows: list[dict]) -> list[str]:
    """First 120 in deterministic operation-balanced order; never sample per eval."""
    groups = defaultdict(deque)
    for row in rows:
        groups[row["metadata"]["operation"]].append(row.get("id") or row["row_id"])
    chosen, counts = [], Counter()
    while len(chosen) < 120:
        for operation in sorted(groups):
            if groups[operation] and len(chosen) < 120:
                chosen.append(groups[operation].popleft())
                counts[operation] += 1
        if not any(groups.values()) and len(chosen) < 120:
            raise ValueError("validation_eval needs at least 120 rows")
    # Two pip operations have only five independent heldout layouts each.
    # Saturate those real examples rather than duplicate them to pad a quota.
    unsaturated = [counts[op] for op, remaining in groups.items() if remaining]
    if (not unsaturated or max(unsaturated) - min(unsaturated) > 1
            or set(counts) != set(groups)):
        raise ValueError("validation_eval cannot supply a balanced fixed teacher-forced panel")
    return chosen


def inspect_data(root: Path, review: Path) -> dict:
    inventory = load_token_inventory(root / "trainable_tokens.json")
    if sha256_file(review) != REVIEW_SHA256:
        raise ValueError("the unchanged 200-row review hash differs")
    panels = {name: row_contract(rows_at(root / f"{name}.jsonl"))
              for name in ("train", "validation", "test", "validation_eval")}
    panels["review"] = row_contract(rows_at(review))
    if (panels["train"]["rows"] != 3200 or len(panels["train"]["by_operation"]) != 20
            or panels["validation_eval"]["rows"] != 190 or panels["review"]["rows"] != 200):
        raise ValueError("expected train3200/20 operations, validation_eval190 and review200")
    for name in ("validation", "test"):
        if not 300 <= panels[name]["rows"] <= 450:
            raise ValueError(f"{name} differs from the approved approximately 370 rows")
    for name in ("train", "validation", "test", "validation_eval"):
        if set(panels[name]["state_sha256"]) & set(panels["review"]["state_sha256"]):
            raise ValueError(f"review state leaked into {name}")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(panels[a]["state_sha256"]) & set(panels[b]["state_sha256"]):
            raise ValueError(f"state leakage between {a} and {b}")
    validation = {r.get("id") or r["row_id"]: r for r in rows_at(root / "validation.jsonl")}
    eval_rows = rows_at(root / "validation_eval.jsonl")
    if any(validation.get(r.get("id") or r["row_id"]) != r for r in eval_rows):
        raise ValueError("validation_eval must contain unchanged validation rows")
    return {"panels": panels, "teacher_ids": teacher_ids(eval_rows),
            "files": shared.file_manifest(root, list(DATA_FILES)),
            "review_sha256": REVIEW_SHA256, "atlas_tokens": inventory["tokens"]}


def build_plan(run_name: str, budget_usd: float) -> dict:
    config = configuration(run_name)
    budget = budget_plan(budget_usd)
    require_dependencies()
    config.validate()
    if (PROJECT_ROOT / "artifacts/runs/sft" / run_name).exists():
        raise FileExistsError("local run name already used")
    # The data agent owns semantic recomputation/provenance admission. Preserve
    # its real report, then pin the exact bytes the launcher actually uploads.
    validation = validate_dataset(DATASET)
    if validation.get("valid") is not True or validation.get("admitted_for_training") is not True:
        raise ValueError("dataset validator did not admit this corpus for SFT")
    inputs = inspect_data(DATASET, REVIEW)
    root = str(Path(config.output_dir).parent)
    plan = {"schema": "catan_board_fluency_sft_launch/v1", "run_name": run_name,
            "created_at": shared.now(), "reservation_id": uuid.uuid4().hex,
            "root": root, "data_dir": str(Path(config.train_jsonl).parent),
            "parent": PARENT, "base": BASE, "model_revision": shared.MODEL_REVISION,
            "profile": "icebear5h", "hf_secret_name": "huggingface-secret-2",
            "config": asdict(config), "budget": budget, "inputs": inputs,
            "dataset_validation": validation, "source_sha256": source_hashes(),
            "limits": {"stage_seconds": STAGE_SECONDS, "startup_seconds": STARTUP,
                       "coordinator_seconds": COORDINATOR_SECONDS,
                       "absolute_seconds": ABSOLUTE_SECONDS, "retries": 0,
                       "max_containers": 1, "scaledown_seconds": 2},
             "gate_tolerance": {"atol": ATOL, "rtol": RTOL},
             "prevalidation_reference": {"root": BASELINE_ROOT, "records_sha256": BASELINE_SHA256,
                                         "completed_rows": 144, "requested_rows": 190},
             "review_baseline": {"correct": 24, "rows": 200,
                                 "source": "user-approved retained baseline", "review_sha256": REVIEW_SHA256}}
    # Audits contain integer histogram keys. Normalize once before both RPC and
    # persistence so the remote object is identical to its JSON receipt.
    return json.loads(json.dumps(plan, allow_nan=False))


def verify_plan(plan: dict) -> None:
    if plan["config"] != asdict(configuration(plan["run_name"])):
        raise ValueError("configuration differs from the approved continuation")
    if plan["budget"] != budget_plan(plan["budget"]["approved_usd"]):
        raise ValueError("budget/rate guard differs")
    if (plan["parent"], plan["base"], plan["model_revision"]) != (PARENT, BASE, shared.MODEL_REVISION):
        raise ValueError("parent or pinned base differs")
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("source files changed after local admission")
    require_dependencies()
    TrainConfig(**plan["config"]).validate()


def verify_uploaded(plan: dict, *, semantic: bool = False) -> None:
    root = Path(plan["data_dir"])
    if shared.read_json(root / "launch.json") != plan:
        raise ValueError("uploaded launch receipt differs")
    check_manifest(root, plan["inputs"]["files"])
    if sha256_file(root / "review.jsonl") != REVIEW_SHA256:
        raise ValueError("uploaded review changed")
    if semantic and inspect_data(root, root / "review.jsonl") != plan["inputs"]:
        raise ValueError("uploaded data/inventory differs from local admission")


def check_manifest(root: Path, expected: dict, *, hashes: bool = True) -> None:
    for name, evidence in expected.items():
        path = root / name
        if not path.is_file() or path.stat().st_size != evidence["bytes"]:
            raise ValueError(f"prepared file missing/incomplete: {path}")
        if hashes and "sha256" not in evidence and path.suffix != ".safetensors":
            raise ValueError(f"metadata file lacks a content hash: {path}")
        if hashes and "sha256" in evidence and sha256_file(path) != evidence["sha256"]:
            raise ValueError(f"prepared file changed: {path}")


def cached_base_files() -> list[str]:
    """Reuse the evaluator's header audit without calling its downloading snapshot()."""
    base = Path(BASE)
    if not base.is_dir() or base.name != shared.MODEL_REVISION:
        raise FileNotFoundError("the exact approved base snapshot must already be cached")
    index = base / "model.safetensors.index.json"
    weights = set(shared.read_json(index)["weight_map"].values()) if index.is_file() else {"model.safetensors"}
    if any(Path(name).name != name or not name.endswith(".safetensors") for name in weights):
        raise ValueError("unexpected cached weight shard path")
    names = weights | {name for name in shared.INFERENCE_SIDECARS if (base / name).is_file()}
    if index.is_file():
        names.add(index.name)
    if "config.json" not in names or not weights:
        raise ValueError("incomplete native base cache")
    if any(not (base / name).is_file() or not (base / name).stat().st_size for name in names):
        raise FileNotFoundError("missing base cache shard/sidefile; downloads are disabled")
    return sorted(names)


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def prepare_work(plan: dict, deadline: float) -> dict:
    verify_uploaded(plan, semantic=True)
    progress("CPU: validating pinned cache, parent and all native text boundaries")
    inventory = load_token_inventory(Path(plan["config"]["token_inventory"]))
    parent, parent_files = shared.volume_bundle(PARENT)
    tokenizer, checkpoint = shared.adapter_preflight(parent, inventory, shared.MODEL_ID, shared.MODEL_REVISION)
    if (checkpoint["adapter_config"]["r"], checkpoint["adapter_config"]["lora_alpha"]) != (8, 16):
        raise ValueError("expected the preserved rank8/alpha16 September 12 checkpoint")
    if checkpoint["saved_model_id"] != BASE:
        raise ValueError("parent must reference the exact approved cache path")
    base_files = cached_base_files()
    base_audit = shared.base_preflight(Path(BASE), base_files, parent, inventory, checkpoint)
    lengths = {}
    for split in ("train", "validation", "test", "validation_eval", "review"):
        items = rows_at(Path(plan["data_dir"]) / f"{split}.jsonl")
        maximum, max_prompt, max_answer = 0, 0, 0
        for row in items:
            prompt, answer = _message_pair(row, line_number=0, input_mode="text")
            pair = encode_text_pair(tokenizer, prompt, answer, max_sequence_length=4096)
            prefix = pair["labels"].count(-100)
            completion = len(pair["input_ids"]) - prefix
            if split in ("validation_eval", "review") and (prefix + 512 > 4096 or completion > 512):
                raise ValueError(f"{split} exceeds the greedy512 context allowance")
            maximum, max_prompt, max_answer = max(maximum, len(pair["input_ids"])), max(max_prompt, prefix), max(max_answer, completion)
        lengths[split] = {"rows": len(items), "max_sequence": maximum,
                          "max_prompt": max_prompt, "max_completion": max_answer}
        progress(f"CPU: {split} boundaries checked ({len(items)} rows)")
        check_deadline(deadline)
    directory = Path(plan["root"]) / "prepare"
    panel = {r.get("id") or r["row_id"]: r
             for r in rows_at(Path(plan["data_dir"]) / "validation_eval.jsonl")}
    write_rows(directory / "teacher120.jsonl", [panel[i] for i in plan["inputs"]["teacher_ids"]])
    parent_hashes = shared.file_manifest(parent, parent_files)
    expanded = Path(plan["config"]["initial_bundle"])
    if expanded.exists():
        raise FileExistsError("expanded bundle is immutable; choose a fresh run name")
    progress("CPU: expanding the preserved adapter to rank16/alpha32")
    conversion = expand_lora_bundle(parent, expanded, rank=16, alpha=32, seed=44)
    expanded, expanded_files = shared.volume_bundle(str(expanded))
    _, expanded_audit = shared.adapter_preflight(expanded, inventory, shared.MODEL_ID, shared.MODEL_REVISION)
    shared.base_preflight(Path(BASE), base_files, expanded, inventory, expanded_audit)
    if (expanded_audit["adapter_config"]["r"], expanded_audit["adapter_config"]["lora_alpha"]) != (16, 32):
        raise ValueError("converter did not produce rank16/alpha32")
    if expanded_audit["semantic_tokens"]["token_ids"] != checkpoint["semantic_tokens"]["token_ids"]:
        raise ValueError("conversion changed atlas token IDs")
    check_manifest(parent, parent_hashes)
    expanded_hashes = shared.file_manifest(expanded, expanded_files + ["lora_expansion.json"])
    if expanded_hashes[trainer.VISUAL_STATE_FILE] != parent_hashes[trainer.VISUAL_STATE_FILE]:
        raise ValueError("conversion changed the preserved visual sidefile")
    return {"checkpoint": checkpoint, "base_audit": base_audit, "conversion": conversion,
            "parent_files": parent_hashes, "expanded_files": expanded_hashes,
             "base_files": {name: {"bytes": (Path(BASE) / name).stat().st_size,
                                    **({"sha256": sha256_file(Path(BASE) / name)}
                                       if not name.endswith(".safetensors") else {})}
                            for name in base_files},
            "teacher_sha256": sha256_file(directory / "teacher120.jsonl"), "token_lengths": lengths}


def check_deadline(deadline: float) -> None:
    if time.time() >= deadline:
        raise TimeoutError("fixed stage deadline reached; no automatic extension")


@contextmanager
def deadline_alarm(deadline: float):
    def expired(signum, frame):
        raise TimeoutError("inner stage deadline reached; preserving completed artifacts")

    check_deadline(deadline)
    started = time.time()
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    old_handler = signal.signal(signal.SIGALRM, expired)
    allowance = deadline - started
    signal.setitimer(signal.ITIMER_REAL, min(allowance, old_timer[0]) if old_timer[0] else allowance)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        remaining = old_timer[0] - (time.time() - started)
        if old_timer[0] and remaining > 0:
            signal.setitimer(signal.ITIMER_REAL, remaining, old_timer[1])


def visual_digest(model: Any) -> str:
    digest = hashlib.sha256()
    visual = trainer.resolve_wrapped_module(model, "model.visual")
    for name, tensor in sorted(visual.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def visual_file_digest(path: Path) -> str:
    """Hash every visual tensor, not nondeterministic safetensors JSON ordering."""
    digest = hashlib.sha256()
    with safe_open(path, framework="pt", device="cpu") as handle:
        names = sorted(handle.keys())
        if len(names) != 333:
            raise ValueError("expected the full 333-tensor visual sidefile")
        for name in names:
            tensor = handle.get_tensor(name)
            if tensor.dtype != torch.float32:
                raise ValueError(f"frozen visual precision changed: {name}")
            digest.update(name.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def probe(model: Any, tokenizer: Any, rows: list[dict]) -> dict:
    """Full active-vocabulary logits at fixed real native prompt boundaries."""
    features = []
    for row in rows:
        prompt, answer = _message_pair(row, line_number=0, input_mode="text")
        pair = encode_text_pair(tokenizer, prompt, answer, max_sequence_length=4096)
        prefix = pair["labels"].count(-100)
        features.append({"input_ids": pair["input_ids"][:prefix]})
    active = sorted(set(tokenizer.get_vocab().values()))
    was_training = model.training
    tf32 = torch.backends.cuda.matmul.allow_tf32
    wrapped_forward = model.forward
    try:
        model.eval()
        torch.backends.cuda.matmul.allow_tf32 = False
        # Accelerate installs an inner autocast wrapper during training. Compare
        # the same inference arithmetic without altering the training wrapper.
        if "_original_forward" in model.__dict__:
            model.forward = model.__dict__["_original_forward"]
        inputs = {k: v.to(model.device) for k, v in pad_text_inputs(tokenizer, features, left=True).items()}
        # Match text inference: BF16 base with PEFT's FP32 adapter arithmetic.
        # Autocasting the LoRA matmuls changes numerical behavior with rank.
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            output = model(**inputs, use_cache=False, logits_to_keep=1)
        logits = output.logits[:, -1].detach().float().cpu()
        if not torch.isfinite(logits).all():
            raise RuntimeError("gate produced nonfinite logits")
        return {"logits": logits[:, active], "active_ids": active,
                "vocab_sha256": shared.digest(tokenizer.get_vocab()),
                "greedy_ids": logits.argmax(-1).tolist(),
                "atlas_ids": tokenizer.convert_tokens_to_ids(atlas_tokens()),
                "prompt_ids": [f["input_ids"] for f in features]}
    finally:
        model.forward = wrapped_forward
        model.train(was_training)
        torch.backends.cuda.matmul.allow_tf32 = tf32


def compare_probes(before: dict, after: dict) -> dict:
    a, b = before["logits"], after["logits"]
    maximum = float((a - b).abs().max()) if a.shape == b.shape else None
    progress(f"parity: max_abs={maximum}, before_ids={before['greedy_ids']}, after_ids={after['greedy_ids']}")
    for key in ("active_ids", "vocab_sha256", "greedy_ids", "atlas_ids", "prompt_ids"):
        if before[key] != after[key]:
            raise ValueError(f"gate equivalence failed: {key}; max_abs={maximum}; before={before[key] if key == 'greedy_ids' else 'metadata'}; after={after[key] if key == 'greedy_ids' else 'metadata'}")
    if a.shape != b.shape or not torch.allclose(a, b, atol=ATOL, rtol=RTOL):
        maximum = float((a - b).abs().max()) if a.shape == b.shape else None
        raise ValueError(f"active-vocabulary logits differ: max_abs={maximum}, atol={ATOL}, rtol={RTOL}")
    return {"atol": ATOL, "rtol": RTOL, "max_abs": float((a - b).abs().max()),
            "active_vocab_size": len(before["active_ids"]), "vocab_sha256": before["vocab_sha256"],
            "greedy_ids": before["greedy_ids"], "atlas_ids": before["atlas_ids"],
            "prompt_ids_sha256": shared.digest(before["prompt_ids"])}


def load_eval(plan: dict, checkpoint: str):
    progress(f"loading {checkpoint}")
    torch.manual_seed(44)
    return evaluator.load_model(model_id=BASE, adapter_dir=checkpoint, bits=16,
                                disable_flash_attn2=True, token_inventory=plan["config"]["token_inventory"],
                                preserve_visual_fp32=True, input_mode="text",
                                model_revision=shared.MODEL_REVISION)


def eval_panel(plan: dict, model: Any, tokenizer: Any, evidence: dict,
               checkpoint: str, panel: str, output: Path) -> dict:
    args = argparse.Namespace(
        model_id=BASE, model_revision=shared.MODEL_REVISION, adapter_dir=checkpoint,
        input_mode="text", max_sequence_length=4096, bits=16, batch_size=16,
        long_batch_size=16, max_new_tokens=512, long_max_new_tokens=512,
        candidate_scoring=False, enable_thinking=False, do_sample=False,
        preserve_visual_fp32=True, disable_flash_attn2=True, image_root=None, limit=None,
        token_inventory=plan["config"]["token_inventory"], occlusion_margin=0.03,
    )
    progress(f"greedy512/batch16 evaluation: {panel}")
    evaluator.run_eval_job(model=model, processor=tokenizer, adapter_evidence=evidence, args=args,
                           eval_jsonl=plan["data_dir"] + f"/{panel}.jsonl",
                           image_variant="original", output_dir=output)
    records = rows_at(output / "records.jsonl")
    expected = plan["inputs"]["panels"][panel]
    ids = [r.get("id") for r in records]
    if len(ids) != expected["rows"] or Counter(ids) != Counter(expected["ids"]):
        raise ValueError(f"{panel}: incomplete/duplicate/wrong evaluation IDs")
    by_id = {r.get("id") or r["row_id"]: r for r in rows_at(Path(plan["data_dir"]) / f"{panel}.jsonl")}
    for record in records:
        row = by_id[record["id"]]
        metadata = evaluator.evaluation_metadata(row, image_variant="original")
        metadata["input_mode"] = "text"
        if (record["expected"] != evaluator.expected_text(row) or record["metadata"] != metadata
                or not isinstance(record["response"], str) or record.get("candidate_score") is not None
                or record["score"] != evaluator.score_response(record["expected"], record["response"], metadata=metadata)
                or type(record["score"].get("correct")) is not bool):
            raise ValueError(f"{panel}: incomplete or incorrectly dispatched score")
    summary = shared.read_json(output / "summary.json")
    for key, value in {"model_id": BASE, "model_revision": shared.MODEL_REVISION,
                       "adapter_dir": checkpoint, "input_mode": "text", "bits": 16,
                       "max_sequence_length": 4096, "batch_size": 16, "max_new_tokens": 512,
                       "long_max_new_tokens": 512, "candidate_scoring": False,
                       "reasoning_enabled": False, "rows": expected["rows"], "attempted": expected["rows"]}.items():
        if summary.get(key) != value:
            raise ValueError(f"{panel}: inference condition differs: {key}")
    if summary["correct"] != sum(r["score"]["correct"] for r in records):
        raise ValueError("eval summary disagrees with verified records")
    sft_runs.commit()
    return {"correct": summary["correct"], "rows": summary["rows"],
            "exact_accuracy": summary["exact_accuracy"], "output_dir": str(output),
            "files": shared.file_manifest(output, ["records.jsonl", "summary.json"])}


class CheckpointCallback(TrainerCallback):
    def __init__(self, config: TrainConfig, deadline: float):
        self.config, self.deadline = config, deadline
        self.saved_steps = []

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        check_deadline(self.deadline)
        if state.global_step != 0 or optimizer.state:
            raise RuntimeError("continuation requires a fresh optimizer at step zero")
        scope = shared.read_json(Path(self.config.output_dir) / trainer.TRAINABLE_SCOPE_FILE)
        if scope["errors"]:
            raise RuntimeError("trainer scope audit failed")
        # Discovery must run before PEFT wraps Embedding/Linear. Reuse the
        # trainer's actual pre-wrap component audit instead of rediscovering it.
        self.components = trainer.ModelComponents(**scope["components"])
        groups = Counter(trainer.parameter_category(n, self.components) for n, p in model.named_parameters() if p.requires_grad)
        if set(groups) != ALLOWED or groups["atlas_input_rows"] != 1 or groups["atlas_output_rows"] != 1:
            raise RuntimeError(f"unapproved trainable scope: {groups}")
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        optimized = [id(p) for g in optimizer.param_groups for p in g["params"]]
        if set(optimized) != trainable or len(set(optimized)) != len(optimized):
            raise RuntimeError("optimizer must cover exactly the approved trainable tensors once")
        for group in optimizer.param_groups:
            if not group["params"]:
                continue
            expected = self.config.language_lora_learning_rate if group["catan_name"] == "language_lora" else self.config.learning_rate
            if group.get("initial_lr", group["lr"]) != expected:
                raise RuntimeError("optimizer group learning rate differs")

    def on_step_begin(self, args, state, control, **kwargs):
        check_deadline(self.deadline)

    def on_substep_end(self, args, state, control, **kwargs):
        check_deadline(self.deadline)

    def on_prediction_step(self, args, state, control, **kwargs):
        check_deadline(self.deadline)

    def on_log(self, args, state, control, logs=None, **kwargs):
        check_deadline(self.deadline)
        if logs:
            if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in logs.values()):
                raise RuntimeError("nonfinite training/evaluation metric")
            progress(f"step {state.global_step}: {json.dumps(logs, sort_keys=True)}")

    def on_save(self, args, state, control, **kwargs):
        # Trainer emits on_save after adapter, visual state AND optimizer/state
        # files are complete. Caller callbacks run after its bundle-saving hooks.
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        shared.volume_bundle(str(checkpoint))
        for name in ("optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(f"incomplete Trainer checkpoint: {checkpoint / name}")
        if shared.read_json(checkpoint / "trainer_state.json")["global_step"] != state.global_step:
            raise ValueError("checkpoint global step differs")
        self.saved_steps.append(state.global_step)
        write_json_atomic(Path(self.config.output_dir) / "committed_checkpoints.json",
                          {"steps": self.saved_steps, "latest": str(checkpoint), "at": shared.now()})
        sft_runs.commit()
        progress(f"committed complete Trainer checkpoint {state.global_step}")
        check_deadline(self.deadline)


class GateCallback(CheckpointCallback):
    def __init__(self, config: TrainConfig, deadline: float, rows: list[dict], before: dict):
        super().__init__(config, deadline)
        self.rows, self.before = rows, before
        self.initial, self.frozen, self.losses = {}, {}, []
        self.new_b_gradient = 0.0
        self.group_gradients = dict.fromkeys(ALLOWED, 0.0)
        self.after = None
        self.report = {}

    def on_train_begin(self, args, state, control, model=None, processing_class=None, **kwargs):
        super().on_train_begin(args, state, control, model=model, **kwargs)
        self.tokenizer = processing_class
        self.report["trainer_initial_equivalence"] = compare_probes(self.before, probe(model, self.tokenizer, self.rows))
        self.visual_before = visual_digest(model)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.initial[name] = parameter.detach().cpu().clone()
                if ".lora_B." in name and torch.count_nonzero(parameter[:, 8:]).item():
                    raise RuntimeError("new rank-B columns must begin at zero")
            else:
                self.frozen[name] = (id(parameter), parameter._version)

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        check_deadline(self.deadline)
        for name, parameter in model.named_parameters():
            if name in self.frozen:
                if parameter.requires_grad or parameter.grad is not None or (id(parameter), parameter._version) != self.frozen[name]:
                    raise RuntimeError(f"frozen base/visual tensor changed: {name}")
            elif parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"nonfinite gate gradient: {name}")
                category = trainer.parameter_category(name, self.components)
                self.group_gradients[category] = max(
                    self.group_gradients[category], float(parameter.grad.abs().max()),
                )
                if ".lora_B." in name:
                    self.new_b_gradient = max(self.new_b_gradient, float(parameter.grad[:, 8:].abs().max()))

    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs=logs, **kwargs)
        if logs and "nll_loss" in logs:
            self.losses.append(float(logs["nll_loss"]))

    def on_save(self, args, state, control, model=None, **kwargs):
        super().on_save(args, state, control, **kwargs)
        updates, new_b_update = dict.fromkeys(ALLOWED, 0.0), 0.0
        for name, parameter in model.named_parameters():
            if name in self.frozen:
                if parameter.grad is not None or (id(parameter), parameter._version) != self.frozen[name]:
                    raise RuntimeError(f"frozen tensor changed during gate: {name}")
            else:
                current = parameter.detach().cpu()
                if not torch.isfinite(current).all():
                    raise RuntimeError("nonfinite updated gate parameter")
                delta = (current - self.initial[name]).abs()
                category = trainer.parameter_category(name, self.components)
                updates[category] = max(updates[category], float(delta.max()))
                if ".lora_B." in name:
                    new_b_update = max(new_b_update, float(delta[:, 8:].max()))
        if (state.global_step != 2 or len(self.losses) != 2 or not all(math.isfinite(v) for v in self.losses)
                or not all(v > 0 for v in updates.values())
                or not all(v > 0 for v in self.group_gradients.values())
                or new_b_update <= 0 or self.new_b_gradient <= 0):
            raise RuntimeError(f"gate did not learn in all approved groups: updates={updates}, new_B={new_b_update}")
        self.after = probe(model, self.tokenizer, self.rows)
        visual_after = visual_digest(model)
        if self.visual_before != visual_after:
            raise ValueError("visual tensors changed during the two training steps")
        self.report.update(losses=self.losses, max_abs_update=updates,
                            max_abs_gradient=self.group_gradients,
                           new_rank_b_max_gradient=self.new_b_gradient,
                           new_rank_b_max_update=new_b_update, visual_tensor_sha256=visual_after,
                           frozen_tensors_checked=len(self.frozen), added_rank_a_gradient_required=False)
        write_json_atomic(Path(self.config.output_dir) / "gate_updates.json", self.report)
        sft_runs.commit()


def prepared_for(plan: dict) -> dict:
    receipt = shared.read_json(Path(plan["root"]) / "prepare/result.json")
    if receipt["status"] != "completed" or receipt["launch_sha256"] != shared.digest(plan):
        raise ValueError("CPU preparation did not complete for this launch")
    prepared = receipt["result"]
    check_manifest(Path(PARENT), prepared["parent_files"])
    check_manifest(Path(plan["config"]["initial_bundle"]), prepared["expanded_files"])
    check_manifest(Path(BASE), prepared["base_files"])
    if sha256_file(Path(plan["config"]["eval_jsonl"])) != prepared["teacher_sha256"]:
        raise ValueError("fixed teacher-forced panel changed")
    return prepared


def retained_prevalidation(plan: dict, prepared: dict) -> dict:
    """Reuse verified completed predictions without claiming a complete baseline."""
    reference = plan["prevalidation_reference"]
    root = Path(reference["root"])
    previous = shared.read_json(root / "launch.json")
    if previous["parent"] != PARENT or previous["base"] != BASE:
        raise ValueError("retained baseline checkpoint differs")
    if previous["inputs"]["files"]["validation_eval.jsonl"] != plan["inputs"]["files"]["validation_eval.jsonl"]:
        raise ValueError("retained baseline input differs")
    for name in ("sft/scripts/eval_qwen_vl_adapter.py", "sft/scripts/train_trl_catan_vision.py", "sft/board_fluency_scoring.py"):
        if previous["source_sha256"][name] != plan["source_sha256"][name]:
            raise ValueError(f"retained baseline inference/scoring code changed: {name}")
    old_preparation = shared.read_json(root / "prepare/result.json")
    if old_preparation["status"] != "completed":
        raise ValueError("retained baseline preparation was incomplete")
    old_files = old_preparation["result"]["expanded_files"]
    for name in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "chat_template.jinja", trainer.VISUAL_STATE_FILE):
        if old_files[name] != prepared["expanded_files"][name]:
            raise ValueError(f"retained baseline initialization changed: {name}")
    if shared.read_json(root / "gate/equivalence.json")["max_abs"] != 0.0:
        raise ValueError("retained baseline lacks exact parent parity evidence")
    source = root / "gate/pre-validation190/records.jsonl"
    if sha256_file(source) != reference["records_sha256"]:
        raise ValueError("retained raw predictions changed")
    records = rows_at(source)
    ids = [row["id"] for row in records]
    expected = {row["id"]: row for row in rows_at(Path(plan["data_dir"]) / "validation_eval.jsonl")}
    if len(ids) != reference["completed_rows"] or len(set(ids)) != len(ids) or not set(ids) <= set(expected):
        raise ValueError("retained baseline ID coverage differs")
    for record in records:
        row = expected[record["id"]]
        metadata = evaluator.evaluation_metadata(row, image_variant="original")
        metadata["input_mode"] = "text"
        if (record["expected"] != evaluator.expected_text(row) or record["metadata"] != metadata
                or record["score"] != evaluator.score_response(record["expected"], record["response"], metadata=metadata)):
            raise ValueError("retained baseline score/metadata differs")
    output = Path(plan["root"]) / "gate/pre-validation190"
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


def gate_work(plan: dict, deadline: float) -> dict:
    prepared = prepared_for(plan)
    config = replace(TrainConfig(**plan["config"]), max_steps=2, save_steps=2, eval_steps=2,
                     eval_jsonl=None, output_dir=plan["root"] + "/gate/training")
    probe_rows = rows_at(Path(plan["config"]["train_jsonl"]))[:4]
    model, tokenizer, evidence = load_eval(plan, PARENT)
    parent_probe = probe(model, tokenizer, probe_rows)
    parent_visual = visual_digest(model)
    del model, tokenizer, evidence
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer, evidence = load_eval(plan, config.initial_bundle)
    expanded_probe = probe(model, tokenizer, probe_rows)
    save_file({"parent": parent_probe["logits"], "expanded": expanded_probe["logits"]},
              Path(plan["root"]) / "gate/parity_logits.safetensors")
    write_json_atomic(Path(plan["root"]) / "gate/parity_metadata.json", {
        "parent": {k: v for k, v in parent_probe.items() if k != "logits"},
        "expanded": {k: v for k, v in expanded_probe.items() if k != "logits"},
        "precision": "text inference: no autocast; BF16 base, FP32 LoRA",
    })
    sft_runs.commit()
    equivalence = compare_probes(parent_probe, expanded_probe)
    if visual_digest(model) != parent_visual:
        raise ValueError("expanded model changed the preserved visual tensor hash")
    write_json_atomic(Path(plan["root"]) / "gate/equivalence.json", equivalence)
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
    check_manifest(Path(config.initial_bundle), prepared["expanded_files"])
    return {"equivalence": equivalence, "updates": callback.report,
             "reload_equivalence": reload_equivalence, "pre_validation190": baseline,
             "training": training, "gate_output_is_main_initialization": False,
             "frozen_visual_file_tensor_sha256": frozen_visual}


def train_work(plan: dict, deadline: float) -> dict:
    prepared = prepared_for(plan)
    gate = shared.read_json(Path(plan["root"]) / "gate/result.json")
    if gate["status"] != "completed" or gate["launch_sha256"] != shared.digest(plan):
        raise ValueError("real GPU gate must complete before main training")
    config = TrainConfig(**plan["config"])
    if Path(config.output_dir).exists():
        raise FileExistsError("training output exists; resumption/retry is not authorized")
    progress("training: fresh optimizer from immutable expanded-r16, 128 steps / effective batch8")
    callback = CheckpointCallback(config, deadline)
    result = trainer.run_training(config, extra_callbacks=[callback])
    if result["status"] != "completed" or callback.saved_steps != [32, 64, 96, 128]:
        raise RuntimeError("main training did not commit exactly checkpoints 32/64/96/128")
    final = Path(config.output_dir) / "checkpoints/checkpoint-128"
    frozen_visual = visual_file_digest(Path(PARENT) / trainer.VISUAL_STATE_FILE)
    if visual_file_digest(final / trainer.VISUAL_STATE_FILE) != frozen_visual:
        raise ValueError("main training changed frozen visual state")
    check_manifest(Path(config.initial_bundle), prepared["expanded_files"])
    return {"training": result, "final_checkpoint": str(final), "committed_steps": callback.saved_steps,
            "frozen_visual_file_tensor_sha256": frozen_visual}


def posteval_work(plan: dict, deadline: float) -> dict:
    prepared_for(plan)
    training = shared.read_json(Path(plan["root"]) / "train/result.json")
    if training["status"] != "completed" or training["launch_sha256"] != shared.digest(plan):
        raise ValueError("main training must complete before post-evaluation")
    checkpoint = str(Path(plan["config"]["output_dir"]) / "checkpoints/checkpoint-128")
    if training["result"]["final_checkpoint"] != checkpoint:
        raise ValueError("post-evaluation checkpoint differs")
    model, tokenizer, evidence = load_eval(plan, checkpoint)
    panels = {}
    for panel in ("review", "validation_eval"):
        check_deadline(deadline)
        panels[panel] = eval_panel(plan, model, tokenizer, evidence, checkpoint, panel,
                                   Path(plan["root"]) / "posteval" / panel)
    return {"checkpoint": checkpoint, "panels": panels}


def worker(plan: dict, stage: str, deadline: float) -> None:
    os.environ.update(OFFLINE)
    torch.set_num_threads(4 if stage == "prepare" else 16)
    directory = Path(plan["root"]) / stage
    receipt = {"status": "running", "stage": stage, "launch_sha256": shared.digest(plan),
               "started_at": shared.now(), "inner_deadline_unix": deadline}
    write_json_atomic(directory / "result.json", receipt)
    sft_runs.commit()
    try:
        with deadline_alarm(deadline):
            verify_plan(plan)
            receipt["dependencies"] = trainer.assert_runtime_versions()
            if stage != "prepare":
                verify_uploaded(plan)
            function = {"prepare": prepare_work, "gate": gate_work,
                        "train": train_work, "posteval": posteval_work}[stage]
            receipt.update(result=function(plan, deadline), status="completed")
    except BaseException as exc:
        receipt.update(status="failed", error=shared.error_record(exc))
        progress(f"{stage} failed: {receipt['error']}")
        raise
    finally:
        receipt["ended_at"] = shared.now()
        write_json_atomic(directory / "result.json", receipt)
        sft_runs.commit()


def bounded_stage(plan: dict, stage: str, deadline: float) -> dict:
    """Hard subprocess limit backs up the cooperative alarm/callback deadlines."""
    started = time.time()
    hard_deadline = min(deadline - 30, started + STAGE_SECONDS[stage] - 30)
    inner_deadline = hard_deadline - 45
    check_deadline(inner_deadline)
    for volume in VOLUMES.values():
        volume.reload()
    directory = Path(plan["root"]) / stage
    # Delivery/startup failures cannot silently reuse a paid stage's work.
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "sft.modal_board_fluency_sft", "--worker", stage,
               "--plan", plan["root"] + "/launch.json", "--deadline", str(inner_deadline)]
    write_json_atomic(directory / "wrapper.json", {"status": "running", "command": command,
                      "call_id": modal.current_function_call_id(), "hard_deadline_unix": hard_deadline})
    sft_runs.commit()
    process = None
    relay = None
    error = None
    # An open file on /runs can prevent Volume.commit's reload. Keep the live
    # tee on the container disk and copy it after the worker closes its handles.
    log_path = Path("/tmp") / f"board-fluency-{plan['reservation_id']}-{stage}.log"
    try:
        progress(f"{stage}: starting bounded worker, {int(inner_deadline - time.time())}s inner allowance")
        with log_path.open("x", buffering=1) as log:
            process = subprocess.Popen(command, env={**os.environ, **OFFLINE}, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)

            def forward_logs():
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)

            relay = threading.Thread(target=forward_logs, daemon=True)
            relay.start()
            try:
                code = process.wait(timeout=max(0.001, hard_deadline - time.time()))
                if code:
                    raise RuntimeError(f"{stage} worker exited {code}; see {directory}/worker.log")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                relay.join(timeout=5)
        result = shared.read_json(directory / "result.json")
        if result["status"] != "completed":
            raise RuntimeError(f"{stage} failed: {result.get('error')}")
        return result
    except BaseException as exc:
        error = shared.error_record(exc)
        raise
    finally:
        if log_path.is_file():
            shutil.copyfile(log_path, directory / "worker.log")
        write_json_atomic(directory / "wrapper.json", {"status": "failed" if error else "completed",
                          "error": error, "call_id": modal.current_function_call_id(),
                          "hard_deadline_unix": hard_deadline, "ended_at": shared.now()})
        sft_runs.commit()


@app.function(**COMMON, cpu=(4.0, 4.0), memory=(16 * 1024, 16 * 1024), timeout=1200)
def prepare_cpu(plan: dict, deadline: float) -> dict:
    return bounded_stage(plan, "prepare", deadline)


@app.function(**GPU, timeout=STAGE_SECONDS["gate"])
def gate_h200(plan: dict, deadline: float) -> dict:
    return bounded_stage(plan, "gate", deadline)


@app.function(**GPU, timeout=STAGE_SECONDS["train"])
def train_h200(plan: dict, deadline: float) -> dict:
    return bounded_stage(plan, "train", deadline)


@app.function(**GPU, timeout=1800)
def posteval_h200(plan: dict, deadline: float) -> dict:
    return bounded_stage(plan, "posteval", deadline)


@app.function(**COMMON, cpu=(CONTROL_CPU, CONTROL_CPU), memory=(2048, 2048), timeout=300)
def reserve(plan: dict) -> None:
    sft_runs.reload()
    root = Path(plan["root"])
    root.mkdir(parents=True, exist_ok=False)
    write_json_atomic(root / "launch.json", plan)
    write_json_atomic(root / "budget_guard.json", plan["budget"])
    sft_runs.commit()


@app.function(**COMMON, cpu=(CONTROL_CPU, CONTROL_CPU), memory=(2048, 2048), timeout=COORDINATOR_SECONDS)
def coordinate(plan: dict, absolute_deadline: float) -> dict:
    with deadline_alarm(absolute_deadline - 30):
        return coordinate_work(plan, absolute_deadline)


def coordinate_work(plan: dict, absolute_deadline: float) -> dict:
    sft_runs.reload()
    root = Path(plan["root"])
    if shared.read_json(root / "launch.json") != plan or (root / "coordinator.json").exists():
        raise ValueError("reservation differs or coordinator already used")
    if not 0 < absolute_deadline - time.time() <= ABSOLUTE_SECONDS:
        raise ValueError("invalid absolute coordinator deadline")
    result = {"status": "running", "started_at": shared.now(), "stages": {},
              "coordinator_call_id": modal.current_function_call_id(),
              "absolute_deadline_unix": absolute_deadline, "budget": plan["budget"]}
    active = None
    phase = None

    def persist():
        # A sibling worker commits checkpoints; reload before publishing control
        # receipts so this low-CPU process never obscures a completed save.
        sft_runs.reload()
        write_json_atomic(root / "coordinator.json", result)
        sft_runs.commit()

    persist()
    try:
        verify_plan(plan)
        for phase, function in (("prepare", prepare_cpu), ("gate", gate_h200),
                                ("train", train_h200), ("posteval", posteval_h200)):
            stage_deadline = min(absolute_deadline - 30, time.time() + STAGE_SECONDS[phase] + STARTUP)
            check_deadline(stage_deadline)
            entry = result["stages"][phase] = {"status": "starting", "started_at": shared.now(),
                                             "deadline_unix": stage_deadline}
            persist()
            with deadline_alarm(stage_deadline):
                active = function.spawn(plan, stage_deadline)
                entry.update(status="running", call_id=active.object_id)
                persist()  # The child ID is durable before waiting on ANY stage.
                progress(f"coordinator: {phase} {active.object_id}, deadline={stage_deadline}")
                while True:
                    remaining = min(stage_deadline, absolute_deadline - 30) - time.time()
                    if remaining <= 0:
                        raise TimeoutError(f"{phase} absolute wait deadline; cancelling child including startup loops")
                    try:
                        value = active.get(timeout=min(30, remaining))
                        break
                    except TimeoutError as polling_timeout:
                        # FunctionCall.get's polling timeout is the built-in
                        # exception, distinct from a remote function timeout.
                        if str(polling_timeout):
                            raise
                        progress(f"coordinator: {phase} still running; {int(remaining)}s left")
            if value["status"] != "completed":
                raise RuntimeError(f"{phase} failed: {value.get('error')}")
            entry.update(status="completed", ended_at=shared.now(), result=value)
            persist()
            active = None
            # Bound the preceding container's scaledown before the next GPU.
            if phase in ("gate", "train"):
                time.sleep(2)
        pre = result["stages"]["gate"]["result"]["result"]["pre_validation190"]
        post = result["stages"]["posteval"]["result"]["result"]["panels"]
        matched = [row for row in rows_at(Path(post["validation_eval"]["output_dir"]) / "records.jsonl")
                   if row["id"] in set(pre["ids"])]
        if len(matched) != pre["rows"]:
            raise ValueError("post-evaluation does not cover the retained baseline IDs")
        result.update(status="completed", comparison={
            "validation190": {"after": post["validation_eval"]["correct"], "rows": 190,
                              "baseline_complete": False},
            "validation_matched": {"before": pre["correct"], "after": sum(row["score"]["correct"] for row in matched),
                                   "rows": pre["rows"], "baseline_partial": True},
            "unchanged_review200": {"saved_before": 24, "after": post["review"]["correct"], "rows": 200}})
    except BaseException as exc:
        result.update(status="failed", error=shared.error_record(exc))
        if phase in result["stages"]:
            result["stages"][phase].update(status="failed", error=result["error"])
        if active is not None:
            try:
                active.cancel(terminate_containers=True)
                result["cancelled_call_id"] = active.object_id
            except BaseException as cancellation:
                result["cancellation_error"] = shared.error_record(cancellation)
        raise
    finally:
        result["ended_at"] = shared.now()
        persist()
    return result


def stop_app(app_id: str) -> None:
    """Stop the whole dedicated app, covering coordinator/child spawn races."""
    if not isinstance(app_id, str) or not app_id.startswith("ap-"):
        raise ValueError("a recorded Modal app ID is required to stop this run")
    subprocess.run([sys.executable, "-m", "modal", "app", "stop", app_id],
                   check=True, timeout=60)


def stop_run(run_name: str) -> dict:
    configuration(run_name)  # Validate the exact run-name/path boundary.
    if os.environ.get("MODAL_PROFILE") != "icebear5h":
        raise ValueError("stop requires MODAL_PROFILE=icebear5h")
    path = PROJECT_ROOT / "artifacts/runs/sft" / run_name / "orchestration.json"
    state = shared.read_json(path)
    stop_app(state["app_id"])
    state.update(status="app_stop_requested", stop_requested_at=shared.now())
    write_json_atomic(path, state)
    return state


def launch(run_name: str, budget_usd: float = 15, execute: bool = False) -> dict:
    plan = build_plan(run_name, budget_usd)
    if not execute:
        return {"dry_run": True, "remote_calls": False, "plan": plan,
                "deferred_to_cpu": "cached parent/base, token boundaries, immutable rank expansion",
                "deferred_to_real_gate": "numerical equivalence, actual gradients/updates, reload, validation190"}
    if os.environ.get("MODAL_PROFILE") != "icebear5h" or HF_SECRET_NAME != "huggingface-secret-2":
        raise ValueError("execute requires MODAL_PROFILE=icebear5h CATAN_HF_SECRET_NAME=huggingface-secret-2")
    local = PROJECT_ROOT / "artifacts/runs/sft" / run_name
    local.mkdir(parents=True, exist_ok=False)
    write_json_atomic(local / "launch.json", plan)
    state = {"status": "uploading", "launch_sha256": shared.digest(plan)}
    app_id = None
    write_json_atomic(local / "orchestration.json", state)
    try:
        with app.run(detach=True):
            app_id = app.app_id
            state["app_id"] = app_id
            write_json_atomic(local / "orchestration.json", state)
            reservation = reserve.spawn(plan)
            try:
                state["reservation_call_id"] = reservation.object_id
                write_json_atomic(local / "orchestration.json", state)
                reservation.get(timeout=600)
            except BaseException:
                reservation.cancel(terminate_containers=True)
                raise
            remote = plan["data_dir"].removeprefix("/data")
            with sft_data.batch_upload(force=False) as batch:
                for name in DATA_FILES:
                    payload = (DATASET / name).read_bytes()
                    if hashlib.sha256(payload).hexdigest() != plan["inputs"]["files"][name]["sha256"]:
                        raise ValueError("local inputs changed after admission")
                    batch.put_file(io.BytesIO(payload), remote + "/" + name)
                review_bytes = REVIEW.read_bytes()
                if hashlib.sha256(review_bytes).hexdigest() != REVIEW_SHA256:
                    raise ValueError("review changed after admission")
                batch.put_file(io.BytesIO(review_bytes), remote + "/review.jsonl")
                batch.put_file(local / "launch.json", remote + "/launch.json")
            absolute_deadline = time.time() + ABSOLUTE_SECONDS
            call = coordinate.spawn(plan, absolute_deadline)
            state.update(status="spawned", coordinator_call_id=call.object_id,
                         absolute_deadline_unix=absolute_deadline,
                         remote_receipt=plan["root"] + "/coordinator.json")
            write_json_atomic(local / "orchestration.json", state)
    except BaseException as exc:
        state.update(status="launch_failed", error=shared.error_record(exc))
        if app_id is not None:
            try:
                stop_app(app_id)
                state["whole_app_stop_requested"] = True
            except BaseException as stop_error:
                state["stop_error"] = shared.error_record(stop_error)
        write_json_atomic(local / "orchestration.json", state)
        raise
    return {**state, "local_receipt": str(local / "orchestration.json"),
             "stop": f"MODAL_PROFILE=icebear5h {sys.executable} -m modal app stop {app_id}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name")
    parser.add_argument("--budget-usd", type=float, default=15)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop", action="store_true", help="Stop the recorded run's entire Modal app")
    parser.add_argument("--worker", choices=tuple(STAGE_SECONDS), help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        # A subprocess has no Modal ContainerIOManager singleton; is_local()
        # therefore returns True even inside the mounted remote container.
        if os.environ.get("MODAL_IS_REMOTE") != "1" or args.plan is None or args.deadline is None:
            parser.error("internal worker requires a remote container, plan and deadline")
        worker(shared.read_json(args.plan), args.worker, args.deadline)
    else:
        if not args.run_name:
            parser.error("--run-name is required")
        if args.stop and args.execute:
            parser.error("--stop and --execute are mutually exclusive")
        result = stop_run(args.run_name) if args.stop else launch(args.run_name, args.budget_usd, args.execute)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
