"""Bounded two-step O-LoRA initialization/checkpoint/reload smoke on one H200."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path, PurePosixPath

import modal
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_str, load_json_dict
from sft.launchers._json import at_str
from sft.launchers._train_config import train_config
from sft.launchers.modal_catan_vision_sft import (
    hf_cache,
    sft_data,
    sft_runs,
    training_image,
    upload_training_bundle,
)
from sft.scripts.train.train_trl_catan_vision import (
    MODEL_ID,
    PROFILE_OLORA_FROZEN_BUNDLE,
    TrainConfig,
    iter_jsonl,
    run_training,
    sha256_file,
    write_json_atomic,
)

app = modal.App("catan-olora-smoke")
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PARENT = "/runs/catan-vision-sft/catan-qwen38-gauss-s2-terrain-20260904/398f0a023ec9/checkpoints/checkpoint-384"
ROOT = Path("artifacts/generated/board_recognition/replay_v1")
FACTORS = Path("artifacts/diagnostics/sft/visual_delta_gauss_s2_ck384_20260905/derived/visual_delta_rank16_linear.safetensors")
VOLUMES: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {"/cache": hf_cache, "/data": sft_data, "/runs": sft_runs}


@app.function(
    image=training_image, cpu=(4.0, 4.0), memory=(8192, 8192),
    timeout=1200, startup_timeout=300, retries=0, max_containers=1,
    scaledown_window=2, volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface-secret-2", required_keys=["HF_TOKEN"])],
)
def prepare() -> dict[str, str]:
    """Populate the new workspace's cache on CPU before requesting a GPU."""
    required = ("adapter_config.json", "adapter_model.safetensors", "visual_model.safetensors", "trainable_parameters.json", "training_config.json")
    for name in required:
        if not (Path(PARENT) / name).is_file():
            raise FileNotFoundError(Path(PARENT) / name)
    snapshot_download(MODEL_ID, revision=REVISION)
    snapshot = f"/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/{REVISION}"
    if not Path(snapshot).is_dir():
        raise FileNotFoundError(snapshot)
    hf_cache.commit()
    return {"snapshot": snapshot, "revision": REVISION, "parent": PARENT}


@app.function(
    image=training_image, gpu="H200", cpu=(16.0, 16.0),
    memory=(131072, 131072), timeout=900, startup_timeout=300,
    retries=0, max_containers=1, scaledown_window=2, volumes=VOLUMES,
)
def smoke(payload: JsonDict) -> JsonLikeDict:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(16)
    config = train_config(payload)
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    result: JsonLikeDict = {"status": "running", "config": payload, "timeout_seconds": 900}
    write_json_atomic(output / "smoke.json", result)
    sft_runs.commit()
    try:
        if sha256_file(Path(PARENT) / "visual_model.safetensors") != "94534c119727435bc2011fb76a6038c1e71facd120abbf482e71bd80a2d252f2":
            raise RuntimeError("parent visual checkpoint hash differs from extraction provenance")
        if config.visual_delta_factors is None:
            raise RuntimeError("the O-LoRA smoke requires visual_delta_factors")
        if sha256_file(Path(config.visual_delta_factors)) != "9dff0339e8deb54d0ced0af3a05f523fab901cbc369d925644d5f82476f1ae73":
            raise RuntimeError("visual protection factors differ from extraction provenance")
        result["training"] = run_training(config)
        frozen = load_json_dict(output / "frozen_bundle.json")
        if frozen["unprotected_modules"]:
            raise RuntimeError("unprotected O-LoRA modules")
        if as_dict(frozen["new_adapter"])["vision_targets"] != 110:
            raise RuntimeError("unexpected visual target count")
        # Check that every frozen visual parameter survived the optimizer steps
        # bit-for-bit; periodic checkpoints preserve full FP32 state.
        parent = load_file(str(Path(PARENT) / "visual_model.safetensors"))
        checkpoint = load_file(str(output / "checkpoints/checkpoint-2/visual_model.safetensors"))
        canonical = {key.replace(".base_layer", ""): value for key, value in checkpoint.items() if ".lora_" not in key}
        if set(parent) != set(canonical):
            raise RuntimeError("frozen visual coverage changed")
        for key, value in parent.items():
            if canonical[key].dtype != torch.float32 or not torch.equal(value, canonical[key]):
                raise RuntimeError(f"frozen visual parameter changed: {key}")
        first = load_file(str(output / "checkpoints/checkpoint-1/adapter_model.safetensors"))
        second = load_file(str(output / "checkpoints/checkpoint-2/adapter_model.safetensors"))
        if set(first) != set(second):
            raise RuntimeError("adapter checkpoint coverage changed")
        updates = {"vision_lora": 0, "language_lora": 0, "token_rows": 0}
        for key, value in second.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"nonfinite adapter tensor: {key}")
            if not torch.equal(first[key], value):
                group = "token_rows" if "trainable_tokens_delta" in key else (
                    "vision_lora" if ".visual." in key else "language_lora")
                updates[group] += 1
        if not all(updates.values()):
            raise RuntimeError(f"missing effective optimizer updates: {updates}")
        result.update(status="completed", frozen_visual_tensors_verified=len(parent),
                      changed_tensors_between_steps=updates,
                      parent_visual_sha256=sha256_file(Path(PARENT) / "visual_model.safetensors"))
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write_json_atomic(output / "smoke.json", result)
        sft_runs.commit()
    return result


@app.local_entrypoint()
def main(
    execute: bool = False,
    prepare_only: bool = False,
    run_name: str = "olora-smoke-20260907",
) -> None:
    """Default is read-only planning; execute never starts full training."""
    if not execute:
        print(json.dumps({"parent": PARENT, "revision": REVISION, "steps": 2,
                          "gpu": "H200", "gpu_timeout_seconds": 900,
                          "cpu_prepare_timeout_seconds": 1200, "run_name": run_name}))
        return
    prepared = prepare.remote()
    print(json.dumps({"prepared": prepared}), flush=True)
    if prepare_only:
        return
    output = f"/runs/catan-vision-sft/{run_name}"
    # Select four short training rows, including occupied pieces and terrain.
    chosen: list[JsonDict] = []
    families: set[str] = set()
    for _, row in iter_jsonl(ROOT / "mixed_rung3b_v1/stage1/train.jsonl"):
        answer = at_str(row, "messages", -1, "content")
        family = as_str(row.get("task", row.get("task_type", "")))
        if len(answer) < 50 and family not in families:
            chosen.append(row)
            families.add(family)
        if len(chosen) == 4:
            break
    if len(chosen) != 4:
        raise RuntimeError(f"expected four short task families, found {families}")
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "train.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in chosen))
        train, _, images, tokens, contract, _ = upload_training_bundle(
            source, ROOT / "mixed_rung3b_v1/images",
            ROOT / "ms_swift_bidirectional_v1/trainable_tokens.json",
            remote_dir=f"catan-vision-sft/datasets/{run_name}", require_curriculum=False,
            extra_files={"visual_delta_factors.safetensors": FACTORS},
        )
    config = TrainConfig(
        train_jsonl=train, image_root=images, token_inventory=tokens,
        output_dir=output, model_id=prepared["snapshot"],
        profile=PROFILE_OLORA_FROZEN_BUNDLE, frozen_bundle=PARENT,
        visual_delta_factors=f"/data/catan-vision-sft/datasets/{run_name}/visual_delta_factors.safetensors",
        token_init="keep", lora_rank=16, lora_alpha=32,
        max_steps=2, per_device_train_batch_size=2, gradient_accumulation_steps=1,
        save_steps=1, save_total_limit=2, dataloader_num_workers=0,
        require_curriculum=False,
    )
    config.validate()
    receipt = Path("artifacts/runs/sft") / run_name / "launch.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    call = smoke.spawn(asdict(config))
    write_json_atomic(receipt, {"call_id": call.object_id, "output": output,
                               "config": asdict(config), "dataset": contract,
                               "source_sha256": {str(path): sha256_file(path) for path in (
                                    Path(__file__), Path("sft/scripts/train/train_trl_catan_vision/__init__.py"),
                                    Path("sft/scripts/train/train_trl_catan_vision/_common.py"),
                                    Path("sft/scripts/train/train_trl_catan_vision/_config.py"),
                                    Path("sft/scripts/train/train_trl_catan_vision/_text_data.py"),
                                    Path("sft/scripts/train/train_trl_catan_vision/_vision_data.py"),
                                    Path("sft/scripts/train/train_trl_catan_vision/_datasets.py"),
                                    Path("sft/scripts/train/train_trl_catan_vision/_visual.py"),
                                    Path("sft/scripts/train/train_trl_catan_vision/_structure.py"),
                                    Path("sft/scripts/train/train_trl_catan_vision/_model_tokens.py"),
                                    Path("sft/scripts/train/train_trl_catan_vision/_frozen.py"),
                                    Path("sft/scripts/train/train_trl_catan_vision/_bundles.py"),
                                    Path("sft/scripts/train/train_trl_catan_vision/_optim.py"),
                                    Path("sft/scripts/train/train_trl_catan_vision/_trainer.py"),
                                    Path("sft/scripts/train/train_trl_catan_vision/_sft.py"),
                                    Path("sft/scripts/train/train_trl_catan_vision/_run.py"))}})
    print(receipt.read_text(), flush=True)
