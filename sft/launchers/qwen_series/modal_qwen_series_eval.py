"""Qwen-VL Catan adapter generation eval launcher in ``sft.launchers``."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sft.json_types import JsonDict, load_json_dict
from sft.launchers.qwen_series._eval_support import (
    REMOTE_CACHE_MOUNT,
    REMOTE_DATA_MOUNT,
    REMOTE_RUNS_MOUNT,
    EvalKwargs,
    app,
    eval_image,
    hf_cache,
    sft_data,
    sft_runs,
    upload_eval_jsonl,
)


def _run_eval(
    eval_jsonl: str,
    output_dir: str,
    adapter_dir: str | None = None,
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    limit: int | None = None,
    max_new_tokens: int = 16,
    bits: int = 4,
    token_inventory: str | None = None,
    batch_size: int = 1,
    image_variant: str = "original",
    occlusion_margin: float = 0.03,
    candidate_scoring: bool = True,
    long_max_new_tokens: int = 512,
    long_batch_size: int = 8,
    preserve_visual_fp32: bool = False,
) -> JsonDict:
    command = [
        "python",
        "-m",
        "sft.scripts.eval.eval_qwen_vl_adapter",
        "--eval-jsonl",
        *eval_jsonl.split(","),
        "--output-dir",
        output_dir,
        "--model-id",
        model_id,
        "--max-new-tokens",
        str(max_new_tokens),
        "--bits",
        str(bits),
        "--batch-size",
        str(batch_size),
        "--image-variant",
        image_variant,
        "--occlusion-margin",
        str(occlusion_margin),
        "--long-max-new-tokens",
        str(long_max_new_tokens),
        "--long-batch-size",
        str(long_batch_size),
    ]
    if adapter_dir:
        command.extend(["--adapter-dir", adapter_dir])
    if token_inventory:
        command.extend(["--token-inventory", token_inventory])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if not candidate_scoring:
        command.append("--no-candidate-scoring")
    if preserve_visual_fp32:
        command.append("--preserve-visual-fp32")

    print("running:", " ".join(command))
    subprocess.run(command, check=True)
    sft_runs.commit()

    for name in ("batch_summary.json", "summary.json"):
        summary_path = Path(output_dir) / name
        if summary_path.exists():
            return load_json_dict(summary_path)
    return {"output_dir": output_dir}


@app.function(
    image=eval_image,
    gpu="L40S",
    volumes={
        REMOTE_CACHE_MOUNT: hf_cache,
        REMOTE_DATA_MOUNT: sft_data,
        REMOTE_RUNS_MOUNT: sft_runs,
    },
    timeout=60 * 60 * 8,
)
def eval_remote(
    eval_jsonl: str,
    output_dir: str,
    adapter_dir: str | None = None,
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    limit: int | None = None,
    max_new_tokens: int = 16,
    bits: int = 4,
    token_inventory: str | None = None,
    batch_size: int = 1,
    image_variant: str = "original",
    occlusion_margin: float = 0.03,
    candidate_scoring: bool = True,
    long_max_new_tokens: int = 512,
    long_batch_size: int = 8,
    preserve_visual_fp32: bool = False,
) -> JsonDict:
    return _run_eval(
        eval_jsonl,
        output_dir,
        adapter_dir,
        model_id,
        limit,
        max_new_tokens,
        bits,
        token_inventory,
        batch_size,
        image_variant,
        occlusion_margin,
        candidate_scoring,
        long_max_new_tokens,
        long_batch_size,
        preserve_visual_fp32=preserve_visual_fp32,
    )


@app.function(
    image=eval_image,
    gpu="H200",
    volumes={
        REMOTE_CACHE_MOUNT: hf_cache,
        REMOTE_DATA_MOUNT: sft_data,
        REMOTE_RUNS_MOUNT: sft_runs,
    },
    timeout=60 * 60 * 8,
)
def eval_h200(
    eval_jsonl: str,
    output_dir: str,
    adapter_dir: str | None = None,
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    limit: int | None = None,
    max_new_tokens: int = 16,
    bits: int = 16,
    token_inventory: str | None = None,
    batch_size: int = 48,
    image_variant: str = "original",
    occlusion_margin: float = 0.03,
    candidate_scoring: bool = True,
    long_max_new_tokens: int = 512,
    long_batch_size: int = 8,
    preserve_visual_fp32: bool = False,
) -> JsonDict:
    return _run_eval(
        eval_jsonl,
        output_dir,
        adapter_dir,
        model_id,
        limit,
        max_new_tokens,
        bits,
        token_inventory,
        batch_size,
        image_variant,
        occlusion_margin,
        candidate_scoring,
        long_max_new_tokens,
        long_batch_size,
        preserve_visual_fp32=preserve_visual_fp32,
    )


@app.local_entrypoint()
def main(
    eval_jsonl: str,
    image_root: str | None = None,
    token_inventory: str | None = None,
    remote_dir: str = "catan-qwen-series-eval/heldout",
    output_dir: str = f"{REMOTE_RUNS_MOUNT}/qwen-series-eval",
    adapter_dir: str | None = None,
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    limit: int | None = None,
    max_new_tokens: int = 16,
    bits: int = 4,
    batch_size: int = 1,
    gpu: str = "l40s",
    image_variant: str = "original",
    occlusion_margin: float = 0.03,
    candidate_scoring: bool = True,
    long_max_new_tokens: int = 512,
    long_batch_size: int = 8,
    spawn_eval: bool = False,
    preserve_visual_fp32: bool = False,
) -> None:
    if bits not in {4, 8, 16}:
        raise ValueError("--bits must be 4, 8, or 16")
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if gpu not in {"l40s", "h200"}:
        raise ValueError("--gpu must be l40s or h200")
    for variant in image_variant.split(","):
        if variant.strip() not in {
            "original",
            "blank",
            "shuffle",
            "target_occlusion",
            "control_occlusion",
        }:
            raise ValueError(f"unsupported --image-variant: {variant}")
    # Comma-separated eval sets share one image root and token inventory and are
    # scored in one container with a single model load.
    remote_sets = []
    remote_token_inventory = None
    local_sets = [item.strip() for item in eval_jsonl.split(",")]
    roots = [item.strip() for item in image_root.split(",")] if image_root else [None] * len(local_sets)
    if len(roots) == 1 and len(local_sets) > 1:
        roots = roots * len(local_sets)
    if len(roots) != len(local_sets):
        raise ValueError("--image-root must be one root or one root per eval set")
    for index, (local_eval, local_root) in enumerate(zip(local_sets, roots, strict=True)):
        local_path = Path(local_eval)
        set_stem = "-".join(part for part in local_path.parts[-3:] if part).removesuffix(".jsonl")
        set_dir = remote_dir if index == 0 else f"{remote_dir}/{set_stem}"
        remote_set, remote_inventory = upload_eval_jsonl(
            local_path,
            set_dir,
            eval_set_id=set_stem,
            image_root=Path(local_root) if local_root else None,
            token_inventory=Path(token_inventory) if token_inventory else None,
        )
        remote_sets.append(remote_set)
        remote_token_inventory = remote_token_inventory or remote_inventory
    kwargs: EvalKwargs = {
        "eval_jsonl": ",".join(remote_sets),
        "output_dir": output_dir,
        "adapter_dir": adapter_dir,
        "model_id": model_id,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "bits": bits,
        "token_inventory": remote_token_inventory,
        "batch_size": batch_size,
        "image_variant": image_variant,
        "occlusion_margin": occlusion_margin,
        "candidate_scoring": candidate_scoring,
        "long_max_new_tokens": long_max_new_tokens,
        "long_batch_size": long_batch_size,
        "preserve_visual_fp32": preserve_visual_fp32,
    }
    remote_function = eval_h200 if gpu == "h200" else eval_remote
    if spawn_eval:
        call = remote_function.spawn(**kwargs)
        print(
            json.dumps(
                {
                    "status": "spawned",
                    "function_call_id": call.object_id,
                    "output_dir": output_dir,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = remote_function.remote(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
