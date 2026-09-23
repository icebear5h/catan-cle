"""LEGACY Modal launcher for the upstream Qwen-VL-Series-Finetune trainer.

The active path is ``sft/launchers/modal_catan_vision_sft.py``. This file is retained for
historical smoke evidence. It converts local JSONL data into
the Qwen repo's conversation JSON format, uploads images to a Modal Volume, and
runs the pinned upstream trainer through a small Catan-token wrapper.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
from pathlib import Path

from sft.launchers.qwen_series._train_support import (
    QWEN_COMMIT,
    REMOTE_CACHE_MOUNT,
    REMOTE_DATA_MOUNT,
    REMOTE_QWEN_DIR,
    REMOTE_RUNS_MOUNT,
    app,
    hf_cache,
    qwen_series_image,
    sft_data,
    sft_runs,
    upload_qwen_series_json,
)


@app.function(
    image=qwen_series_image,
    gpu="L40S",
    volumes={
        REMOTE_CACHE_MOUNT: hf_cache,
        REMOTE_DATA_MOUNT: sft_data,
        REMOTE_RUNS_MOUNT: sft_runs,
    },
    timeout=60 * 60 * 8,
)
def train_remote(
    train_json: str,
    image_folder: str,
    output_dir: str = f"{REMOTE_RUNS_MOUNT}/qwen-series-catan-smoke",
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    max_steps: int | None = 1,
) -> dict[str, str | int | None]:
    command = [
        "python",
        "-m",
        "sft.scripts.train.train_qwen_series_with_catan_tokens",
        "--catan_patch_no_deepspeed_savers",
        "--model_id",
        model_id,
        "--data_path",
        train_json,
        "--image_folder",
        image_folder,
        "--output_dir",
        output_dir,
        "--remove_unused_columns",
        "False",
        "--lora_enable",
        "True",
        "--lora_namespan_exclude",
        '["lm_head", "embed_tokens", "embed_token"]',
        "--lora_rank",
        "8",
        "--lora_alpha",
        "16",
        "--lora_dropout",
        "0.05",
        "--num_lora_modules",
        "-1",
        "--bits",
        "4",
        "--double_quant",
        "True",
        "--quant_type",
        "nf4",
        "--freeze_llm",
        "True",
        "--freeze_vision_tower",
        "True",
        "--freeze_merger",
        "True",
        "--bf16",
        "True",
        "--fp16",
        "False",
        "--disable_flash_attn2",
        "True",
        "--use_liger_kernel",
        "False",
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--image_min_pixels",
        str(256 * 28 * 28),
        "--image_max_pixels",
        str(512 * 28 * 28),
        "--learning_rate",
        "1e-4",
        "--weight_decay",
        "0.1",
        "--warmup_ratio",
        "0.03",
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        "1",
        "--save_strategy",
        "no",
        "--report_to",
        "none",
        "--lazy_preprocess",
        "True",
        "--gradient_checkpointing",
        "True",
        "--dataloader_num_workers",
        "0",
    ]
    if max_steps is not None:
        command.extend(["--max_steps", str(max_steps)])

    print("running:", " ".join(command))
    subprocess.run(command, check=True, cwd=REMOTE_QWEN_DIR)
    sft_runs.commit()
    return {
        "output_dir": output_dir,
        "model_id": model_id,
        "max_steps": max_steps,
        "qwen_commit": QWEN_COMMIT,
    }


@app.function(
    image=qwen_series_image,
    timeout=60 * 30,
)
def check_fast_kernel_deps_remote() -> dict[str, object]:
    torch = importlib.import_module("torch")

    modules = [
        "fla",
        "fla.ops.gated_delta_rule",
        "causal_conv1d",
        "transformers.models.qwen3_5.modeling_qwen3_5",
    ]
    result: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "modules": {},
    }
    module_status: dict[str, dict[str, str | bool | None]] = {}
    for module_name in modules:
        status: dict[str, str | bool | None] = {
            "found": importlib.util.find_spec(module_name) is not None,
            "error": None,
        }
        if status["found"]:
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - remote env diagnostic.
                status["error"] = repr(exc)
        module_status[module_name] = status

    result["modules"] = module_status

    try:
        qwen35 = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        result["qwen3_5_fast_path_available"] = bool(
            getattr(qwen35, "is_fast_path_available", False)
        )
        result["causal_conv1d_fn_available"] = getattr(qwen35, "causal_conv1d_fn", None) is not None
        result["chunk_gated_delta_rule_available"] = (
            getattr(qwen35, "chunk_gated_delta_rule", None) is not None
        )
        result["fused_recurrent_gated_delta_rule_available"] = (
            getattr(qwen35, "fused_recurrent_gated_delta_rule", None) is not None
        )
    except Exception as exc:  # pragma: no cover - remote env diagnostic.
        result["qwen3_5_import_error"] = repr(exc)

    return result


@app.local_entrypoint()
def main(
    train_jsonl: str | None = None,
    remote_dir: str = "catan-qwen-series-sft/smoke",
    output_dir: str = f"{REMOTE_RUNS_MOUNT}/qwen-series-catan-smoke",
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    max_steps: int = 1,
    check_fast_kernels: bool = False,
) -> None:
    if check_fast_kernels:
        result = check_fast_kernel_deps_remote.remote()
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if train_jsonl is None:
        raise ValueError("--train-jsonl is required unless --check-fast-kernels is set")

    remote_train_json, remote_image_folder = upload_qwen_series_json(
        Path(train_jsonl),
        remote_dir,
    )
    training = train_remote.remote(
        train_json=remote_train_json,
        image_folder=remote_image_folder,
        output_dir=output_dir,
        model_id=model_id,
        max_steps=max_steps,
    )
    print(training)
