"""Pinned container, Volumes, budget rates and Modal app for the vision launcher."""

from __future__ import annotations

import os
from pathlib import PurePosixPath
from typing import TypedDict

import modal

from sft.scripts.train.train_trl_catan_vision import (
    ACCELERATE_VERSION,
    DATASETS_VERSION,
    HUGGINGFACE_HUB_VERSION,
    PEFT_VERSION,
    PILLOW_VERSION,
    SAFETENSORS_VERSION,
    TORCH_VERSION,
    TORCHVISION_VERSION,
    TRANSFORMERS_VERSION,
    TRL_VERSION,
)

APP_NAME = "catan-qwen3-8-vision-sft"
HF_SECRET_NAME = os.environ.get("CATAN_HF_SECRET_NAME", "catan-hf")
REMOTE_CACHE = "/cache"
REMOTE_DATA = "/data"
REMOTE_RUNS = "/runs"
LAUNCH_MANIFEST = "modal_launch.json"
BUDGET_GUARD_FILE = "budget_guard.json"
BUDGET_TIMEOUT_SECONDS = 7 * 60 * 60
BUDGET_STARTUP_SECONDS = 10 * 60
BUDGET_EVAL_RESERVE_USD = 25.0
# Published standard Function rates checked 2026-09-05; recheck before reuse.
BUDGET_RATE_PER_SECOND = 0.001261 + 16 * 0.0000131 + 128 * 0.00000222


class BudgetFunctionOptions(TypedDict):
    """`modal.App.function` options for the bounded H200 run, checkable through `**`."""

    gpu: str
    cpu: tuple[float, float]
    memory: tuple[int, int]
    timeout: int
    startup_timeout: int
    retries: int
    max_containers: int
    scaledown_window: int


BUDGET_FUNCTION_OPTIONS: BudgetFunctionOptions = {
    "gpu": "H200",
    "cpu": (16.0, 16.0),
    "memory": (128 * 1024, 128 * 1024),
    "timeout": BUDGET_TIMEOUT_SECONDS,
    "startup_timeout": BUDGET_STARTUP_SECONDS,
    "retries": 0,
    "max_containers": 1,
    "scaledown_window": 2,
}

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("catan-hf-cache", create_if_missing=True)
sft_data = modal.Volume.from_name("catan-sft-data", create_if_missing=True)
sft_runs = modal.Volume.from_name("catan-sft-runs", create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])

training_base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        f"trl=={TRL_VERSION}",
        f"peft=={PEFT_VERSION}",
        f"datasets=={DATASETS_VERSION}",
        f"accelerate=={ACCELERATE_VERSION}",
        f"huggingface-hub=={HUGGINGFACE_HUB_VERSION}",
        f"safetensors=={SAFETENSORS_VERSION}",
        f"Pillow=={PILLOW_VERSION}",
    )
    .env(
        {
            "HF_HOME": f"{REMOTE_CACHE}/huggingface",
            "HF_HUB_CACHE": f"{REMOTE_CACHE}/huggingface/hub",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # Stream trainer prints immediately; otherwise the container is
            # silent until the first tqdm bar on stderr.
            "PYTHONUNBUFFERED": "1",
        }
    )
)

training_image = (
    training_base_image.add_local_python_source("cle")
    .add_local_python_source("evals")
    .add_local_python_source("sft")
)

_VOLUMES: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    REMOTE_CACHE: hf_cache,
    REMOTE_DATA: sft_data,
    REMOTE_RUNS: sft_runs,
}
