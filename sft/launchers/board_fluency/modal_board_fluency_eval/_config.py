from __future__ import annotations

from pathlib import PurePosixPath
from typing import TypedDict

import modal

from sft.launchers.modal_catan_vision_sft import (
    HF_SECRET_NAME,
    hf_cache,
    sft_data,
    sft_runs,
    training_base_image,
)
from sft.scripts.train.train_trl_catan_vision import (
    RUN_CONFIG_FILE,
    TRAINABLE_SCOPE_FILE,
    VISUAL_STATE_FILE,
)

MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DEFAULT_REVIEW = "artifacts/generated/sft/symbolic_board_fluency_review_v1/review.jsonl"
DEFAULT_INVENTORY = "artifacts/generated/sft/symbolic_board_v2/trainable_tokens.json"
ORIGINAL_INVENTORY = (
    "artifacts/generated/board_recognition/replay_v1/"
    "ms_swift_bidirectional_v1/trainable_tokens.json"
)
ROWS = 200
CONTEXT = 4096
NEW_TOKENS = 512
BATCH = 16
GPU_DEADLINE = 840
COMPARISON_GPU_DEADLINE = 1740
CACHE = "/cache/huggingface/hub"
VOLUMES: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    "/cache": hf_cache, "/data": sft_data, "/runs": sft_runs}


def reload_volumes() -> None:
    """Reload every mounted volume before reading container-visible paths.

    `VOLUMES` is typed for Modal's own `volumes=` signature, which accepts
    cloud-bucket mounts too; only real volumes expose `reload`.
    """
    for volume in VOLUMES.values():
        if isinstance(volume, modal.Volume):
            volume.reload()


class ModalCommon(TypedDict):
    """`modal.App.function` options shared by every stage of a launcher.

    Spelled as a TypedDict so `@app.function(**COMMON, ...)` keeps checking each
    option against Modal's own signature instead of collapsing to `object`.
    """

    image: modal.Image
    volumes: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]
    secrets: list[modal.Secret]
    startup_timeout: int
    retries: int
    max_containers: int
    scaledown_window: int


class ModalStage(ModalCommon, total=False):
    """`ModalCommon` plus a per-stage shape; CPU stages omit `gpu`."""

    gpu: str
    cpu: tuple[float, float]
    memory: tuple[int, int]


class ModalGpu(ModalCommon):
    """`ModalCommon` plus the fixed accelerator shape for GPU stages."""

    gpu: str
    cpu: tuple[float, float]
    memory: tuple[int, int]
REQUIRED_BUNDLE = {
    "adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json",
    "tokenizer.json", RUN_CONFIG_FILE, TRAINABLE_SCOPE_FILE, VISUAL_STATE_FILE,
}
INFERENCE_SIDECARS = {
    "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
    "tokenizer.model", "special_tokens_map.json", "added_tokens.json", "vocab.json",
    "merges.txt", "chat_template.jinja", "chat_template.json", "processor_config.json",
    "preprocessor_config.json", "video_preprocessor_config.json",
}

app = modal.App("catan-board-fluency-eval")
# The symbolic scorer imports the board data package, whose source-contract
# helpers also import the viewer's detached ServerState definition.
eval_image = training_base_image.pip_install("jsonschema==4.25.1", "pydantic==2.12.3").add_local_python_source(
    "cle", "evals", "sft", "data_pipeline", "playground",
)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])
