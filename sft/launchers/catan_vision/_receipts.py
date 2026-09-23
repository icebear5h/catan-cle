"""Launch-manifest hashing, atomic writes and trained-artifact checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from sft.json_types import JsonLike
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(payload: Mapping[str, JsonLike]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, JsonLike]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _dependency_versions() -> dict[str, str]:
    return {
        "torch": TORCH_VERSION,
        "torchvision": TORCHVISION_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "trl": TRL_VERSION,
        "peft": PEFT_VERSION,
        "datasets": DATASETS_VERSION,
        "accelerate": ACCELERATE_VERSION,
        "huggingface_hub": HUGGINGFACE_HUB_VERSION,
        "safetensors": SAFETENSORS_VERSION,
        "pillow": PILLOW_VERSION,
    }


def _required_artifacts(output_dir: Path) -> dict[str, dict[str, str | int]]:
    final_dir = output_dir / "final"
    candidates = {
        "adapter_config": final_dir / "adapter_config.json",
        "adapter_model": final_dir / "adapter_model.safetensors",
        "visual_model": final_dir / "visual_model.safetensors",
        "tokenizer_config": final_dir / "tokenizer_config.json",
        "trainable_parameters": final_dir / "trainable_parameters.json",
        "reload_validation": final_dir / "reload_validation.json",
    }
    missing = [name for name, path in candidates.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"training completed without artifacts: {', '.join(missing)}")
    return {
        name: {"path": str(path), "bytes": path.stat().st_size}
        for name, path in candidates.items()
    }


def _latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return the highest numbered complete-looking Trainer checkpoint."""

    candidates: list[tuple[int, Path]] = []
    if not checkpoints_dir.is_dir():
        return None
    for path in checkpoints_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
        if all((path / name).is_file() for name in required):
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]
