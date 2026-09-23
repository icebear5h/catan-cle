"""Receipts, token upload and the remote run behind the vision-SFT launcher."""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from sft.json_types import JsonDict, JsonLike, JsonLikeDict, as_dict, as_int, loads_json
from sft.launchers.qwen_series._train_support import (
    QWEN_COMMIT,
    REMOTE_DATA_MOUNT,
    REMOTE_QWEN_DIR,
    hf_cache,
    sft_data,
    sft_runs,
)
from sft.launchers.qwen_series._vision_config import vision_config
from sft.qwen_series_vision_sft import (
    build_vision_sft_command,
    launch_identity,
    validate_hardware_profile,
)

LAUNCH_MANIFEST = "launch_manifest.json"
TRAINABLE_MANIFEST = "trainable_parameters.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: Mapping[str, JsonLike]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _upload_token_inventory(token_inventory: Path, remote_dir: str) -> str:
    local_path = token_inventory.expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    remote_path = f"/{remote_dir.strip('/')}/trainable_tokens.json"
    with sft_data.batch_upload(force=True) as batch:
        batch.put_file(local_path, remote_path)
    return f"{REMOTE_DATA_MOUNT}{remote_path}"


def _artifact_evidence(output_dir: Path) -> dict[str, dict[str, str | int]]:
    required_paths = {
        "adapter_config": output_dir / "adapter_config.json",
        "non_lora_state": output_dir / "non_lora_state_dict.bin",
        "trainable_parameters": output_dir / TRAINABLE_MANIFEST,
        "tokenizer_config": output_dir / "tokenizer_config.json",
    }
    adapter_candidates = [
        output_dir / "adapter_model.safetensors",
        output_dir / "adapter_model.bin",
    ]

    missing = [name for name, path in required_paths.items() if not path.is_file()]
    adapter_path = next((path for path in adapter_candidates if path.is_file()), None)
    if adapter_path is None:
        missing.append("adapter_model")
    if missing:
        raise RuntimeError(
            f"Training command completed without required artifacts in {output_dir}: "
            + ", ".join(sorted(missing))
        )

    paths = dict(required_paths)
    if adapter_path is not None:  # Always set: a missing adapter raised above.
        paths["adapter_model"] = adapter_path
    return {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
        }
        for name, path in sorted(paths.items())
    }


def _identity_dataset(dataset: Mapping[str, JsonLike]) -> JsonLikeDict:
    keys = (
        "source_sha256",
        "combined_sha256",
        "rows",
        "unique_images",
        "max_prompt_characters",
        "max_answer_characters",
        "annotation_root",
        "image_root",
        "token_inventory",
    )
    return {key: dataset[key] for key in keys}


def _run_remote(
    *,
    hardware: str,
    train_json: str,
    image_folder: str,
    output_dir: str,
    token_inventory: str,
    config_payload: JsonDict,
    dataset: JsonLikeDict,
) -> JsonLikeDict:
    config = vision_config(config_payload)
    validate_hardware_profile(config.model_id, hardware)
    command = build_vision_sft_command(
        config,
        train_json=train_json,
        image_folder=image_folder,
        output_dir=output_dir,
        token_inventory=token_inventory,
    )

    output_path = Path(output_dir)
    manifest_path = output_path / LAUNCH_MANIFEST
    identity_payload: JsonLikeDict = {
        "schema": "catan_qwen_vision_sft_launch_identity/v2",
        "config": config.as_manifest_dict(),
        "dataset": _identity_dataset(dataset),
        "hardware": hardware,
        "qwen_repo_commit": QWEN_COMMIT,
    }
    identity = launch_identity(identity_payload)

    existing_manifest: JsonDict | None = None
    if manifest_path.is_file():
        existing_manifest = as_dict(loads_json(manifest_path.read_text()))
        if existing_manifest.get("identity") != identity:
            raise RuntimeError(f"Refusing to reuse {output_path} for a different launch identity")
        if existing_manifest.get("status") == "completed":
            artifacts = _artifact_evidence(output_path)
            return {
                "status": "already_completed",
                "identity": identity,
                "output_dir": output_dir,
                "artifacts": artifacts,
            }
    elif output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError(
            f"Refusing non-empty output directory without {LAUNCH_MANIFEST}: {output_path}"
        )

    attempts = as_int((existing_manifest or {}).get("attempts", 0)) + 1
    manifest: JsonLikeDict = {
        **identity_payload,
        "identity": identity,
        "status": "running",
        "attempts": attempts,
        "started_at": _utc_now(),
        "completed_at": None,
        "command": command,
        "output_dir": output_dir,
    }
    _write_json_atomic(manifest_path, manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("running:", shlex.join(command))
    try:
        subprocess.run(command, check=True, cwd=REMOTE_QWEN_DIR)
        artifacts = _artifact_evidence(output_path)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["completed_at"] = _utc_now()
        manifest["error"] = repr(exc)
        _write_json_atomic(manifest_path, manifest)
        raise
    else:
        manifest["status"] = "completed"
        manifest["completed_at"] = _utc_now()
        manifest["artifacts"] = artifacts
        _write_json_atomic(manifest_path, manifest)
        return {
            "status": "completed",
            "identity": identity,
            "output_dir": output_dir,
            "artifacts": artifacts,
        }
    finally:
        hf_cache.commit()
        sft_runs.commit()
