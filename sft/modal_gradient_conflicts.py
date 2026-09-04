"""Dry-run-first Modal launcher for Catan behavior gradient diagnostics."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from sft.gradient_diagnostics import (
    iter_jsonl,
    select_probe_rows,
    selection_manifest,
)
from sft.scripts.probe_gradient_conflicts import run_gradient_probe
from sft.scripts.train_trl_catan_vision import (
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
    load_token_inventory,
    resolve_image_path,
)


APP_NAME = "catan-gradient-conflicts"
REMOTE_CACHE = "/cache"
REMOTE_DATA = "/data"
REMOTE_RUNS = "/runs"
HF_SECRET_NAME = "catan-hf"
LAUNCH_MANIFEST = "modal_launch.json"

REPLAY_ROOT = Path("artifacts/generated/board_recognition/replay_v1")
DEFAULT_PAIR_JSONL = REPLAY_ROOT / "spatial_localization_pairs_v2/stage1/train.jsonl"
DEFAULT_PAIR_IMAGES = REPLAY_ROOT / "spatial_localization_pairs_v2/images"
DEFAULT_SINGLE_JSONL = REPLAY_ROOT / "spatial_localization_v7/stage1/train.jsonl"
DEFAULT_SINGLE_IMAGES = REPLAY_ROOT / "spatial_localization_v7/images"
DEFAULT_TOKEN_INVENTORY = REPLAY_ROOT / "ms_swift_bidirectional_v1/trainable_tokens.json"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("catan-hf-cache", create_if_missing=True)
sft_data = modal.Volume.from_name("catan-sft-data", create_if_missing=True)
sft_runs = modal.Volume.from_name("catan-sft-runs", create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])

probe_image = (
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
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_python_source("cle")
    .add_local_python_source("evals")
    .add_local_python_source("sft")
)


def _canonical_hash(payload: JsonDict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


JsonDict = dict[str, Any]


def build_local_selection(
    pair_jsonl: Path,
    single_jsonl: Path,
    *,
    seed: int,
    rows_per_behavior: int,
) -> tuple[list[Any], JsonDict]:
    selected = select_probe_rows(
        list(iter_jsonl(pair_jsonl)),
        list(iter_jsonl(single_jsonl)),
        rows_per_behavior=rows_per_behavior,
        seed=seed,
    )
    manifest = selection_manifest(
        selected,
        seed=seed,
        rows_per_behavior=rows_per_behavior,
    )
    return selected, manifest


def upload_probe_bundle(
    selected: list[Any],
    manifest: JsonDict,
    *,
    pair_image_root: Path,
    single_image_root: Path,
    token_inventory: Path,
    remote_dir: str,
) -> tuple[str, str, str, str]:
    load_token_inventory(token_inventory)
    remote_root = "/" + remote_dir.strip("/")
    remote_jsonl = f"{remote_root}/probe.jsonl"
    remote_images = f"{remote_root}/images"
    remote_inventory = f"{remote_root}/trainable_tokens.json"
    remote_manifest = f"{remote_root}/selection.json"
    image_names: dict[Path, str] = {}

    with tempfile.TemporaryDirectory(prefix="catan-gradient-probe-") as temporary_dir:
        temp_root = Path(temporary_dir)
        rewritten_jsonl = temp_root / "probe.jsonl"
        manifest_path = temp_root / "selection.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        with rewritten_jsonl.open("w") as handle:
            for index, selected_row in enumerate(selected, start=1):
                root = pair_image_root if selected_row.source == "pairs" else single_image_root
                image_path = resolve_image_path(root, selected_row.row, line_number=index)
                image_name = image_names.get(image_path)
                if image_name is None:
                    image_name = f"{len(image_names):06d}{image_path.suffix.lower() or '.png'}"
                    image_names[image_path] = image_name
                row = dict(selected_row.row)
                row.pop("image", None)
                row["images"] = [image_name]
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        with sft_data.batch_upload(force=True) as batch:
            for image_path, image_name in image_names.items():
                batch.put_file(image_path, f"{remote_images}/{image_name}")
            batch.put_file(rewritten_jsonl, remote_jsonl)
            batch.put_file(manifest_path, remote_manifest)
            batch.put_file(token_inventory, remote_inventory)

    return (
        f"{REMOTE_DATA}{remote_jsonl}",
        f"{REMOTE_DATA}{remote_images}",
        f"{REMOTE_DATA}{remote_inventory}",
        f"{REMOTE_DATA}{remote_manifest}",
    )


@app.function(
    image=probe_image,
    gpu="H200",
    cpu=16.0,
    memory=128 * 1024,
    secrets=[hf_secret],
    volumes={REMOTE_CACHE: hf_cache, REMOTE_DATA: sft_data, REMOTE_RUNS: sft_runs},
    timeout=60 * 60 * 8,
)
def probe_h200(
    probe_args: JsonDict,
    launch_payload: JsonDict,
) -> JsonDict:
    output_dir = Path(probe_args["output_dir"])
    manifest_path = output_dir / LAUNCH_MANIFEST
    identity = _canonical_hash(launch_payload)
    existing = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if existing is not None and existing.get("identity") != identity:
        raise RuntimeError(f"refusing to reuse {output_dir} for a different probe")
    if existing is not None and existing.get("status") == "completed":
        report_path = output_dir / "gradient_conflicts.json"
        if not report_path.is_file():
            raise RuntimeError("completed launch is missing gradient_conflicts.json")
        return json.loads(report_path.read_text())
    if existing is not None:
        raise RuntimeError("an incomplete probe exists; use a new label after inspecting it")

    launch = {
        **launch_payload,
        "identity": identity,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }
    _write_json_atomic(manifest_path, launch)
    try:
        report = run_gradient_probe(**probe_args)
    except Exception as exc:
        launch.update(
            status="failed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=repr(exc),
        )
        _write_json_atomic(manifest_path, launch)
        raise
    else:
        launch.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat())
        _write_json_atomic(manifest_path, launch)
        return report
    finally:
        hf_cache.commit()
        sft_runs.commit()


@app.local_entrypoint()
def main(
    adapter_dir: str,
    label: str,
    pair_jsonl: str = str(DEFAULT_PAIR_JSONL),
    pair_image_root: str = str(DEFAULT_PAIR_IMAGES),
    single_jsonl: str = str(DEFAULT_SINGLE_JSONL),
    single_image_root: str = str(DEFAULT_SINGLE_IMAGES),
    token_inventory: str = str(DEFAULT_TOKEN_INVENTORY),
    seed: int = 42,
    rows_per_behavior: int = 24,
    microbatch_size: int = 4,
    spawn_probe: bool = False,
    dry_run: bool = True,
) -> None:
    """Print an inspectable selection by default; ``--no-dry-run`` uses an H200."""

    if not adapter_dir.startswith("/runs/"):
        raise ValueError("adapter_dir must be the container path under /runs/")
    if not label or "/" in label:
        raise ValueError("label must be one non-empty path component")
    pair_path = Path(pair_jsonl)
    single_path = Path(single_jsonl)
    pair_images = Path(pair_image_root)
    single_images = Path(single_image_root)
    inventory_path = Path(token_inventory)
    selected, selection = build_local_selection(
        pair_path,
        single_path,
        seed=seed,
        rows_per_behavior=rows_per_behavior,
    )
    load_token_inventory(inventory_path)
    remote_dir = f"catan-gradient-probe/{selection['identity'][:16]}"
    output_dir = f"{REMOTE_RUNS}/qwen-series-eval/gradient-conflicts/{label}"
    remote_jsonl = f"{REMOTE_DATA}/{remote_dir}/probe.jsonl"
    remote_images = f"{REMOTE_DATA}/{remote_dir}/images"
    remote_inventory = f"{REMOTE_DATA}/{remote_dir}/trainable_tokens.json"
    remote_manifest = f"{REMOTE_DATA}/{remote_dir}/selection.json"
    probe_args = {
        "adapter_dir": adapter_dir,
        "probe_jsonl": remote_jsonl,
        "image_root": remote_images,
        "token_inventory": remote_inventory,
        "selection_manifest_path": remote_manifest,
        "output_dir": output_dir,
        "microbatch_size": microbatch_size,
    }
    launch = {
        "schema": "catan_modal_gradient_conflict_launch/v1",
        "hardware": "H200",
        "adapter_dir": adapter_dir,
        "selection": selection,
        "probe_args": probe_args,
    }
    overview = {
        **launch,
        "identity": _canonical_hash(launch),
        "dry_run": dry_run,
        "paid_gpu_requested": not dry_run,
        "spawn_probe": spawn_probe,
        "behavior_counts": dict(Counter(item.behavior for item in selected)),
    }
    print(json.dumps(overview, indent=2, sort_keys=True))
    if dry_run:
        return

    uploaded = upload_probe_bundle(
        selected,
        selection,
        pair_image_root=pair_images,
        single_image_root=single_images,
        token_inventory=inventory_path,
        remote_dir=remote_dir,
    )
    if uploaded != (remote_jsonl, remote_images, remote_inventory, remote_manifest):
        raise RuntimeError("uploaded paths disagree with the launch plan")
    if spawn_probe:
        call = probe_h200.spawn(probe_args, launch)
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
    report = probe_h200.remote(probe_args, launch)
    print(json.dumps(report, indent=2, sort_keys=True))
