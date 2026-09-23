"""Compose Bridge-trained language weights with the exact validated frozen base."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from sft.miles_sft.merge._contracts import (
    INDEX,
    MANIFEST,
    Checkpoint,
    file_identity,
    identities,
    object_map,
    require,
    sha256,
    write_json,
)
from sft.safetensor_types import open_tensors

from ._validation import (
    COMPLETE,
    COMPOSITION,
    SCHEMA,
    admit_bridge,
    base_inventory,
    bridge_checkpoint,
    validate_complete_export,
    validate_payload,
)

__all__ = ["assemble_complete_export", "validate_complete_export"]


def _tensor_hash(tensor: torch.Tensor) -> str:
    # A byte view avoids a full FP32/BF16 conversion or a copied Python bytes buffer.
    raw = tensor.detach().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(raw.data).hexdigest()


def _write_shard(base: Path, bridge_export: Path, output: Path, source: str, name: str,
                 checkpoint: Checkpoint, bridge: Checkpoint, trained: set[str],
                 ) -> dict[str, object]:
    records: dict[str, object] = {}
    tensors: dict[str, torch.Tensor] = {}
    with open_tensors(base / source) as handle:
        for key in checkpoint.shards[source]:
            if key in trained:
                with open_tensors(bridge_export / bridge.weight_map[key]) as exported:
                    tensors[key] = exported.get_tensor(key)
            else:
                tensors[key] = handle.get_tensor(key)
            records[key] = {**checkpoint.headers[key], "shard": name,
                            "source": "bridge" if key in trained else "base",
                            "source_shard": bridge.weight_map[key] if key in trained else source,
                            "source_sha256": _tensor_hash(tensors[key])}
        save_file(tensors, output / name, metadata={"format": "pt"})
    del tensors
    with open_tensors(output / name) as saved:
        for key, raw in records.items():
            entry = object_map(raw)
            digest = _tensor_hash(saved.get_tensor(key))
            require(digest == entry["source_sha256"], f"tensor bytes changed during composition: {key}")
            records[key] = {**entry, "sha256": digest}
    return records


def assemble_complete_export(base: Path, bridge_export: Path, output: Path) -> Path:
    """Write a fresh sibling ``model`` and return its composition manifest path.

    ``base`` must be our fully validated merge. Only its manifest's
    ``lora_fp32_then_bf16`` keys are admitted from Bridge, with exact shapes/dtypes.
    Frozen tensors (including MTP, FP32 vision and full token rows) and all assets
    come from the base. Memory is bounded by one base-shaped shard. Failed writes
    leave an incomplete new directory; neither input is modified or deleted.
    """
    base, bridge_export = base.resolve(), bridge_export.resolve()
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"fresh composition output required: {output}")
    require(output.resolve() == bridge_export.parent / "model" and output.resolve() != bridge_export,
            "composition output must be sibling model")
    require(not output.resolve().is_relative_to(base) and not base.is_relative_to(output.resolve()),
            "composition output overlaps base")
    original, checkpoint, trained, assets = base_inventory(base)
    bridge = bridge_checkpoint(bridge_export)
    admit_bridge(checkpoint, bridge, trained)
    base_digest = sha256(base / MANIFEST)
    bridge_files = identities(bridge_export, set(bridge.shards) | {INDEX, COMPLETE})
    source_files = object_map(original["output_files"])
    output.mkdir(parents=True, exist_ok=False)
    weights: dict[str, str] = {}
    tensors: dict[str, object] = {}
    for index, source in enumerate(checkpoint.shards, start=1):
        require(file_identity(base / source) == source_files[source], f"base shard hash/size mismatch: {source}")
        name = f"model-{index:05d}-of-{len(checkpoint.shards):05d}.safetensors"
        records = _write_shard(base, bridge_export, output, source, name, checkpoint, bridge, trained)
        weights.update(dict.fromkeys(records, name))
        tensors.update(records)
    write_json(output / INDEX, {"metadata": {"total_size": checkpoint.total_size}, "weight_map": weights})
    for name in sorted(assets):
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (base / name).open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 << 20)
    files = identities(output, set(weights.values()) | assets | {INDEX})
    manifest: dict[str, object] = {
        "schema": SCHEMA, "status": "completed", "base": str(base), "bridge_export": str(bridge_export),
        "base_manifest_sha256": base_digest, "base_files": source_files, "bridge_files": bridge_files,
        "trained_keys": sorted(trained), "assets": sorted(assets), "tensors": tensors, "output_files": files,
    }
    write_json(output / COMPOSITION, manifest)
    validate_payload(base, output, manifest, bridge_export)
    # Completion is distinct from both the raw Bridge marker and the base merge seal.
    write_json(output / COMPLETE, {"schema": SCHEMA, "manifest_sha256": sha256(output / COMPOSITION)})
    return output / COMPOSITION
