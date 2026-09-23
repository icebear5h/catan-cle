"""Inspect native adapter contents and merged HF headers without loading the base."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_str, load_json_dict
from sft.miles_sft.export import validate_complete_export
from sft.miles_sft.export._validation import COMPOSITION

from .contracts import require
from .storage import file_hash


def _file(path: Path) -> None:
    require(path.is_file() and path.stat().st_size > 0, f"missing/empty checkpoint file: {path}")


def _tensor_state(path: Path) -> dict[str, torch.Tensor]:
    _file(path)
    loaded: object = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(loaded, Mapping) and bool(loaded), f"empty/invalid native adapter: {path}")
    if not isinstance(loaded, Mapping):
        raise ValueError(str(path))
    state: dict[str, torch.Tensor] = {}
    for name, tensor in loaded.items():
        require(isinstance(name, str) and isinstance(tensor, torch.Tensor), "invalid adapter entry")
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(str(path))
        require(bool(torch.isfinite(tensor).all().item()), f"nonfinite saved adapter: {name}")
        state[name] = tensor
    return state


def audit_native(
    checkpoint: Path, rollout_id: int, scopes: list[JsonDict],
) -> JsonLikeDict:
    adapter = checkpoint / "adapter"
    expected_files = {f"adapter_megatron_rank{rank}.pt" for rank in range(len(scopes))}
    require({p.name for p in adapter.glob("adapter_megatron_rank*.pt")} == expected_files,
            "missing/extra native adapter rank shards")
    result: JsonLikeDict = {}
    for rank, scope in enumerate(scopes):
        path = adapter / f"adapter_megatron_rank{rank}.pt"
        state = _tensor_state(path)
        expected = as_dict(scope["trainable_shapes"])
        require({name: list(t.shape) for name, t in state.items()} == expected,
                f"saved rank {rank} adapter names/shapes differ from live trainable scope")
        training = adapter / f"training_state_rank{rank}.pt"
        _file(training)
        payload: object = torch.load(training, map_location="cpu", weights_only=True)
        require(isinstance(payload, dict), "invalid native training state")
        if not isinstance(payload, dict):
            raise ValueError(str(training))
        require(payload.get("iteration") == rollout_id and payload.get("optimizer") is not None
                and payload.get("opt_param_scheduler") is not None, "incomplete native optimizer/scheduler state")
        result[str(rank)] = {"adapter_sha256": file_hash(path), "adapter_bytes": path.stat().st_size,
                             "training_state_sha256": file_hash(training), "tensors": len(state)}
    return result


def audit_hf_export(base: Path, exported: Path, *, bridge_export: Path | None = None,
                    expected_sha256: str | None = None) -> JsonLikeDict:
    validate_complete_export(base, exported, bridge_export=bridge_export, expected_sha256=expected_sha256)
    manifest = load_json_dict(exported / COMPOSITION)
    tensors = as_dict(manifest["tensors"])
    files = as_dict(manifest["output_files"])
    shards = {as_str(as_dict(entry)["shard"]) for entry in tensors.values()}
    return {"path": str(exported.resolve()), "tensor_count": len(tensors),
            "composition_manifest_sha256": file_hash(exported / COMPOSITION),
            "base_manifest_sha256": manifest["base_manifest_sha256"],
            "trained_keys": manifest["trained_keys"], "frozen_tensor_count": sum(
                as_dict(entry)["source"] == "base" for entry in tensors.values()),
            "index_sha256": as_dict(files["model.safetensors.index.json"])["sha256"],
            "asset_sha256": {name: as_dict(entry)["sha256"] for name, entry in files.items()
                             if name not in shards and name != "model.safetensors.index.json"},
            "shards": {name: files[name] for name in sorted(shards)}}
