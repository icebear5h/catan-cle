"""Inspect native adapter contents and merged HF headers without loading the base."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open

from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_str, load_json_dict

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


def _weight_headers(directory: Path) -> tuple[dict[str, list[int]], dict[str, str]]:
    index = directory / "model.safetensors.index.json"
    _file(index)
    weight_map = {key: as_str(value) for key, value in as_dict(load_json_dict(index)["weight_map"]).items()}
    require(bool(weight_map), "empty merged HF weight index")
    by_shard: dict[str, set[str]] = {}
    for name, shard in weight_map.items():
        require(Path(shard).name == shard and shard.endswith(".safetensors"), "unsafe HF shard path")
        require(not any(term in name for term in (".adapter.", "lora_", "trainable_tokens")),
                "HF export still contains unmerged adapter/token-row tensors")
        by_shard.setdefault(shard, set()).add(name)
    headers: dict[str, list[int]] = {}
    for shard, names in by_shard.items():
        path = directory / shard
        _file(path)
        with safe_open(path, framework="pt", device="cpu") as handle:
            require(set(handle.keys()) == names, f"HF index/shard tensor mismatch: {shard}")
            headers.update({name: list(handle.get_slice(name).get_shape()) for name in names})
    return headers, weight_map


def audit_hf_export(base: Path, exported: Path) -> JsonLikeDict:
    require((exported / ".complete").is_file(), "merged HF export has no .complete marker")
    expected, _ = _weight_headers(base)
    actual, weights = _weight_headers(exported)
    require(actual == expected, "merged HF export has missing/extra/reshaped base weights")
    require(any("embed_tokens.weight" in n for n in actual)
            and any(n.endswith("lm_head.weight") for n in actual)
            and any("visual." in n or "vision_model." in n for n in actual),
            "merged checkpoint must include vision and both full token matrices")
    _file(exported / "config.json")
    assets = ("tokenizer.json", "tokenizer_config.json")
    asset_hashes: dict[str, str] = {}
    for name in assets:
        _file(base / name)
        _file(exported / name)
        asset_hashes[name] = file_hash(exported / name)
        require(asset_hashes[name] == file_hash(base / name), f"export changed saved tokenizer: {name}")
    for name in ("added_tokens.json", "special_tokens_map.json", "chat_template.jinja"):
        if (base / name).is_file():
            _file(exported / name)
            asset_hashes[name] = file_hash(exported / name)
            require(asset_hashes[name] == file_hash(base / name), f"export changed tokenizer asset: {name}")
    return {"path": str(exported.resolve()), "tensor_count": len(actual),
            "index_sha256": file_hash(exported / "model.safetensors.index.json"),
            "tokenizer_sha256": asset_hashes,
            "shards": {name: (exported / name).stat().st_size for name in sorted(set(weights.values()))}}
