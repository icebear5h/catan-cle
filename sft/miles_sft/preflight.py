"""CPU identity and tokenization checks before a real Miles training worker starts."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Protocol, cast

from transformers import AutoTokenizer

from sft.json_types import JsonLikeDict, as_dict, as_int, as_list, as_str, load_json_dict

from .config import BRIDGE_REVISION, MEGATRON_REVISION, MILES_COMMIT, TrainPlan
from .data.contracts import SCHEMA, STATIC_OPERATIONS, ChatTokenizer
from .data.encoding import encode_pair, message_pair
from .merge import validate_merged_export
from .merge._contracts import MANIFEST


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tokenizer_for(checkpoint: Path) -> ChatTokenizer:
    return cast(ChatTokenizer, AutoTokenizer.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False,
    ))


def git_revision(root: Path, expected: str) -> str:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != expected:
        raise ValueError(f"{root}: expected {expected}, got {actual}")
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=root, text=True)
    if changed.strip():
        raise ValueError(f"tracked runtime source changed: {root}")
    return actual


def runtime_identity(miles_root: Path, megatron_root: Path, bridge_root: Path) -> JsonLikeDict:
    return {
        "revisions": {
            "miles": git_revision(miles_root, MILES_COMMIT),
            "megatron": git_revision(megatron_root, MEGATRON_REVISION),
            "bridge": git_revision(bridge_root, BRIDGE_REVISION),
        },
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "safetensors", "ray", "sglang", "transformer-engine")},
    }


def inspect_plan(plan: TrainPlan) -> JsonLikeDict:
    """Validate the complete merged base and reproduce every prepared token boundary."""
    if plan.output.exists():
        raise FileExistsError(f"fresh run output required: {plan.output}")
    manifest = validate_merged_export(plan.checkpoint)
    metadata = load_json_dict(plan.data / "metadata.json")
    if metadata.get("schema") != SCHEMA:
        raise ValueError("unsupported SFT dataset")
    source = plan.data / "input.jsonl"
    if file_hash(source) != metadata.get("input_sha256"):
        raise ValueError("prepared data hash differs")
    tokenizer = tokenizer_for(plan.checkpoint)
    template_hash = hashlib.sha256(json.dumps(tokenizer.chat_template, sort_keys=True).encode()).hexdigest()
    if template_hash != metadata.get("chat_template_sha256"):
        raise ValueError("prepared data uses a different chat template")
    rows = [as_dict(json.loads(line)) for line in source.read_text().splitlines()]
    if len(rows) != as_int(metadata.get("rows")) or len(rows) < plan.batch_size * plan.steps:
        raise ValueError("prepared row count differs or smoke would repeat examples")
    seen: set[str] = set()
    for row in rows:
        row_id = as_str(row.get("id"))
        if row_id in seen:
            raise ValueError("duplicate prepared row")
        seen.add(row_id)
        meta = as_dict(row.get("metadata"))
        if meta.get("split") != "train" or meta.get("operation") not in STATIC_OPERATIONS:
            raise ValueError("nonprimitive/nontraining row")
        encoded = encode_pair(tokenizer, message_pair(row.get("messages")), plan.max_tokens)
        if any(meta.get(key) != value for key, value in encoded.items()):
            raise ValueError(f"token/mask mismatch: {row_id}")
    ids = [as_int(value) for value in as_list(manifest["token_ids"])]
    if ids != list(range(248077, 248231)):
        raise ValueError("unexpected retained atlas IDs")
    return {
        "merged_manifest_sha256": file_hash(plan.checkpoint / MANIFEST),
        "dataset_sha256": file_hash(source), "dataset_metadata_sha256": file_hash(plan.data / "metadata.json"),
        "rows": len(rows), "presentations": plan.steps * plan.batch_size,
        "max_tokens": plan.max_tokens, "token_rows": "merged into frozen base",
        "initialization": "merged checkpoint + fresh fused language LoRA; fresh optimizer",
    }


def verify_import_location(name: str, root: Path) -> None:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{name} does not import from the pinned checkout {root}")


class BridgeInstance(Protocol):
    def to_megatron_provider(self, *, load_weights: bool) -> object: ...


class BridgeFactory(Protocol):
    def from_hf_pretrained(self, path: str, *, trust_remote_code: bool) -> BridgeInstance: ...


class BridgeModule(Protocol):
    AutoBridge: BridgeFactory


def inspect_backend(checkpoint: Path, megatron_root: Path, bridge_root: Path) -> JsonLikeDict:
    """Build the actual provider config without weights; TE imports require a CUDA driver."""
    verify_import_location("megatron.core", megatron_root)
    verify_import_location("megatron.bridge", bridge_root)
    module = cast(BridgeModule, importlib.import_module("megatron.bridge"))
    bridge = module.AutoBridge.from_hf_pretrained(str(checkpoint), trust_remote_code=False)
    provider = bridge.to_megatron_provider(load_weights=False)
    if type(provider).__name__ != "Qwen35VLModelProvider":
        raise ValueError(f"unexpected exact-model provider: {type(provider).__name__}")
    return {"bridge_factory": type(bridge).__name__, "provider": type(provider).__name__,
            "weights_loaded": False, "cuda_driver_required": True}
