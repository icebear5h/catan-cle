"""Streaming r04 HF export for a functional warmstart with fresh Megatron LoRA.

``merge_checkpoint(base, adapter, output)`` returns the identity manifest path.
IDs come from both PEFT trainable_token_indices mappings, the saved semantic
inventory, or an explicit ordered 154-ID tuple; supplied sources must agree.
The output retains full vocabulary/MTP state and is a frozen-base initialization,
not an exact-factor or optimizer continuation. Checkpoints do not encode training
requires_grad flags: the consuming trainer must freeze this base, including rows.

Memory is bounded by one base shard, its serialized output, and per-matrix CPU
workspaces. A failed export may leave an incomplete new directory; it cannot
overwrite an existing directory or produce a valid completion marker.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ._contracts import (
    COMPLETE,
    INDEX,
    MANIFEST,
    SCHEMA,
    identities,
    require,
    sha256,
    write_json,
)
from ._manifest import (
    build_manifest,
    load_manifest,
    output_size,
    validate_merged_export,
    validate_payload,
)
from ._preflight import preflight
from ._stream import stream_shards

__all__ = ["load_manifest", "merge_checkpoint", "validate_merged_export"]


def merge_checkpoint(base_dir: Path, adapter_dir: Path, output: Path,
                     token_ids: tuple[int, ...] = (), *, base_revision: str | None = None) -> Path:
    """Validate locally, export fresh BF16 language/FP32 vision shards, and seal last.

    ``base_revision`` is caller-declared provenance (e.g. a pinned HF commit).
    All consumed source files also have SHA256 identities, checked again before
    completion. There is no fallback for missing shards, rows, vision or assets.
    """
    base = base_dir.expanduser().resolve()
    adapter = adapter_dir.expanduser().resolve()
    output = output.expanduser().absolute()
    require(base_revision is None or bool(base_revision.strip()), "base_revision cannot be empty")
    plan = preflight(base, adapter, output, token_ids)
    output.mkdir(parents=True, exist_ok=False)
    weights, headers, rounding = stream_shards(base, adapter, output, plan)
    write_json(output / INDEX, {"metadata": {"total_size": output_size(headers)}, "weight_map": weights})
    for name, source in sorted(plan.assets.items()):
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 << 20)
    files = identities(output, set(weights.values()) | set(plan.assets) | {INDEX})
    for name, path in plan.assets.items():
        original = plan.base_files[name] if path.parent == base else plan.adapter_files[name]
        require(files[name] == original, f"inference asset changed during copy: {name}")
    manifest = build_manifest(base, adapter, plan, base_revision, weights, headers, rounding, files)
    write_json(output / MANIFEST, manifest)
    validate_payload(output, manifest)
    require(identities(base, set(plan.base_files)) == plan.base_files, "base changed during export")
    require(identities(adapter, set(plan.adapter_files)) == plan.adapter_files, "adapter changed during export")
    # The only success marker is written after every tensor, asset and provenance check.
    write_json(output / COMPLETE, {"schema": SCHEMA, "manifest_sha256": sha256(output / MANIFEST)})
    return output / MANIFEST
