"""Content and source-selection checks for complete, composed HF exports."""

from __future__ import annotations

from pathlib import Path

from sft.miles_sft.merge import load_manifest
from sft.miles_sft.merge._checkpoint import inspect_checkpoint, tensor_bytes, tensor_headers
from sft.miles_sft.merge._contracts import (
    INDEX,
    MANIFEST,
    Checkpoint,
    file_identity,
    object_map,
    read_json,
    relative_name,
    require,
    sha256,
    text,
)

COMPOSITION = "composition_manifest.json"
COMPLETE = ".complete"
SCHEMA = "catan_miles_composed_checkpoint/v1"


def local_file(root: Path, name: str) -> Path:
    path = root / relative_name(name)
    require(path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(root.resolve()), f"missing/nonlocal export file: {name}")
    return path


def bridge_checkpoint(root: Path) -> Checkpoint:
    """Inspect the raw Bridge payload, allowing missing frozen tensors and rounded vision."""
    local_file(root, COMPLETE)
    index = read_json(local_file(root, INDEX))
    weights = {key: relative_name(value) for key, value in object_map(index.get("weight_map")).items()}
    require(bool(weights), "empty Bridge weight index")
    names = set(weights.values())
    require(all(name.endswith(".safetensors") for name in names), "non-safetensors Bridge shard")
    require(names == {p.relative_to(root).as_posix() for p in root.rglob("*.safetensors")},
            "Bridge shards differ from index")
    headers = {}
    shards = {}
    for name in sorted(names):
        found = tensor_headers(local_file(root, name))
        require(set(found) == {key for key, shard in weights.items() if shard == name},
                f"Bridge index/shard tensor mismatch: {name}")
        headers.update(found)
        shards[name] = tuple(sorted(found))
    size = sum(tensor_bytes(header) for header in headers.values())
    declared_size = object_map(index.get("metadata")).get("total_size")
    require(type(declared_size) in (int, float) and declared_size == size,
            "Bridge index total_size mismatch")
    return Checkpoint(weights, headers, shards, size)


def base_inventory(base: Path) -> tuple[dict[str, object], Checkpoint, set[str], set[str]]:
    manifest = load_manifest(base)
    checkpoint = inspect_checkpoint(base)
    tensors = object_map(manifest.get("tensors"))
    require(set(tensors) == set(checkpoint.headers), "base manifest tensor coverage mismatch")
    for key, header in checkpoint.headers.items():
        entry = object_map(tensors[key])
        require(all(entry.get(field) == value for field, value in header.items())
                and entry.get("shard") == checkpoint.weight_map[key], f"base tensor metadata mismatch: {key}")
    trained = {key for key, entry in tensors.items()
               if object_map(entry).get("operation") == "lora_fp32_then_bf16"}
    require(bool(trained), "base merge manifest has no trained keys")
    assets = set(object_map(manifest.get("assets")))
    files = object_map(manifest.get("output_files"))
    require(assets == set(files) - set(checkpoint.shards) - {INDEX}, "base asset inventory mismatch")
    require(not assets & {MANIFEST, "MERGE_COMPLETE.json", COMPOSITION, COMPLETE},
            "base assets contain completion/provenance files")
    for name in files:
        local_file(base, name)
    require(file_identity(base / INDEX) == files[INDEX], "base index hash mismatch")
    return manifest, checkpoint, trained, assets


def admit_bridge(base: Checkpoint, bridge: Checkpoint, trained: set[str]) -> None:
    require(set(bridge.headers) <= set(base.headers), "unexpected Bridge tensor keys outside base")
    require(trained <= set(bridge.headers), "Bridge export missing trained keys")
    for key in trained:
        require(bridge.headers[key] == base.headers[key], f"Bridge trained shape/dtype mismatch: {key}")


def _digest(value: object) -> str:
    digest = text(value)
    require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "invalid tensor SHA256")
    return digest


def validate_payload(base: Path, output: Path, manifest: dict[str, object],
                     bridge_export: Path | None = None) -> None:
    require(manifest.get("schema") == SCHEMA and manifest.get("status") == "completed",
            "unsupported/incomplete composition manifest")
    original, checkpoint, trained, assets = base_inventory(base)
    require(manifest.get("base") == str(base.resolve())
            and manifest.get("base_manifest_sha256") == sha256(base / MANIFEST)
            and manifest.get("base_files") == original["output_files"], "composition base identity mismatch")
    require(manifest.get("trained_keys") == sorted(trained), "composition trained-key inventory mismatch")
    bridge_root = Path(text(manifest.get("bridge_export")))
    require(output.resolve() == bridge_root.resolve().parent / "model"
            and output.resolve() != bridge_root.resolve(), "composition output must be sibling model")
    if bridge_export is not None:
        require(bridge_root == bridge_export.resolve(), "composition Bridge path mismatch")
    bridge = bridge_checkpoint(bridge_root)
    admit_bridge(checkpoint, bridge, trained)
    bridge_files = object_map(manifest.get("bridge_files"))
    require(set(bridge_files) == set(bridge.shards) | {INDEX, COMPLETE}, "Bridge file inventory mismatch")
    for name, identity in bridge_files.items():
        require(file_identity(local_file(bridge_root, name)) == identity, f"Bridge hash/size mismatch: {name}")
    files = object_map(manifest.get("output_files"))
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    require(actual - {COMPOSITION, COMPLETE} == set(files), "composition file inventory mismatch")
    for name, identity in files.items():
        require(file_identity(local_file(output, name)) == identity, f"composition hash/size mismatch: {name}")
    final = inspect_checkpoint(output)
    require(final.headers == checkpoint.headers, "composition tensor coverage/shape/dtype mismatch")
    require(set(files) == set(final.shards) | assets | {INDEX}, "composition output inventory mismatch")
    require(manifest.get("assets") == sorted(assets), "composition asset inventory mismatch")
    source_files = object_map(original["output_files"])
    for name in assets:
        require(files[name] == source_files[name], f"composition changed base asset: {name}")
    tensors = object_map(manifest.get("tensors"))
    require(set(tensors) == set(checkpoint.headers), "composition tensor inventory mismatch")
    for key, header in final.headers.items():
        entry = object_map(tensors[key])
        source = bridge if key in trained else checkpoint
        require(all(entry.get(field) == value for field, value in header.items())
                and entry.get("shard") == final.weight_map[key], f"composition tensor metadata mismatch: {key}")
        require(entry.get("source") == ("bridge" if key in trained else "base")
                and entry.get("source_shard") == source.weight_map[key]
                and _digest(entry.get("sha256")) == _digest(entry.get("source_sha256")),
                f"composition frozen/trained identity mismatch: {key}")


def validate_complete_export(base: Path, output: Path, *, bridge_export: Path | None = None,
                             expected_sha256: str | None = None) -> dict[str, object]:
    """Verify source selection and all export hashes; no full base rehash at finalization.

    Tensor byte identity is established while composing and bound by the sealed
    manifest and shard hashes. Pin the manifest digest from the post-save receipt
    to detect rewriting both payload and seal. Source paths must remain available.
    """
    digest = sha256(local_file(output, COMPOSITION))
    marker = read_json(local_file(output, COMPLETE))
    require(marker == {"schema": SCHEMA, "manifest_sha256": digest}, "invalid composition completion seal")
    require(expected_sha256 is None or digest == expected_sha256, "pinned composition manifest differs")
    manifest = read_json(output / COMPOSITION)
    validate_payload(base, output, manifest, bridge_export)
    return manifest
