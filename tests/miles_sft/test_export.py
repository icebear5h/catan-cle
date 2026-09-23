"""Real shard composition: admitted language updates, exact frozen tensors/assets."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from sft.miles_sft.export import assemble_complete_export, validate_complete_export
from sft.miles_sft.export._validation import COMPOSITION, SCHEMA
from sft.miles_sft.merge import merge_checkpoint
from sft.miles_sft.merge._contracts import (
    COMPLETE,
    INDEX,
    MANIFEST,
    file_identity,
    read_json,
    sha256,
)
from sft.miles_sft.runtime.artifacts import audit_hf_export
from tests.miles_sft.test_merge import MTP, Q, checkpoint, write
from tests.miles_sft.test_merge import bundle as merge_fixture

bundle = merge_fixture
NORM = "model.language_model.norm.weight"


def _save_bridge(directory: Path, state: dict[str, torch.Tensor]) -> None:
    weights: dict[str, str] = {}
    # Different partitioning from the two base shards exercises per-key routing.
    for i in range(3):
        shard = {key: state[key] for key in sorted(state)[i::3]}
        name = f"bridge-{i}.safetensors"
        save_file(shard, directory / name)
        weights.update(dict.fromkeys(shard, name))
    write(directory / INDEX, {"weight_map": weights, "metadata": {
        "total_size": sum(t.numel() * t.element_size() for t in state.values())}})


@pytest.fixture
def export_bundle(bundle: tuple[Path, Path, Path], tmp_path: Path) -> tuple[Path, Path, Path]:
    original, adapter, base = bundle
    state = load_file(original / "source-0.safetensors")
    state[NORM] = torch.tensor([-0.0, 1.0003, -2.0003])
    save_file(state, original / "source-0.safetensors")
    index = read_json(original / INDEX)
    index["weight_map"][NORM] = "source-0.safetensors"
    index["metadata"]["total_size"] += state[NORM].numel() * state[NORM].element_size()
    write(original / INDEX, index)
    merge_checkpoint(original, adapter, base)
    bridge = tmp_path / "exports" / "rollout-0" / "bridge"
    bridge.mkdir(parents=True)
    # Reproduce the actual failure: absent MTP, rounded FP32 vision, reserialized
    # tokenizer. Also perturb every frozen matrix to ensure none is admitted.
    exported = {key: (tensor + 0.5).bfloat16() for key, tensor in checkpoint(base).items() if key != MTP}
    _save_bridge(bridge, exported)
    (bridge / "tokenizer.json").write_text('{"reserialized": true}\n')
    (bridge / "config.json").write_text('{"torch_dtype": "bfloat16"}\n')
    (bridge / ".complete").touch()
    return base, bridge, bridge.parent / "model"


def test_complete_export_preserves_all_frozen_bytes_and_assets(export_bundle: tuple[Path, Path, Path]) -> None:
    base, bridge, output = export_bundle
    base_manifest = read_json(base / MANIFEST)
    raw_before = {p.name: file_identity(p) for p in bridge.iterdir()}
    result = assemble_complete_export(base, bridge, output)
    assert result == output / COMPOSITION
    manifest = validate_complete_export(base, output, bridge_export=bridge)
    trained = {key for key, entry in base_manifest["tensors"].items()
               if entry["operation"] == "lora_fp32_then_bf16"}
    original, raw, actual = checkpoint(base), checkpoint(bridge), checkpoint(output)
    assert actual.keys() == original.keys()
    assert MTP not in raw and MTP in actual
    for key, tensor in actual.items():
        expected = raw[key] if key in trained else original[key]
        assert tensor.dtype == expected.dtype
        assert torch.equal(tensor.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)), key
        entry = manifest["tensors"][key]
        assert entry["sha256"] == entry["source_sha256"]
        assert entry["source"] == ("bridge" if key in trained else "base")
    assert actual[NORM].dtype == torch.float32
    assert actual["model.visual.blocks.0.weight"].dtype == torch.float32
    assert not torch.equal(actual[Q + ".weight"], original[Q + ".weight"])
    for name in base_manifest["assets"]:
        assert (output / name).read_bytes() == (base / name).read_bytes()
    assert not (output / MANIFEST).exists() and not (output / COMPLETE).exists()
    assert raw_before == {p.name: file_identity(p) for p in bridge.iterdir()}
    audit = audit_hf_export(base, output, bridge_export=bridge)
    assert audit["frozen_tensor_count"] == len(original) - len(trained)
    assert audit["composition_manifest_sha256"] == sha256(result)
    with pytest.raises(FileExistsError):
        assemble_complete_export(base, bridge, output)
    assert raw_before == {p.name: file_identity(p) for p in bridge.iterdir()}


@pytest.mark.parametrize(("damage", "message"), [
    ("missing", "missing trained"), ("extra", "outside base"),
    ("shape", "shape/dtype"), ("dtype", "shape/dtype"), ("index", "index/shard"),
    ("incomplete", "missing/nonlocal"),
])
def test_invalid_bridge_rejected_before_output(export_bundle: tuple[Path, Path, Path],
                                               damage: str, message: str) -> None:
    base, bridge, output = export_bundle
    state = checkpoint(bridge)
    if damage == "missing":
        state.pop(Q + ".weight")
    elif damage == "extra":
        state["unexpected.weight"] = torch.ones(2)
    elif damage == "shape":
        state[Q + ".weight"] = state[Q + ".weight"][:1]
    elif damage == "dtype":
        state[Q + ".weight"] = state[Q + ".weight"].float()
    _save_bridge(bridge, state)
    if damage == "index":
        index = read_json(bridge / INDEX)
        index["weight_map"].pop(Q + ".weight")
        write(bridge / INDEX, index)
    elif damage == "incomplete":
        (bridge / ".complete").rename(bridge / "unfinished")
    before = {p.name: file_identity(p) for p in bridge.iterdir()}
    with pytest.raises(ValueError, match=message):
        assemble_complete_export(base, bridge, output)
    assert not output.exists()
    assert before == {p.name: file_identity(p) for p in bridge.iterdir()}


def test_changed_base_never_seals_partial_export(export_bundle: tuple[Path, Path, Path]) -> None:
    base, bridge, output = export_bundle
    index = read_json(base / INDEX)
    path = base / index["weight_map"][NORM]
    state = load_file(path)
    state[NORM][1] += 1
    save_file(state, path)
    with pytest.raises(ValueError, match="base shard hash/size"):
        assemble_complete_export(base, bridge, output)
    assert not (output / ".complete").exists()
    with pytest.raises(FileExistsError):
        assemble_complete_export(base, bridge, output)


@pytest.mark.parametrize("damage", ["frozen_tensor", "tokenizer", "bridge"])
def test_audit_binds_tensor_and_asset_bytes(export_bundle: tuple[Path, Path, Path], damage: str) -> None:
    base, bridge, output = export_bundle
    assemble_complete_export(base, bridge, output)
    if damage == "tokenizer":
        (output / "tokenizer.json").write_text("{}")
    else:
        root = bridge if damage == "bridge" else output
        name = Q + ".weight" if damage == "bridge" else NORM
        path = root / read_json(root / INDEX)["weight_map"][name]
        state = load_file(path)
        state[name].reshape(-1)[0] = 5
        save_file(state, path)
    with pytest.raises(ValueError, match="hash/size"):
        audit_hf_export(base, output, bridge_export=bridge)


@pytest.mark.parametrize("damage", ["source", "identity", "dtype", "coverage"])
def test_resealed_metadata_cannot_weaken_composition(export_bundle: tuple[Path, Path, Path], damage: str) -> None:
    base, bridge, output = export_bundle
    assemble_complete_export(base, bridge, output)
    digest = sha256(output / COMPOSITION)
    manifest = read_json(output / COMPOSITION)
    if damage == "source":
        manifest["tensors"][NORM]["source"] = "bridge"
    elif damage == "identity":
        manifest["tensors"][NORM]["sha256"] = "0" * 64
    elif damage == "dtype":
        manifest["tensors"][NORM]["dtype"] = "BF16"
    else:
        manifest["tensors"].pop(MTP)
    write(output / COMPOSITION, manifest)
    write(output / ".complete", {"schema": SCHEMA, "manifest_sha256": sha256(output / COMPOSITION)})
    with pytest.raises(ValueError, match="pinned composition"):
        validate_complete_export(base, output, expected_sha256=digest)
    with pytest.raises(ValueError, match="composition (frozen/trained identity|tensor)"):
        validate_complete_export(base, output)


def test_output_must_be_fresh_sibling_model(export_bundle: tuple[Path, Path, Path]) -> None:
    base, bridge, output = export_bundle
    with pytest.raises(FileExistsError):
        assemble_complete_export(base, bridge, bridge)
    with pytest.raises(ValueError, match="sibling model"):
        assemble_complete_export(base, bridge, output.parent / "elsewhere")
    assert not output.exists()
