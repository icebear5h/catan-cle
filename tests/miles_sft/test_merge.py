"""Deterministic real safetensors exports; no model/GPU/network instantiation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from sft.miles_sft.merge import load_manifest, merge_checkpoint, validate_merged_export
from sft.miles_sft.merge.__main__ import main
from sft.miles_sft.merge._contracts import (
    ADAPTER,
    COMPLETE,
    INDEX,
    MANIFEST,
    PREFIX,
    ROW_MODULES,
    ROW_SUFFIX,
    VISUAL,
    read_json,
    sha256,
)

IDS = tuple(reversed(range(100, 254)))
Q = "model.language_model.layers.0.self_attn.q_proj"
K = "model.language_model.layers.0.self_attn.k_proj"
MTP = "mtp.layers.0.self_attn.q_proj.weight"


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    base, adapter, output = (tmp_path / name for name in ("base", "adapter", "merged"))
    base.mkdir()
    adapter.mkdir()
    config = {"text_config": {"vocab_size": 248320, "tie_word_embeddings": False},
              "tie_word_embeddings": False, "torch_dtype": "bfloat16"}
    write(base / "config.json", config)
    write(base / "generation_config.json", {"eos_token_id": 2})
    write(base / "preprocessor_config.json", {"size": {"longest_edge": 4096}})
    write(adapter / "preprocessor_config.json", {"size": {"longest_edge": 1}})
    write(adapter / "config.json", {"vocab_size": 254})
    write(adapter / "adapter_config.json", {
        "peft_type": "LORA", "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "bias": "none", "use_rslora": False, "use_dora": False,
        "target_modules": ["q_proj", "k_proj"],
        "trainable_token_indices": {module: list(IDS) for module in ROW_MODULES},
    })
    write(adapter / "tokenizer_config.json", {"chat_template": "{{ messages }}"})
    write(adapter / "tokenizer.json", {
        "model": {"type": "BPE", "vocab": {f"token{i}": i for i in range(254)}, "merges": []},
        "added_tokens": [{"id": i, "content": f"token{i}", "special": False} for i in IDS],
    })
    (adapter / "chat_template.jinja").write_text("{{ messages }}")
    (adapter / "chat_templates").mkdir()
    (adapter / "chat_templates" / "tools.jinja").write_text("{{ tools }}")
    state: dict[str, torch.Tensor] = {
        module + ".weight": torch.full((248320, 4), float(i + 1), dtype=torch.bfloat16)
        for i, module in enumerate(ROW_MODULES)
    }
    state.update({Q + ".weight": torch.full((1030, 4), 0.25, dtype=torch.bfloat16),
                  K + ".weight": torch.full((1030, 4), -0.5, dtype=torch.bfloat16),
                  MTP: torch.tensor([[1.0003, -2.0003]], dtype=torch.float32)})
    factors: dict[str, torch.Tensor] = {}
    for i, module in enumerate((Q, K)):
        a = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 512
        factors[PREFIX + module + ".lora_A.weight"] = a.flip(0) if i else a
        factors[PREFIX + module + ".lora_B.weight"] = (
            torch.arange(1030 * 16, dtype=torch.float32).reshape(1030, 16) % 17 - 8) / 256
    for i, module in enumerate(ROW_MODULES):
        factors[PREFIX + module + ROW_SUFFIX] = (
            torch.arange(154 * 4, dtype=torch.float32).reshape(154, 4) / 101 + i + 0.0003)
    visual = {PREFIX + f"model.visual.blocks.{i}.weight": torch.tensor([1.0003 + i / 1000, -0.0])
              for i in range(333)}
    state.update({key.removeprefix(PREFIX): value.bfloat16() for key, value in visual.items()})
    names = sorted(state)
    shards = [dict((key, state[key]) for key in names[i::2]) for i in range(2)]
    weight_map: dict[str, str] = {}
    for i, shard in enumerate(shards):
        name = f"source-{i}.safetensors"
        save_file(shard, base / name)
        weight_map.update(dict.fromkeys(shard, name))
    write(base / INDEX, {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())},
                         "weight_map": weight_map})
    save_file(factors, adapter / ADAPTER)
    save_file(visual, adapter / VISUAL)
    return base, adapter, output


def checkpoint(root: Path) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for path in sorted(root.glob("*.safetensors")):
        result.update(load_file(path))
    return result


def test_merge_real_shards_independent_factors_rows_visual_and_mtp(bundle: tuple[Path, Path, Path]) -> None:
    base, adapter, output = bundle
    original, factors = checkpoint(base), load_file(adapter / ADAPTER)
    # An ambient autocast must not change the export arithmetic.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        report_path = merge_checkpoint(base, adapter, output, IDS, base_revision="pinned-hf-commit")
    merged = checkpoint(output)
    assert report_path == output / MANIFEST
    assert merged.keys() == original.keys()
    for module in (Q, K):
        a, b = (factors[PREFIX + module + f".lora_{letter}.weight"] for letter in ("A", "B"))
        expected = (original[module + ".weight"].float() + 2 * (b @ a)).bfloat16()
        assert torch.equal(merged[module + ".weight"], expected)
    wrong = (original[K + ".weight"].float() + 2 * (
        factors[PREFIX + K + ".lora_B.weight"] @ factors[PREFIX + Q + ".lora_A.weight"])).bfloat16()
    assert not torch.equal(merged[K + ".weight"], wrong)
    for module in ROW_MODULES:
        key = module + ".weight"
        expected = original[key].clone()
        expected[list(IDS)] = factors[PREFIX + module + ROW_SUFFIX].bfloat16()
        assert merged[key].shape == (248320, 4)
        assert torch.equal(merged[key], expected)
    for name, tensor in load_file(adapter / VISUAL).items():
        actual = merged[name.removeprefix(PREFIX)]
        assert actual.dtype == torch.float32
        assert torch.equal(actual.view(torch.uint8), tensor.view(torch.uint8))
    assert torch.equal(merged[MTP].view(torch.uint8), original[MTP].view(torch.uint8))
    for name in ("config.json", "generation_config.json", "preprocessor_config.json"):
        assert (output / name).read_bytes() == (base / name).read_bytes()
    assert (output / "chat_templates/tools.jinja").read_bytes() == (adapter / "chat_templates/tools.jinja").read_bytes()
    manifest = validate_merged_export(output, expected_sha256=sha256(report_path))
    assert manifest["declared_base_revision"] == "pinned-hf-commit"
    assert manifest["rounding"][Q + ".weight"]["rounded_elements"] > 0
    assert manifest["semantics"]["exact_factor_continuation"] is False
    assert manifest["tensors"][MTP]["operation"] == "preserved"
    assert not (output / ADAPTER).exists()
    assert not (output / VISUAL).exists()


@pytest.mark.parametrize("damage", ["unexpected", "missing_b", "missing_visual", "visual_dtype",
                                   "visual_shape", "extra_visual", "missing_shard", "undeclared", "wrong_rows"])
def test_invalid_tensors_fail_before_output(bundle: tuple[Path, Path, Path], damage: str) -> None:
    base, adapter, output = bundle
    path = adapter / (VISUAL if "visual" in damage else ADAPTER)
    state = load_file(path)
    if damage == "unexpected":
        state["unexpected.weight"] = torch.ones(1)
    elif damage == "missing_b":
        del state[PREFIX + Q + ".lora_B.weight"]
    elif damage == "missing_visual":
        del state[next(iter(state))]
    elif damage == "visual_dtype":
        state[next(iter(state))] = state[next(iter(state))].bfloat16()
    elif damage == "visual_shape":
        state[next(iter(state))] = torch.ones(3)
    elif damage == "extra_visual":
        state["wrong.weight"] = torch.ones(2)
    elif damage == "missing_shard":
        (base / "source-0.safetensors").rename(base / "missing.bin")
    elif damage == "undeclared":
        save_file({"hidden.weight": torch.ones(1)}, base / "extra.safetensors")
    elif damage == "wrong_rows":
        state[PREFIX + ROW_MODULES[0] + ROW_SUFFIX] = torch.ones(153, 4)
    save_file(state, path)
    with pytest.raises((ValueError, FileNotFoundError)):
        merge_checkpoint(base, adapter, output)
    assert not output.exists()


@pytest.mark.parametrize(("key", "value"), [("r", 8), ("r", True), ("lora_alpha", 16),
    ("lora_dropout", 0.0), ("bias", "all"), ("use_rslora", True), ("use_dora", True),
    ("rank_pattern", {"q_proj": 8}), ("modules_to_save", ["lm_head"]), ("fan_in_fan_out", True)])
def test_reject_nonstandard_lora(bundle: tuple[Path, Path, Path], key: str, value: object) -> None:
    base, adapter, output = bundle
    config = read_json(adapter / "adapter_config.json")
    config[key] = value
    write(adapter / "adapter_config.json", config)
    with pytest.raises(ValueError):
        merge_checkpoint(base, adapter, output)
    assert not output.exists()


def test_explicit_ids_inventory_and_cli(bundle: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    base, adapter, output = bundle
    config = read_json(adapter / "adapter_config.json")
    config.pop("trainable_token_indices")
    write(adapter / "adapter_config.json", config)
    with pytest.raises(ValueError, match="missing token IDs"):
        merge_checkpoint(base, adapter, output)
    inventory = adapter / "trainable_parameters.json"
    write(inventory, {"semantic_tokens": {"token_ids": list(IDS)}})
    main(["merge", "--base-dir", str(base), "--adapter-dir", str(adapter), "--output", str(output),
          "--token-inventory", str(inventory)])
    assert str(output / MANIFEST) in capsys.readouterr().out
    main(["validate", str(output)])
    assert json.loads(capsys.readouterr().out)["token_ids"] == list(IDS)


@pytest.mark.parametrize("ids", [IDS[:-1], IDS[:-1] + (IDS[0],), tuple(reversed(IDS)),
                                 IDS[:-1] + (248320,), IDS[:-1] + (True,)])
def test_token_ids_fail_closed(bundle: tuple[Path, Path, Path], ids: tuple[int, ...]) -> None:
    base, adapter, output = bundle
    with pytest.raises(ValueError):
        merge_checkpoint(base, adapter, output, ids)
    assert not output.exists()


def test_fresh_output_completion_and_tamper_detection(bundle: tuple[Path, Path, Path]) -> None:
    base, adapter, output = bundle
    merge_checkpoint(base, adapter, output)
    digest = sha256(output / MANIFEST)
    with pytest.raises(FileExistsError):
        merge_checkpoint(base, adapter, output)
    assert sha256(output / MANIFEST) == digest
    with pytest.raises(ValueError, match="pinned"):
        load_manifest(output, expected_sha256="0" * 64)
    (output / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ValueError, match="hash/size"):
        validate_merged_export(output)
    (output / COMPLETE).rename(output / "interrupted.json")
    with pytest.raises(FileNotFoundError):
        validate_merged_export(output)


@pytest.mark.parametrize("value", [float("nan"), 3.4e38])
def test_nonfinite_arithmetic_never_completes(bundle: tuple[Path, Path, Path], value: float) -> None:
    base, adapter, output = bundle
    state = load_file(adapter / ADAPTER)
    # NaNs and finite FP32 values that overflow the final BF16 row cast both fail.
    state[PREFIX + ROW_MODULES[1] + ROW_SUFFIX][0, 0] = value
    save_file(state, adapter / ADAPTER)
    with pytest.raises(ValueError, match="nonfinite"):
        merge_checkpoint(base, adapter, output)
    assert not (output / COMPLETE).exists()
    with pytest.raises(FileExistsError):
        merge_checkpoint(base, adapter, output)


@pytest.mark.parametrize("damage", ["tied", "vocab", "index", "asset", "tokenizer"])
def test_reject_incompatible_metadata(bundle: tuple[Path, Path, Path], damage: str) -> None:
    base, adapter, output = bundle
    if damage in ("tied", "vocab"):
        config = read_json(base / "config.json")
        config["tie_word_embeddings" if damage == "tied" else "vocab_size"] = True if damage == "tied" else 254
        write(base / "config.json", config)
    elif damage == "index":
        config = read_json(base / INDEX)
        config["metadata"] = {"total_size": 1}
        write(base / INDEX, config)
    elif damage == "asset":
        (adapter / "tokenizer.json").rename(adapter / "tokenizer.missing")
    else:
        config = read_json(adapter / "tokenizer.json")
        config["added_tokens"][0]["special"] = True
        write(adapter / "tokenizer.json", config)
    with pytest.raises(ValueError):
        merge_checkpoint(base, adapter, output)
    assert not output.exists()


def test_resealed_manifest_must_still_have_valid_provenance(bundle: tuple[Path, Path, Path]) -> None:
    base, adapter, output = bundle
    merge_checkpoint(base, adapter, output)
    manifest = read_json(output / MANIFEST)
    manifest["adapter_tensor_headers"].pop(PREFIX + Q + ".lora_A.weight")
    write(output / MANIFEST, manifest)
    marker = read_json(output / COMPLETE)
    marker["manifest_sha256"] = sha256(output / MANIFEST)
    write(output / COMPLETE, marker)
    with pytest.raises(ValueError, match="adapter keys"):
        validate_merged_export(output)
