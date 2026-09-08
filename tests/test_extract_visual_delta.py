import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from sft.scripts.extract_visual_delta import (
    canonical_visual_key, decompose_matrix, extract, matrix_kind, sha256_file,
    tensor_delta, visual_keys,
)


def test_key_mapping_does_not_require_same_peft_prefix():
    assert canonical_visual_key("base_model.model.model.visual.blocks.0.weight") == "model.visual.blocks.0.weight"
    assert canonical_visual_key("model.language_model.weight") is None
    with pytest.raises(ValueError, match="duplicate"):
        visual_keys(["model.visual.x", "base_model.model.visual.x"])


def test_base_cast_matches_trainer_and_prevents_false_delta():
    base = torch.tensor([1.001, 0.1001], dtype=torch.float32)
    trained = base.bfloat16().float()
    delta, row = tensor_delta(base, trained, "bfloat16")
    assert torch.equal(delta, torch.zeros_like(base))
    assert row["reconstruction_bit_exact"]
    assert row["changed_fraction"] == 0


def test_nonfinite_and_shapes_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        tensor_delta(torch.ones(1), torch.tensor([float("nan")]), "float32")
    with pytest.raises(ValueError, match="shape"):
        tensor_delta(torch.ones(1), torch.ones(2), "float32")


def test_dense_fp64_delta_survives_cancellation():
    original = torch.tensor([1.0, -0.25], dtype=torch.float32)
    current = torch.tensor([1e-6, -1e-6], dtype=torch.float32)
    assert not torch.equal(original + (current - original), current)
    delta, row = tensor_delta(original, current, "float32")
    assert delta.dtype == torch.float64
    assert row["reconstruction_bit_exact"]
    assert torch.equal((original.double() + delta).float(), current)


def test_extreme_exponent_gap_is_not_claimed_lossless():
    # FP64 is not a universal lossless difference format for arbitrary FP32s.
    # The extractor must check the actual round-trip, not assume it.
    _, row = tensor_delta(torch.tensor([1.0]), torch.tensor([1e-10]), "float32")
    assert not row["reconstruction_bit_exact"]
    assert row["reconstruction_max_abs_error"] > 0


def test_svd_factor_orientation_energy_and_bases():
    matrix = torch.diag(torch.tensor([4.0, 3.0, 0.0]))
    row, factors = decompose_matrix(matrix, 1)
    assert factors["lora_B"].shape == (3, 1)
    assert factors["lora_A"].shape == (1, 3)
    assert row["rank_energy_fraction"]["1"] == pytest.approx(16 / 25)
    assert row["energy_threshold_ranks"]["0.9"] == 2
    assert row["saved_factor_relative_frobenius_error"] == pytest.approx(3 / 5)
    assert row["output_basis_orthogonality_max_abs"] < 1e-6
    full, full_factors = decompose_matrix(matrix, 3)
    torch.testing.assert_close(full_factors["lora_B"] @ full_factors["lora_A"], matrix)
    assert full["saved_factor_relative_frobenius_error"] < 1e-6


def test_zero_matrix_and_non_lora_tensors():
    row, _ = decompose_matrix(torch.zeros(3, 2), 8)
    assert row["saved_factor_rank"] == 2
    assert row["energy_threshold_ranks"]["0.99"] == 0
    assert row["rank_energy_fraction"]["16"] == 1
    assert matrix_kind("model.visual.pos_embed.weight", torch.ones(5, 3)) == "embedding_matrix_not_linear_lora"
    assert matrix_kind("model.visual.patch_embed.proj.weight", torch.ones(2, 3, 2, 2, 2)) == "flattened_convolution_not_linear_lora"
    assert matrix_kind("model.visual.blocks.0.norm1.weight", torch.ones(3)) is None


def fixture_sources(tmp_path):
    original = {"model.visual.blocks.0.attn.proj.weight": torch.eye(3).bfloat16(),
                "model.visual.blocks.0.norm1.weight": torch.ones(3).bfloat16()}
    trained = {"base_model.model." + k: v.float() + 0.125 for k, v in original.items()}
    base = tmp_path / "base.safetensors"
    trained_path = tmp_path / "trained.safetensors"
    save_file(original, base)
    save_file(trained, trained_path)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"weight_map": {k: base.name for k in original}}))
    metadata = tmp_path / "hf.json"
    metadata.write_text(json.dumps({"id": "test/model", "sha": "abc", "siblings": [
        {"rfilename": base.name, "lfs": {"sha256": sha256_file(base)}}]}))
    return dict(trained_path=trained_path, base_index=index, base_dir=tmp_path,
                hf_info=metadata, output_dir=tmp_path / "result", revision="abc",
                model_id="test/model", factor_rank=2)


def test_end_to_end_exact_dense_delta_and_no_source_mutation(tmp_path):
    args = fixture_sources(tmp_path)
    before = sha256_file(args["trained_path"])
    report = extract(**args)
    assert report["status"] == "complete"
    assert report["summary"]["all"]["tensor_count"] == 2
    assert report["summary"]["all"]["matrix_count"] == 1
    assert sha256_file(args["trained_path"]) == before
    delta = load_file(args["output_dir"] / "visual_delta.safetensors")
    for tensor in delta.values():
        torch.testing.assert_close(tensor, torch.full_like(tensor, 0.125))
    with pytest.raises(FileExistsError):
        extract(**args)


def test_provenance_and_checkpoint_precision_validation(tmp_path):
    args = fixture_sources(tmp_path)
    with pytest.raises(ValueError, match="metadata"):
        extract(**{**args, "revision": "wrong"})
    with pytest.raises(ValueError, match="trained visual SHA256"):
        extract(**args, trained_sha256="wrong")
    trained = load_file(args["trained_path"])
    save_file({k: v.bfloat16() for k, v in trained.items()}, args["trained_path"])
    with pytest.raises(ValueError, match="FP32"):
        extract(**args)
