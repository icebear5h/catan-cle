"""CPU tensor/serialization contracts; Miles integration uses only real Miles types."""

from __future__ import annotations

import importlib
import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sft.json_types import load_json_dict
from sft.miles_sft.export import assemble_complete_export
from sft.miles_sft.runtime.admission import encoded_sample, validate_args
from sft.miles_sft.runtime.artifacts import audit_hf_export, audit_native
from sft.miles_sft.runtime.audits import adapter_parameters, audit_fresh_adapter, gradient_witness
from sft.miles_sft.runtime.contracts import MILES_COMMIT, REQUIRED_TARGETS, TARGET_SUFFIXES
from sft.miles_sft.runtime.receipts import finalize_run
from sft.miles_sft.runtime.storage import file_hash, write_receipt
from sft.miles_sft.runtime.witness import audited_train_step
from tests.miles_sft import test_export

bundle = test_export.bundle
export_bundle = test_export.export_bundle


class Outcome(Enum):
    NORMAL = "normal"


def _model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.vision_model = torch.nn.Linear(4, 4)
    language = torch.nn.Module()
    language.word_embeddings = torch.nn.Embedding(32, 4)
    language.output_layer = torch.nn.Linear(4, 32)
    language.decoder = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attention = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    for suffix in TARGET_SUFFIXES:
        parent, name = suffix.split(".")
        projection = torch.nn.Module()
        projection.base = torch.nn.Linear(4, 4)
        projection.adapter = torch.nn.Module()
        projection.adapter.linear_in = torch.nn.Linear(4, 16, bias=False)
        projection.adapter.linear_out = torch.nn.Linear(16, 4, bias=False)
        torch.nn.init.constant_(projection.adapter.linear_in.weight, 0.125)
        torch.nn.init.zeros_(projection.adapter.linear_out.weight)
        getattr(layer, parent).add_module(name, projection)
    language.decoder.layers = torch.nn.ModuleList([layer])
    model.language_model = language
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if ".adapter." in name:
            param.requires_grad_(True)
    return model


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        rollout_global_dataset=True, rollout_batch_size=1, n_samples_per_prompt=1,
        apply_chat_template=False, input_key="messages", metadata_key="metadata",
        rollout_max_prompt_len=None, seq_length=64, train_backend="megatron",
        megatron_to_hf_mode="bridge", debug_train_only=True, debug_rollout_only=False,
        debug_disable_optimizer=False, lora_train_only=True, lora_type="lora",
        lora_rank=16, lora_alpha=32, lora_dropout=0.0, lora_adapter_path=None,
        target_modules=REQUIRED_TARGETS, exclude_modules=None, loss_type="sft_loss",
        calculate_per_token_loss=True, compute_advantages_and_returns=False,
        enable_mtp_training=False, use_critic=False, actor_num_nodes=1, actor_num_gpus_per_node=1,
        rollout_num_gpus=0, eval_num_gpus=0, tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1, context_parallel_size=1, expert_model_parallel_size=1,
        expert_tensor_parallel_size=1, virtual_pipeline_model_parallel_size=None,
        rank=0, world_size=1, hf_checkpoint=str(tmp_path / "base"), load=str(tmp_path / "base"),
        save=str(tmp_path / "native"), save_hf=str(tmp_path / "exports" / "rollout-{rollout_id}" / "bridge"),
        num_rollout=1, start_rollout_id=0, global_batch_size=1, no_save_optim=False,
    )


def _sample() -> SimpleNamespace:
    return SimpleNamespace(
        prompt=[{"role": "user", "content": "Neighbors of <N00>?"},
                {"role": "assistant", "content": "<N01>"}],
        metadata={"tokens": [1, 2, 3, 4, 5], "prefix_length": 3, "response_length": 2,
                  "loss_mask": [1, 1], "split": "train", "operation": "symbolic_neighbors",
                  "source_metadata": {"split": "train", "operation": "symbolic_neighbors"},
                  "source_file": "/source.jsonl", "source_line": 1,
                  "source_sha256": "a" * 64, "source_row_sha256": "b" * 64},
        multimodal_inputs=None, multimodal_train_inputs=None,
    )


def test_encoded_suffix_is_unshifted_and_rejects_bad_boundaries() -> None:
    sample = _sample()
    encoded, answer = encoded_sample(sample, 5)
    assert encoded["tokens"] == [1, 2, 3, 4, 5] and answer == "<N01>"
    assert encoded["loss_mask"] == [1, 1] and encoded["prefix_length"] == 3
    for field, value in (("prefix_length", 2), ("loss_mask", [0, 1]),
                         ("tokens", [True, 2, 3, 4, 5]), ("split", "review")):
        bad = _sample()
        bad.metadata[field] = value
        with pytest.raises(ValueError):
            encoded_sample(bad, 64)
    with pytest.raises(ValueError, match="overflow"):
        encoded_sample(sample, 4)


def test_real_miles_sample_rollout_when_runtime_is_installed(tmp_path: Path) -> None:
    miles_types = pytest.importorskip("miles.utils.types")
    rollout = importlib.import_module("sft.miles_sft.runtime.rollout")
    base_types = importlib.import_module("miles.rollout.base_types")
    view = _sample()
    sample = miles_types.Sample(prompt=view.prompt, metadata=view.metadata)
    source = SimpleNamespace(get_samples=lambda count: [[sample]])
    output = rollout.generate_rollout(_args(tmp_path), 0, source)
    assert isinstance(output, base_types.RolloutFnTrainOutput)
    assert output.samples[0][0] is sample
    assert sample.status is miles_types.Sample.Status.COMPLETED and sample.reward == 0
    assert sample.tokens == [1, 2, 3, 4, 5] and sample.response_length == 2
    sample.validate()
    with pytest.raises(ValueError, match="training data only"):
        rollout.generate_rollout(_args(tmp_path), 0, source, evaluation=True)


def test_configuration_requires_fresh_scoped_train_only_lora(tmp_path: Path) -> None:
    validate_args(_args(tmp_path))
    for key, value in (("lora_adapter_path", "/old"), ("target_modules", ["all-linear"]),
                       ("eval_num_gpus", 1), ("lora_rank", 8), ("lora_type", "canonical_lora"),
                       ("enable_mtp_training", True), ("debug_disable_optimizer", True)):
        args = _args(tmp_path)
        setattr(args, key, value)
        with pytest.raises(ValueError):
            validate_args(args)


def test_trainable_scope_freshness_and_gdn_coverage() -> None:
    model = _model()
    params, scope = adapter_parameters([model])
    audit_fresh_adapter(params)
    assert len(params) == 12 and scope["adapter_pairs"] == 6
    for name in ("vision_model.weight", "language_model.output_layer.weight",
                 "language_model.decoder.layers.0.self_attention.in_proj.base.weight"):
        param = model.get_parameter(name)
        param.requires_grad_(True)
        with pytest.raises(ValueError, match="forbidden trainable"):
            adapter_parameters([model])
        param.requires_grad_(False)
    gdn = model.get_parameter("language_model.decoder.layers.0.self_attention.in_proj.adapter.linear_in.weight")
    gdn.requires_grad_(False)
    with pytest.raises(ValueError, match="unexpectedly frozen"):
        adapter_parameters([model])


def test_main_grad_precedence_and_missing_or_nonfinite_gradients() -> None:
    params, _ = adapter_parameters([_model()])
    with pytest.raises(ValueError, match="missing adapter gradient"):
        gradient_witness(params)
    for param in params.values():
        param.main_grad = torch.ones_like(param)
        param.grad = torch.full_like(param, float("nan"))
    assert gradient_witness(params)["main_grad_tensors"] == len(params)
    next(iter(params.values())).main_grad.fill_(float("inf"))
    with pytest.raises(ValueError, match="nonfinite"):
        gradient_witness(params)


def test_skipped_optimizer_never_writes_success(tmp_path: Path) -> None:
    model = _model()
    optimizer = CpuOptimizer(model)
    original_step = optimizer.step

    def skipped(*args: object, **kwargs: object) -> tuple[dict[str, float], float, Outcome]:
        return {"sft_loss": 1.0}, 0.0, Outcome.NORMAL

    with pytest.raises(ValueError, match="no successful optimizer step"):
        audited_train_step(skipped, tmp_path, _args(tmp_path), 0, 0, [], [model], optimizer,
                           object(), 1, 1, None, 0)
    assert optimizer.step == original_step
    assert not list(tmp_path.glob("*-after.json"))


class CpuOptimizer:
    def __init__(self, model: torch.nn.Module) -> None:
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(self.params, lr=0.1)
        self.result = (True, 1.0, None)

    def step(self) -> tuple[bool, float, None]:
        for param in self.params:
            param.grad = param.main_grad
        self.optimizer.step()
        return self.result


def _exercise_step(directory: Path, model: torch.nn.Module, args: SimpleNamespace) -> CpuOptimizer:
    optimizer = CpuOptimizer(model)
    expected = ({"sft_loss": 1.0}, 1.0, Outcome.NORMAL)

    def original(*positional: object, **keywords: object) -> tuple[dict[str, float], float, Outcome]:
        losses = []
        for name, module in model.named_modules():
            if name.endswith(".adapter"):
                losses.append((module.linear_out(module.linear_in(torch.ones(2, 4))) - 1).square().mean())
        loss = torch.stack(losses).mean()
        loss.backward()
        for param in optimizer.params:
            param.main_grad, param.grad = param.grad, None
        assert optimizer.step() is optimizer.result
        optimizer.optimizer.zero_grad(set_to_none=True)
        return expected

    result = audited_train_step(original, directory, args, 0, 0, [], [model], optimizer, object(), 1, 1, None, 0)
    assert result is expected
    evidence = load_json_dict(directory / "rollout-0-step-0-rank-0-after.json")
    assert evidence["main_grad_tensors"] == 12 and evidence["parameters_changed"] is True
    assert evidence["loss_timing"] == "forward_before_update"
    return optimizer


def test_real_cpu_update_and_checkpoint_cursor_finalization(
    tmp_path: Path, export_bundle: tuple[Path, Path, Path],
) -> None:
    args, model = _args(tmp_path), _model()
    base, bridge_path, hf_path = export_bundle
    args.hf_checkpoint = args.load = str(base)
    directory = tmp_path / "receipts"
    optimizer = _exercise_step(directory, model, args)
    _, scope = adapter_parameters([model])
    write_receipt(directory / "scope-rank-0.json", scope)
    assert bridge_path == Path(args.save_hf.format(rollout_id=0))
    assemble_complete_export(base, bridge_path, hf_path)
    checkpoint = Path(args.save) / "iter_0000000"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    state = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(state, adapter / "adapter_megatron_rank0.pt")
    torch.save({"iteration": 0, "optimizer": optimizer.optimizer.state_dict(), "opt_param_scheduler": {"step": 1}},
               adapter / "training_state_rank0.pt")
    native = audit_native(checkpoint, 0, [scope])
    hf = audit_hf_export(base, hf_path, bridge_export=bridge_path)
    write_receipt(directory / "checkpoint-0.json", {
        "rollout_id": 0, "status": "model_saved_cursor_pending", "complete": False,
        "checkpoint_dir": str(checkpoint), "hf_checkpoint_dir": str(hf_path), "native": native, "hf": hf,
        "bridge_checkpoint_dir": str(bridge_path),
    })
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(_sample().metadata))
    write_receipt(directory / "initialize-rank-0.json", {
        "miles_commit": MILES_COMMIT, "rank": 0, "world_size": 1, "num_rollout": 1,
        "steps_per_rollout": 1, "rollout_batch_size": 1, "prompt_data": str(input_path),
        "input_sha256": file_hash(input_path), "save": args.save, "save_hf": args.save_hf,
        "hf_checkpoint": args.hf_checkpoint,
    })
    with pytest.raises(ValueError, match="data-source state has not yet"):
        finalize_run(directory)
    assert not (directory / "run.json").exists()
    cursor = Path(args.save) / "rollout"
    cursor.mkdir()
    torch.save({"sample_index": 1, "sample_group_index": 1, "sample_offset": 1, "epoch_id": 0},
               cursor / "global_dataset_state_dict_0.pt")
    result = finalize_run(directory)
    assert result["complete"] is True
    assert result["hf_checkpoint_dir"] == str(hf_path)
    assert result["bridge_checkpoint_dir"] == str(bridge_path)
    post_save = load_json_dict(directory / "checkpoint-0.json")
    post_save["bridge_checkpoint_dir"] = str(hf_path)
    (directory / "checkpoint-0.json").write_text(json.dumps(post_save))
    with pytest.raises(ValueError, match="paths disagree"):
        finalize_run(directory)
    state["language_model.output_layer.weight"] = model.language_model.output_layer.weight.detach()
    torch.save(state, adapter / "adapter_megatron_rank0.pt")
    with pytest.raises(ValueError, match="names/shapes"):
        audit_native(checkpoint, 0, [scope])
