"""Trainer loss, metrics, checkpoint round trip, and runtime gates."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import PIL.Image
import pytest
import torch
import transformers
import trl
from datasets import Dataset
from safetensors.torch import load_file
from transformers import (
    CLIPImageProcessor,
    LlavaProcessor,
)

from evals.catan_board_bench.tokens import semantic_recognition_token_inventory
from sft.scripts.train import train_trl_catan_vision as training

from .support import TinyTextVLM, load_text_source, text_row, write_text_source


@pytest.mark.parametrize("legacy_assets", [False, True])
def test_text_trainer_loss_metrics_and_checkpoint_final_roundtrip_on_local_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_assets: bool) -> None:
    """Exercise local SFTTrainer plumbing; the PEFT weight bridge has a separate math test."""
    _, tokenizer, setup, components, config = write_text_source(
        tmp_path / "parent", legacy_processor_serialization=legacy_assets,
    )
    model, _ = load_text_source(tmp_path / "parent", tokenizer, config)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("text trainer tried to capture or supervise vision")

    monkeypatch.setattr(training, "VisionPoolerCapture", forbidden)
    monkeypatch.setattr(training, "SpatialTargetCollator", forbidden)
    monkeypatch.setattr(training, "spatial_patch_loss", forbidden)
    monkeypatch.setattr(PIL.Image, "open", forbidden)
    monkeypatch.setattr(CLIPImageProcessor, "preprocess", forbidden)
    args = trl.SFTConfig(
        output_dir=str(tmp_path / "checkpoints"), use_cpu=True, bf16=False, fp16=False,
        report_to="none", max_length=None, gradient_checkpointing=False,
        remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True},
    )
    instance = training._trainer_class(config, components, setup)(
        model=model, args=args, processing_class=tokenizer,
        train_dataset=Dataset.from_list([text_row()]),
    )
    assert isinstance(instance.data_collator, training.TextCompletionCollator)
    assert not hasattr(instance, "_catan_vision_capture")
    batch = instance.data_collator([text_row("board <N00>", "<N01>")])
    loss, outputs = instance.compute_loss(model, batch, return_outputs=True)
    assert torch.isfinite(loss) and torch.equal(loss, outputs.loss)
    assert instance._catan_metrics["answer_token_count"] == [1.0]
    assert set(instance._catan_metrics) == {
        "nll_loss", "answer_token_accuracy", "answer_row_exact", "answer_token_count",
    }
    loss.backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    with pytest.raises(ValueError, match="text loss accepts only"):
        instance.compute_loss(model, {**instance.data_collator([text_row()]), "pixel_values": torch.ones(1)})
    source_visual = load_file(tmp_path / "parent" / training.VISUAL_STATE_FILE)
    parent_assets = training.processor_asset_hashes(tmp_path / "parent")
    assert "processor_config.json" in parent_assets
    assert ("preprocessor_config.json" in parent_assets) is legacy_assets
    assert "video_preprocessor_config.json" in parent_assets
    for dest in (tmp_path / "checkpoints" / "checkpoint-128", tmp_path / "final"):
        instance._save(str(dest))
        assert training.processor_asset_hashes(dest) == parent_assets
        # This is a real legacy multimodal processor reload, with no pixels.
        vision_processor = transformers.AutoProcessor.from_pretrained(dest, local_files_only=True)
        assert isinstance(vision_processor, LlavaProcessor)
        assert isinstance(vision_processor.image_processor, CLIPImageProcessor)
        assert vision_processor.tokenizer.get_vocab() == tokenizer.get_vocab()
        assert vision_processor.chat_template == tokenizer.chat_template
        saved_visual = load_file(dest / training.VISUAL_STATE_FILE)
        assert all(torch.equal(saved_visual[k], v) and saved_visual[k].dtype == v.dtype
                   for k, v in source_visual.items())
        assert not (dest / training.PATCH_METRICS_FILE).exists()
        reloaded_tokenizer = training.load_checkpoint_text_tokenizer(dest, setup.tokens)
        reloaded, _ = load_text_source(dest, reloaded_tokenizer, replace(config, initial_bundle=str(dest)))
        model.eval()
        reloaded.eval()
        ids = torch.tensor([[setup.token_ids[0], 11]])
        with torch.no_grad():
            assert torch.equal(model(input_ids=ids).logits, reloaded(input_ids=ids).logits)
        assert tokenizer.get_vocab() == reloaded_tokenizer.get_vocab()
        assert tokenizer.chat_template == reloaded_tokenizer.chat_template
    def factory(*args: object, **kwargs: object) -> TinyTextVLM:
        return TinyTextVLM().bfloat16()

    # Dotted target: importing peft replaces sys.modules["transformers"] after collection.
    monkeypatch.setattr("transformers.AutoModelForMultimodalLM", SimpleNamespace(from_pretrained=factory), raising=False)
    report = training.validate_saved_bundle(config, tmp_path / "final", semantic_recognition_token_inventory(), setup)
    assert report["valid"] and report["input_mode"] == "text"
    assert report["processor_assets_sha256"] == parent_assets


@pytest.mark.parametrize("prediction_loss_only", [True, False])
def test_trl_112_prediction_step_flag_reaches_parent_loss_but_never_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prediction_loss_only: bool) -> None:
    """Reproduce the inspected v1.12.0 prediction/compute_loss boundary on local TRL.

    Source: huggingface/trl v1.12.0, trl/trainer/sft_trainer.py:1749-1751,1883-1886.
    Only this boundary is transplanted; the local numerical loss/forward is real.
    This does not claim an installed pinned-runtime integration test.
    """
    _, tokenizer, setup, components, config = write_text_source(tmp_path / "parent")
    model, _ = load_text_source(tmp_path / "parent", tokenizer, config)
    received: list[bool | None] = []

    class PredictionContractTrainer(trl.SFTTrainer):
        def prediction_step(
            self,
            model: torch.nn.Module,
            inputs: dict[str, torch.Tensor],
            prediction_loss_only: bool,
            ignore_keys: list[str] | None = None,
        ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            inputs["_prediction_loss_only"] = prediction_loss_only
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

        def compute_loss(
            self,
            model: torch.nn.Module,
            inputs: dict[str, torch.Tensor],
            return_outputs: bool = False,
            num_items_in_batch: int | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, object]:
            received.append(inputs.pop("_prediction_loss_only", None))
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    monkeypatch.setattr(trl, "SFTTrainer", PredictionContractTrainer)
    args = trl.SFTConfig(
        output_dir=str(tmp_path / "eval"), use_cpu=True, bf16=False, fp16=False,
        report_to="none", max_length=None, gradient_checkpointing=False,
        remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True},
    )
    instance = training._trainer_class(config, components, setup)(
        model=model, args=args, processing_class=tokenizer,
        train_dataset=Dataset.from_list([text_row()]),
    )
    model.eval()
    loss, logits, labels = instance.prediction_step(
        model, instance.data_collator([text_row()]), prediction_loss_only,
    )
    assert received == [prediction_loss_only] and torch.isfinite(loss)
    assert (logits is None) is prediction_loss_only
    assert (labels is None) is prediction_loss_only
    assert instance._catan_metrics["answer_token_count"] == [1.0]
    for bad in ({"pixel_values": torch.ones(1)}, {"_unknown_private_key": True},
                {"_prediction_loss_only": "true"}):
        with pytest.raises(ValueError):
            instance.compute_loss(model, {**instance.data_collator([text_row()]), **bad})
    assert received == [prediction_loss_only]


def test_pinned_runtime_gate_is_explicit_and_fails_on_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("transformers.__version__", "offline-unpinned")
    with pytest.raises(RuntimeError, match="runtime version mismatch.*5.16.1.*1.12.0"):
        training.assert_runtime_versions()


def test_text_scope_and_optimizer_reject_unfrozen_visuals_or_base(tmp_path: Path) -> None:
    _, tokenizer, setup, components, config = write_text_source(tmp_path / "parent")
    for path in (components.vision, components.language + ".layers.0.q_proj.base_layer"):
        model, _ = load_text_source(tmp_path / "parent", tokenizer, config)
        training.resolve_wrapped_module(model, path).requires_grad_(True)
        with pytest.raises(RuntimeError, match="invalid trainable scope"):
            training.audit_trainable_scope(model, components, setup, config, tmp_path / "bad.json")
        with pytest.raises(RuntimeError, match="text optimizer cannot contain"):
            training.build_optimizer(model, components, config)
