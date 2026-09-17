"""Offline real-tokenizer/torch/PEFT contracts, not a pinned 27B training validation."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import peft
import PIL.Image
import pytest
import torch
import torch.nn.functional as F
import transformers
import trl
from datasets import Dataset
from safetensors.torch import load_file
from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import (
    CLIPImageProcessor, GenerationMixin, LlavaProcessor, PretrainedConfig,
    PreTrainedModel, PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast

from evals.catan_board_bench.tokens import semantic_recognition_token_inventory
from sft.scripts import train_trl_catan_vision as training


def text_row(prompt="board <N00> above <N01>?", answer="yes"):
    return {"messages": [
        {"role": "user", "content": prompt}, {"role": "assistant", "content": answer},
    ]}


def text_tokenizer():
    words = ["[UNK]", "<|im_start|>", "<|im_end|>", "user", "assistant", "\n", " ",
             "node", "edge", "tile", "port", "yes", "no", "board", "wood", "empty", "above", "?"]
    raw = Tokenizer(WordLevel(dict(zip(words, range(len(words)), strict=True)), unk_token="[UNK]"))
    raw.pre_tokenizer = Split(Regex(r"\s"), behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", eos_token="<|im_end|>",
        pad_token="<|im_end|>", additional_special_tokens=["<|im_start|>"],
    )
    tokenizer.chat_template = (
        "{% if enable_thinking != false or preserve_thinking != false %}"
        "{{ raise_exception('thinking must be explicitly disabled') }}{% endif %}"
        "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content']"
        " + '<|im_end|>\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    tokenizer.add_tokens(semantic_recognition_token_inventory()["tokens"], special_tokens=False)
    return tokenizer


class DormantVisual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)
        self.merger = torch.nn.Linear(8, 8)
        self.register_buffer("scale", torch.tensor(1.0001))
        self.register_buffer("indices", torch.tensor([0, 257], dtype=torch.int64))

    def forward(self, *args, **kwargs):
        raise AssertionError("a text forward must never execute the visual tower")


class TinyLanguage(torch.nn.Module):
    def __init__(self, num_layers=1):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(256, 8)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            layer = torch.nn.Module()
            layer.q_proj = torch.nn.Linear(8, 8, bias=False)
            layer.down_proj = torch.nn.Linear(8, 8, bias=False)
            self.layers.append(layer)

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = torch.sigmoid(layer.down_proj(torch.tanh(layer.q_proj(hidden))))
        return BaseModelOutput(last_hidden_state=hidden)


class TinyTextVLM(PreTrainedModel, GenerationMixin):
    """Real numerical forward/generation with Qwen's module paths and untied heads."""

    config_class = PretrainedConfig

    def __init__(self, num_layers=1):
        super().__init__(PretrainedConfig(
            vocab_size=256, hidden_size=8, tie_word_embeddings=False,
            eos_token_id=2, pad_token_id=2, use_cache=False,
            max_position_embeddings=16384,
        ))
        self.model = torch.nn.Module()
        self.model.language_model = TinyLanguage(num_layers)
        self.model.visual = DormantVisual()
        self.lm_head = torch.nn.Linear(8, 256, bias=False)
        with torch.no_grad():
            for parameter in self.parameters():
                values = torch.arange(parameter.numel()).reshape(parameter.shape)
                parameter.copy_(torch.sin(values.float() + 1) * 0.03)
            self.lm_head.weight[11].fill_(1.0)  # The fixture's deterministic greedy answer is yes.

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, new_num_tokens, **kwargs):
        assert new_num_tokens == 256
        return self.get_input_embeddings()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        assert not training.TEXT_MEDIA_KEYS.intersection(kwargs)
        assert "_prediction_loss_only" not in kwargs
        hidden = self.model.language_model(input_ids).last_hidden_state
        logits = self.lm_head(hidden)
        loss = None if labels is None else F.cross_entropy(
            logits[:, :-1].float().reshape(-1, 256), labels[:, 1:].reshape(-1),
        )
        return CausalLMOutputWithPast(logits=logits, loss=loss)


def write_text_source(bundle, *, source_mode=None, num_layers=1, legacy_processor_serialization=False):
    """Save a real rank-8/154-row parent, including nonempty optimizer state."""
    tokenizer = text_tokenizer()
    inventory = semantic_recognition_token_inventory()
    base = TinyTextVLM(num_layers).bfloat16()
    setup, components = training.prepare_semantic_tokens(tokenizer, base, inventory)
    config = training.TrainConfig(
        train_jsonl="unused.jsonl", image_root="unused-images", token_inventory="tokens.json",
        output_dir=str(bundle), model_id="offline-tiny", token_init="keep",
    )
    model = training.wrap_trainable_model(base, setup, components, config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "trainable_tokens_delta" in name:
                values = torch.arange(parameter.numel()).reshape(parameter.shape)
                parameter.copy_(torch.cos(values.float() + 1) * 0.1)
            elif ".lora_" in name:
                parameter.fill_(0.0125)
        for parameter in training.resolve_wrapped_module(model, components.vision).parameters():
            parameter.fill_(1.0001)
        training.resolve_wrapped_module(model, components.vision).scale.fill_(1.0001)
    model.eval()
    optimizer = training.build_optimizer(model, components, config)
    batch = training.TextCompletionCollator(tokenizer, max_sequence_length=128)([text_row()])
    model(**batch).loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    bundle.mkdir(parents=True)
    model.save_pretrained(bundle, save_embedding_layers=False)
    # A real HF image/text processor exercises serialization without torchvision.
    # This fixture is not a claim that Qwen's pinned processor has been executed.
    processor = LlavaProcessor(
        tokenizer=tokenizer, image_processor=CLIPImageProcessor(),
        chat_template=tokenizer.chat_template,
    )
    processor.save_pretrained(bundle, legacy_serialization=legacy_processor_serialization)
    # Explicit video configuration fixture: preserve bytes without importing the
    # torchvision-dependent video processor or manufacturing video data.
    training.write_json_atomic(bundle / "video_preprocessor_config.json", {
        "video_processor_type": "Qwen2VLVideoProcessor", "patch_size": 14,
        "temporal_patch_size": 2, "merge_size": 2, "min_pixels": 128 * 28 * 28,
        "max_pixels": 768 * 28 * 28,
    })
    training.save_visual_state(model, components, bundle)
    torch.save(optimizer.state_dict(), bundle / "optimizer.pt")
    assert optimizer.state
    saved_config = asdict(config)
    if source_mode is None:
        saved_config.pop("input_mode")  # Historical checkpoints predate this field.
    else:
        saved_config["input_mode"] = source_mode
    training.write_json_atomic(bundle / training.RUN_CONFIG_FILE, saved_config)
    training.write_json_atomic(bundle / training.TRAINABLE_SCOPE_FILE, {
        "semantic_tokens": setup.as_dict(),
    })
    return model, tokenizer, setup, components, replace(
        config, input_mode="text", image_root=None, initial_bundle=str(bundle),
        max_sequence_length=8192,
    )


def load_text_source(bundle, tokenizer, config, *, num_layers=1):
    base = TinyTextVLM(num_layers).bfloat16()
    setup, components = training.prepare_semantic_tokens(
        tokenizer, base, semantic_recognition_token_inventory(),
    )
    return training.load_initial_bundle(base, setup, components, config)


def test_config_keeps_vision_defaults_and_requires_explicit_text_contract():
    old = dict(train_jsonl="t", image_root="images", token_inventory="i", output_dir="o")
    config = training.TrainConfig(**old)
    config.validate()
    assert config.input_mode == "vision" and config.max_sequence_length is None
    assert (config.profile, config.lora_rank, config.lora_alpha, config.lora_dropout) == (
        "vision_tokens_lora", 8, 16, 0.05,
    )
    with pytest.raises(ValueError, match="image_root"):
        replace(config, image_root=None).validate()
    args = training.parse_args([
        "--train-jsonl", "t", "--token-inventory", "i", "--output-dir", "o",
        "--input-mode", "text", "--initial-bundle", "checkpoint-128", "--token-init", "keep",
        "--max-sequence-length", "8192", "--eval-jsonl", "e",
    ])
    args.validate()
    assert args.image_root is args.eval_image_root is None
    for updates, error in [
        ({"initial_bundle": None}, "initial_bundle"),
        ({"token_init": "mean_noise"}, "token_init=keep"),
        ({"lora_rank": 4}, "rank 8 or 16"),
        ({"profile": "vision_tokens"}, "vision_tokens_lora"),
        ({"max_sequence_length": None}, "max_sequence_length"),
        ({"max_sequence_length": 0}, "max_sequence_length"),
        ({"patch_loss_weight": 1}, "patch objectives"),
        ({"spatial_target_mode": "shuffled"}, "shuffled targets"),
    ]:
        with pytest.raises(ValueError, match=error):
            replace(args, **updates).validate()


@pytest.mark.parametrize("saved_mode", ["vision", "text"])
def test_text_resume_is_explicitly_rejected_and_cross_mode_vision_resume_fails(tmp_path, saved_mode):
    training.write_json_atomic(tmp_path / training.RUN_CONFIG_FILE, {"input_mode": saved_mode})
    config = training.TrainConfig(
        train_jsonl="t", token_inventory="i", output_dir="o", input_mode="text",
        resume_from_checkpoint=str(tmp_path), token_init="keep", max_sequence_length=128,
    )
    with pytest.raises(ValueError, match="text-mode resume is not supported"):
        config.validate()
    vision = replace(config, input_mode="vision", image_root="images")
    if saved_mode == "text":
        with pytest.raises(ValueError, match="cross-mode resume"):
            vision.validate()
    else:
        vision.validate()


@pytest.mark.parametrize("row", [
    {**text_row(), "images": []}, {**text_row(), "image": "must-not-open.png"},
    text_row("<image> board"), text_row(answer="<|im_end|>"),
    text_row([{"type": "image", "image": "x"}]),
    text_row([{"type": "text", "text": 17}]), text_row(17), text_row(answer=""),
    {**text_row(), "spatial_targets": [{"bbox": [0, 0, 1, 1]}]},
    {"messages": [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]},
])
def test_text_rows_reject_media_controls_and_malformed_pairs(row):
    with pytest.raises(ValueError):
        training._message_pair(row, line_number=7, input_mode="text")


def test_native_collation_masks_exact_boundary_and_supervises_eot_even_when_pad_equals_eot():
    tokenizer = text_tokenizer()
    rows = [text_row(answer="<N00>"), text_row("board", "yes")]
    batch = training.TextCompletionCollator(tokenizer, max_sequence_length=128)(rows)
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    for i, row in enumerate(rows):
        messages = row["messages"]
        full = tokenizer.apply_chat_template(
            messages, tokenize=True, enable_thinking=False, preserve_thinking=False,
        )
        prefix = tokenizer.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True,
            enable_thinking=False, preserve_thinking=False,
        )
        assert batch["input_ids"][i, :len(full)].tolist() == full
        assert batch["labels"][i, :len(prefix)].tolist() == [-100] * len(prefix)
        assert batch["labels"][i, len(prefix):len(full)].tolist() == full[len(prefix):]
        assert tokenizer.eos_token_id in batch["labels"][i, len(prefix):len(full)]
        assert torch.all(batch["labels"][i, len(full):] == -100)
    assert tokenizer.encode("<N00>", add_special_tokens=False) == [
        tokenizer.convert_tokens_to_ids("<N00>"),
    ]


def test_native_boundary_mismatch_and_missing_eot_are_rejected():
    tokenizer = text_tokenizer()
    tokenizer.chat_template = (
        "{{ 'assistant' }}{% if not add_generation_prompt %}"
        "{{ messages[-1]['content'] + '<|im_end|>' }}{% endif %}"
    )
    with pytest.raises(ValueError, match="token boundary"):
        training.encode_text_pair(tokenizer, "board", "yes", max_sequence_length=128)
    tokenizer.chat_template = (
        "{{ 'assistant\n' }}{% if not add_generation_prompt %}"
        "{{ messages[-1]['content'] }}{% endif %}"
    )
    with pytest.raises(ValueError, match="end-of-turn"):
        training.encode_text_pair(tokenizer, "board", "yes", max_sequence_length=128)


def test_long_board_text_has_exact_budget_and_never_uses_image_or_patch_pipeline(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("text used a media/patch pipeline")

    for name in ("resolve_image_path", "validate_spatial_targets", "ThreadPoolExecutor"):
        monkeypatch.setattr(training, name, forbidden)
    row = text_row("board " * 1000, "<N00>")
    tokenizer = text_tokenizer()
    prompt, answer = training._message_pair(row, line_number=1, input_mode="text")
    assert len(prompt) > training.MAX_PROMPT_CHARACTERS
    length = len(training.encode_text_pair(tokenizer, prompt, answer, max_sequence_length=8192)["input_ids"])
    source = tmp_path / "text.jsonl"
    source.write_text(json.dumps(row) + "\n")
    dataset, report = training.load_text_dataset(
        source, tokenizer=tokenizer, max_sequence_length=length, require_curriculum=False,
    )
    assert dataset.column_names == ["messages"]
    assert report["max_sequence_tokens"] == report["total_sequence_tokens"] == length
    assert report["unique_images"] == 0 and report["image_root"] is None
    assert report["image_manifest_sha256"] is None and not report["truncation"]
    with pytest.raises(ValueError, match="truncation is forbidden"):
        training.load_text_dataset(
            source, tokenizer=tokenizer, max_sequence_length=length - 1, require_curriculum=False,
        )


@pytest.mark.parametrize("run_name", [
    "full-board-new-layouts-20260907", "spatial-continuation-20260912-r01",
])
def test_real_checkpoint_load_keeps_replacement_rows_visual_fp32_and_fresh_optimizer(tmp_path, run_name):
    bundle = tmp_path / run_name / "checkpoints" / "checkpoint-128"
    source, _, setup, components, config = write_text_source(bundle)
    tokenizer = training.load_checkpoint_text_tokenizer(bundle, setup.tokens)
    loaded, receipt = load_text_source(bundle, tokenizer, config)
    assert isinstance(loaded, peft.PeftModel)
    loaded.eval()
    ids = torch.tensor([[setup.token_ids[0], 11, setup.token_ids[-1]]])
    with torch.no_grad():
        assert torch.equal(source(input_ids=ids).logits, loaded(input_ids=ids).logits)
    assert receipt["source_input_mode"] == "vision"
    assert not any(receipt[k] for k in (
        "optimizer_state_restored", "scheduler_state_restored", "rng_state_restored",
    ))
    for name, tensor in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    for path in (components.input_embedding, components.output_head):
        wrapper = training.resolve_wrapped_module(loaded, path)
        replacement = wrapper.token_adapter.trainable_tokens_delta["default"]
        assert replacement.shape == (154, 8)
        functional = wrapper.weight[list(setup.token_ids)]
        assert torch.equal(functional.float(), replacement.to(functional.dtype).float())
        base_rows = wrapper.token_adapter.get_base_layer().weight[list(setup.token_ids)]
        assert not torch.allclose(functional.float(), base_rows.float() + replacement.float())
    scope = training.audit_trainable_scope(loaded, components, setup, config, tmp_path / "scope.json")
    assert not scope["errors"]
    optimizer = training.build_optimizer(loaded, components, config)
    assert not optimizer.state
    assert {g["catan_name"]: g["lr"] for g in optimizer.param_groups} == {
        "token_rows": 5e-4, "language_lora": 1e-4,
    }
    coverage = training.audit_optimizer_coverage(loaded, optimizer, tmp_path / "coverage.json")
    assert coverage["trainable_tensors"] == coverage["covered_tensors"] == 6
    frozen = {name: p.clone() for name, p in loaded.named_parameters() if not p.requires_grad}
    batch = training.TextCompletionCollator(tokenizer, max_sequence_length=128)([
        text_row("board <N00>", "<N01>"),
    ])
    loaded(**batch).loss.backward()
    assert all(p.grad is not None for p in loaded.parameters() if p.requires_grad)
    optimizer.step()
    for name, parameter in loaded.named_parameters():
        if name in frozen:
            assert parameter.grad is None and torch.equal(parameter, frozen[name]), name


def test_actual_tokenizer_mapping_and_adapter_layout_are_validated(tmp_path):
    _, tokenizer, setup, components, config = write_text_source(tmp_path / "parent")
    bundle = tmp_path / "parent"
    raw = json.loads(tokenizer.backend_tokenizer.to_str())
    atlas = [item for item in raw["added_tokens"] if item["content"] in setup.tokens]
    atlas[0]["content"], atlas[1]["content"] = atlas[1]["content"], atlas[0]["content"]
    swapped = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(json.dumps(raw)))
    with pytest.raises(ValueError, match="actual checkpoint tokenizer mapping"):
        training.validate_checkpoint_tokenizer(swapped, bundle, setup.tokens)
    adapter_path = bundle / "adapter_config.json"
    adapter = json.loads(adapter_path.read_text())
    adapter["target_modules"].append("model.visual.proj")
    adapter_path.write_text(json.dumps(adapter))
    with pytest.raises(ValueError, match="compatible language rank-8"):
        training.validate_text_adapter(TinyTextVLM(), bundle, setup, components, config=config)


def test_actual_peft_condensation_of_twenty_targets_loads_exact_language_modules(tmp_path):
    bundle = tmp_path / "parent"
    source, tokenizer, setup, components, config = write_text_source(bundle, num_layers=10)
    intended = set(training.language_linear_targets(TinyTextVLM(10), components))
    assert len(intended) == 20
    saved_targets = set(json.loads((bundle / "adapter_config.json").read_text())["target_modules"])
    assert saved_targets == {"q_proj", "down_proj"}  # Produced by PEFT, not manually condensed.
    loaded, _ = load_text_source(bundle, tokenizer, config, num_layers=10)
    assert set(loaded.base_model.targeted_module_names) == intended
    loaded.eval()
    ids = torch.tensor([[setup.token_ids[0], 11]])
    with torch.no_grad():
        assert torch.equal(source(input_ids=ids).logits, loaded(input_ids=ids).logits)


@pytest.mark.parametrize("extra_path", ["model.visual", "model", "model.language_model"])
@pytest.mark.parametrize("linear", [True, False])
def test_suffix_selectors_cannot_admit_extra_visual_base_or_nonlinear_modules(tmp_path, extra_path, linear):
    bundle = tmp_path / "parent"
    _, _, setup, components, config = write_text_source(bundle, num_layers=10)
    base = TinyTextVLM(10)
    extra = base.get_submodule(extra_path)
    extra.q_proj = torch.nn.Linear(8, 8) if linear else torch.nn.Identity()
    with pytest.raises(ValueError, match="extra=.*q_proj"):
        training.validate_text_adapter(base, bundle, setup, components, config=config)


def test_peft_exclusions_and_layer_filters_cannot_hide_missing_language_targets(tmp_path):
    bundle = tmp_path / "parent"
    _, _, setup, components, config = write_text_source(bundle, num_layers=10)
    path = bundle / "adapter_config.json"
    original = json.loads(path.read_text())
    for change in ({"exclude_modules": ["q_proj"]}, {"layers_to_transform": [0]}):
        path.write_text(json.dumps({**original, **change}))
        with pytest.raises(ValueError, match="missing=.*q_proj"):
            training.validate_text_adapter(TinyTextVLM(10), bundle, setup, components, config=config)


def test_real_peft_output_bridge_matches_direct_loss_and_gradients(tmp_path):
    _, tokenizer, _, components, config = write_text_source(tmp_path / "parent")
    model, _ = load_text_source(tmp_path / "parent", tokenizer, config)
    model.float().eval()
    head = training.resolve_wrapped_module(model, components.output_head)
    replacement = head.token_adapter.trainable_tokens_delta["default"]
    hidden = torch.linspace(-1, 1, 32).reshape(4, 8).requires_grad_(True)
    labels = torch.tensor([11, 18, 19, 171])
    direct = F.cross_entropy(head(hidden).float(), labels)
    direct.backward()
    expected_hidden, expected_rows = hidden.grad.clone(), replacement.grad.clone()
    hidden.grad = replacement.grad = None
    original = model.get_base_model().get_output_embeddings()
    with training.expose_trainable_tokens_head_to_chunked_nll(model):
        view = model.get_base_model().get_output_embeddings()
        # Project separate chunks using the exact same functional weight TRL captures.
        weight = view.weight
        logits = torch.cat([F.linear(chunk, weight) for chunk in hidden.split(2)])
        chunked = F.cross_entropy(logits.float(), labels)
        chunked.backward()
    assert model.get_base_model().get_output_embeddings() is original
    torch.testing.assert_close(chunked, direct)
    torch.testing.assert_close(hidden.grad, expected_hidden)
    torch.testing.assert_close(replacement.grad, expected_rows)
    assert head.token_adapter.get_base_layer().weight.grad is None


@pytest.mark.parametrize("legacy_assets", [False, True])
def test_text_trainer_loss_metrics_and_checkpoint_final_roundtrip_on_local_stack(tmp_path, monkeypatch, legacy_assets):
    """Exercise local SFTTrainer plumbing; the PEFT weight bridge has a separate math test."""
    _, tokenizer, setup, components, config = write_text_source(
        tmp_path / "parent", legacy_processor_serialization=legacy_assets,
    )
    model, _ = load_text_source(tmp_path / "parent", tokenizer, config)

    def forbidden(*args, **kwargs):
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
    def factory(*args, **kwargs):
        return TinyTextVLM().bfloat16()

    monkeypatch.setattr(transformers, "AutoModelForMultimodalLM", SimpleNamespace(from_pretrained=factory), raising=False)
    report = training.validate_saved_bundle(config, tmp_path / "final", semantic_recognition_token_inventory(), setup)
    assert report["valid"] and report["input_mode"] == "text"
    assert report["processor_assets_sha256"] == parent_assets


@pytest.mark.parametrize("prediction_loss_only", [True, False])
def test_trl_112_prediction_step_flag_reaches_parent_loss_but_never_model(tmp_path, monkeypatch, prediction_loss_only):
    """Reproduce the inspected v1.12.0 prediction/compute_loss boundary on local TRL.

    Source: huggingface/trl v1.12.0, trl/trainer/sft_trainer.py:1749-1751,1883-1886.
    Only this boundary is transplanted; the local numerical loss/forward is real.
    This does not claim an installed pinned-runtime integration test.
    """
    _, tokenizer, setup, components, config = write_text_source(tmp_path / "parent")
    model, _ = load_text_source(tmp_path / "parent", tokenizer, config)
    received = []

    class PredictionContractTrainer(trl.SFTTrainer):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs["_prediction_loss_only"] = prediction_loss_only
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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


def test_pinned_runtime_gate_is_explicit_and_fails_on_mismatch(monkeypatch):
    monkeypatch.setattr(transformers, "__version__", "offline-unpinned")
    with pytest.raises(RuntimeError, match="runtime version mismatch.*5.16.1.*1.12.0"):
        training.assert_runtime_versions()


def test_text_scope_and_optimizer_reject_unfrozen_visuals_or_base(tmp_path):
    _, tokenizer, setup, components, config = write_text_source(tmp_path / "parent")
    for path in (components.vision, components.language + ".layers.0.q_proj.base_layer"):
        model, _ = load_text_source(tmp_path / "parent", tokenizer, config)
        training.resolve_wrapped_module(model, path).requires_grad_(True)
        with pytest.raises(RuntimeError, match="invalid trainable scope"):
            training.audit_trainable_scope(model, components, setup, config, tmp_path / "bad.json")
        with pytest.raises(RuntimeError, match="text optimizer cannot contain"):
            training.build_optimizer(model, components, config)
