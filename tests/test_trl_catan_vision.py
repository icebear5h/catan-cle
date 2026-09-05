import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from sft.scripts.train_trl_catan_vision import (
    CURRICULUM_STAGES,
    ModelComponents,
    PROFILE_VISION_TOKENS_LORA,
    TrainConfig,
    _ChunkedNLLTrainableTokensHead,
    _message_pair,
    answer_token_metrics,
    build_optimizer,
    initialize_semantic_token_rows,
    inspect_jsonl_contract,
    load_visual_state,
    parameter_category,
    promote_visual_master_weights,
    resolve_wrapped_module,
    save_visual_state,
    soft_patch_target,
    spatial_patch_loss,
    validate_spatial_targets,
    SpatialTargetCollator,
    TokenSetup,
)


def _row(image: str, stage: str | None) -> dict:
    row = {
        "images": [image],
        "messages": [
            {"role": "user", "content": "<image>\nIs <N00> above <N01>?"},
            {"role": "assistant", "content": "yes"},
        ],
    }
    if stage is not None:
        row["curriculum_stage"] = stage
    return row


def _write_dataset(tmp_path: Path, stages: list[str | None]) -> tuple[Path, Path]:
    image_root = tmp_path / "images"
    image_root.mkdir(parents=True)
    train_jsonl = tmp_path / "train.jsonl"
    with train_jsonl.open("w") as handle:
        for index, stage in enumerate(stages):
            image_name = f"{index}.png"
            (image_root / image_name).write_bytes(b"not-decoded-by-contract-audit")
            handle.write(json.dumps(_row(image_name, stage)) + "\n")
    return train_jsonl, image_root


def test_contract_accepts_all_four_ordered_curriculum_stages(tmp_path):
    train_jsonl, image_root = _write_dataset(tmp_path, list(CURRICULUM_STAGES))

    report = inspect_jsonl_contract(train_jsonl, image_root, require_curriculum=True)

    assert report["rows"] == 4
    assert [stage["name"] for stage in report["stages"]] == list(CURRICULUM_STAGES)
    assert [stage["start_row"] for stage in report["stages"]] == [0, 1, 2, 3]


def test_contract_rejects_missing_and_regressing_curriculum(tmp_path):
    train_jsonl, image_root = _write_dataset(tmp_path / "missing", [None])
    with pytest.raises(ValueError, match="missing curriculum_stage"):
        inspect_jsonl_contract(train_jsonl, image_root, require_curriculum=True)

    train_jsonl, image_root = _write_dataset(
        tmp_path / "regression",
        [CURRICULUM_STAGES[1], CURRICULUM_STAGES[0]],
    )
    with pytest.raises(ValueError, match="order regressed"):
        inspect_jsonl_contract(train_jsonl, image_root, require_curriculum=True)


def test_contract_rejects_image_escape(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text(json.dumps(_row("../outside.png", CURRICULUM_STAGES[0])) + "\n")

    with pytest.raises(ValueError, match="escapes image_root"):
        inspect_jsonl_contract(train_jsonl, image_root, require_curriculum=False)


def test_spatial_target_contract_rejects_bad_or_unnormalized_boxes():
    target = {
        "token": "<N00>",
        "entity_type": "node",
        "bbox": [0.1, 0.2, 0.3, 0.4],
        "center": [0.2, 0.3],
        "control_bbox": [0.5, 0.6, 0.7, 0.8],
    }
    assert validate_spatial_targets({"spatial_targets": [target]}, line_number=1) == [target]

    bad = {**target, "bbox": [-0.1, 0.2, 0.3, 0.4]}
    with pytest.raises(ValueError, match="must be normalized"):
        validate_spatial_targets({"spatial_targets": [bad]}, line_number=2)


def test_message_pair_normalizes_sharegpt_and_short_answer():
    row = {
        "conversations": [
            {"from": "human", "value": "<image>\nWhere is the ore-wheat-sheep node?"},
            {"from": "gpt", "value": "<N17>"},
        ]
    }

    assert _message_pair(row, line_number=1) == (
        "Where is the ore-wheat-sheep node?",
        "<N17>",
    )


def test_message_pair_admits_full_board_readouts_and_rejects_runaway_answers():
    readout = "; ".join(f"<E{i:02d}_{i + 1:02d}> mystic blue road" for i in range(72))
    assert len(readout) > 512
    row = {"messages": [{"role": "user", "content": "<image>\nList every edge."}, {"role": "assistant", "content": readout}]}
    assert _message_pair(row, line_number=1) == ("List every edge.", readout)
    runaway = {"messages": [{"role": "user", "content": "<image>\nx"}, {"role": "assistant", "content": "a" * 2049}]}
    with pytest.raises(ValueError, match="answer-length contract"):
        _message_pair(runaway, line_number=2)


class _NestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = torch.nn.Module()
        self.base_model.model.model = torch.nn.Module()
        self.base_model.model.model.visual = torch.nn.Sequential(torch.nn.Linear(2, 2))


def test_wrapped_module_resolution_does_not_match_every_child():
    model = _NestedModel()

    resolved = resolve_wrapped_module(model, "model.visual")

    assert resolved is model.base_model.model.model.visual


def test_parameter_categories_separate_merger_from_rest_of_vision():
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=128,
        hidden_size=16,
    )

    assert parameter_category(
        "base_model.model.model.visual.blocks.0.weight", components
    ) == "vision"
    assert parameter_category(
        "base_model.model.model.visual.merger.linear.weight", components
    ) == "merger"
    assert parameter_category(
        "base_model.model.lm_head.trainable_tokens_delta.default", components
    ) == "atlas_output_rows"
    assert parameter_category(
        "base_model.model.model.language_model.layers.0.q_proj.weight", components
    ) == "forbidden"


def test_visual_state_round_trip_is_complete(tmp_path):
    model = _NestedModel()
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=128,
        hidden_size=16,
    )
    model._catan_components = components
    expected_weight = model.base_model.model.model.visual[0].weight.detach().clone()
    saved = save_visual_state(model, components, tmp_path)
    model.base_model.model.model.visual[0].weight.data.zero_()

    loaded = load_visual_state(model, tmp_path)

    assert saved["tensors"] == loaded["expected_tensors"] == loaded["tensors"]
    assert torch.equal(model.base_model.model.model.visual[0].weight, expected_weight)


class _OptimizerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = torch.nn.Module()
        self.base_model.model.model = torch.nn.Module()
        visual = torch.nn.Module()
        visual.block = torch.nn.Linear(2, 2)
        visual.merger = torch.nn.Linear(2, 2)
        self.base_model.model.model.visual = visual
        language = torch.nn.Module()
        language.embed_tokens = torch.nn.Module()
        language.embed_tokens.register_parameter(
            "trainable_tokens_delta", torch.nn.Parameter(torch.ones(154, 2))
        )
        language.layers = torch.nn.ModuleList([torch.nn.Module()])
        language.layers[0].q_proj = torch.nn.Module()
        language.layers[0].q_proj.register_parameter(
            "lora_A", torch.nn.Parameter(torch.ones(2, 2))
        )
        self.base_model.model.model.language_model = language
        self.base_model.model.lm_head = torch.nn.Module()
        self.base_model.model.lm_head.register_parameter(
            "trainable_tokens_delta", torch.nn.Parameter(torch.ones(154, 2))
        )


def test_production_default_uses_lower_lr_language_lora_group(tmp_path):
    config = TrainConfig(
        train_jsonl="train.jsonl",
        image_root="images",
        token_inventory="tokens.json",
        output_dir=str(tmp_path),
    )
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=128,
        hidden_size=2,
    )

    optimizer = build_optimizer(_OptimizerModel(), components, config)
    rates = {group["catan_name"]: group["lr"] for group in optimizer.param_groups}

    assert config.profile == PROFILE_VISION_TOKENS_LORA
    assert rates == {
        "vision": 5e-6,
        "merger": 5e-5,
        "token_rows": 5e-4,
        "language_lora": 1e-4,
    }
    assert config.warmup_ratio == 0.1


def test_visual_promotion_yields_fp32_trainable_master_weights():
    model = _NestedModel().to(torch.bfloat16)
    model.requires_grad_(False)
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=128,
        hidden_size=16,
    )

    report = promote_visual_master_weights(model, components)

    visual = model.base_model.model.model.visual
    assert all(p.dtype == torch.float32 and p.requires_grad for p in visual.parameters())
    assert report["dtypes"] == {"torch.float32": 2}


def test_visual_state_can_be_saved_in_bf16_for_the_final_bundle(tmp_path):
    model = _NestedModel()
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=128,
        hidden_size=16,
    )

    (tmp_path / "ckpt").mkdir()
    (tmp_path / "final").mkdir()
    fp32 = save_visual_state(model, components, tmp_path / "ckpt")
    bf16 = save_visual_state(model, components, tmp_path / "final", dtype=torch.bfloat16)

    assert fp32["dtype"] == "torch.float32"
    assert bf16["dtype"] == "torch.bfloat16"
    assert bf16["bytes"] < fp32["bytes"]


class _EmbeddingModel(torch.nn.Module):
    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.embed_tokens = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.model.language_model.embed_tokens.weight[-154:] = 7.0
            self.lm_head.weight[-154:] = -7.0


def test_semantic_rows_are_seeded_from_base_vocabulary_mean_and_are_distinct():
    vocab, hidden = 512, 8
    token_ids = tuple(range(vocab - 154 - 4, vocab - 4))
    setup = TokenSetup(
        tokens=tuple(f"<T{index}>" for index in range(154)),
        token_ids=token_ids,
        tokenizer_size=vocab - 4,
        model_vocab_size=vocab,
        added_tokens=154,
    )
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=vocab,
        hidden_size=hidden,
    )
    torch.manual_seed(0)
    first = _EmbeddingModel(vocab, hidden)
    torch.manual_seed(0)
    second = _EmbeddingModel(vocab, hidden)

    report = initialize_semantic_token_rows(first, components, setup, seed=42)
    initialize_semantic_token_rows(second, components, setup, seed=42)

    ids = torch.tensor(token_ids)
    for module in (first.model.language_model.embed_tokens, first.lm_head):
        base_mean = module.weight[: min(token_ids)].mean(dim=0)
        rows = module.weight[ids]
        assert torch.allclose(rows.mean(dim=0), base_mean, atol=0.2)
        assert rows.unique(dim=0).shape[0] == 154
        assert not torch.any(rows.abs() == 7.0)
    assert torch.equal(first.lm_head.weight, second.lm_head.weight)
    assert report["input_embedding"]["row_norm_after"] < report["input_embedding"]["row_norm_before"]


def test_family_words_init_seeds_each_family_from_its_word_row():
    vocab, hidden = 512, 8
    token_ids = tuple(range(vocab - 154 - 4, vocab - 4))
    tokens = tuple([f"<N{i:02d}>" for i in range(54)] + [f"<E{i:02d}_{i + 1:02d}>" for i in range(72)] + [f"<T{i:02d}>" for i in range(19)] + [f"<P{i:02d}>" for i in range(9)])
    family_word_ids = {"N": (10,), "E": (20, 21), "T": (30,), "P": (40,)}
    setup = TokenSetup(
        tokens=tokens,
        token_ids=token_ids,
        tokenizer_size=vocab - 4,
        model_vocab_size=vocab,
        added_tokens=154,
        family_word_ids=family_word_ids,
    )
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=vocab,
        hidden_size=hidden,
    )
    torch.manual_seed(0)
    model = _EmbeddingModel(vocab, hidden)
    with torch.no_grad():
        for family, ids in family_word_ids.items():
            for module in (model.model.language_model.embed_tokens, model.lm_head):
                module.weight[list(ids)] = float(ord(family)) / 10.0

    report = initialize_semantic_token_rows(model, components, setup, seed=42, mode="family_words")

    ids = torch.tensor(token_ids)
    for module in (model.model.language_model.embed_tokens, model.lm_head):
        rows = module.weight[ids]
        for family, word_ids in family_word_ids.items():
            members = torch.tensor([i for i, token in enumerate(tokens) if token[1] == family])
            base = module.weight[list(word_ids)].mean(dim=0)
            assert torch.allclose(rows[members].mean(dim=0), base, atol=0.2)
        assert rows.unique(dim=0).shape[0] == 154
        node_mean = rows[:54].mean(dim=0)
        tile_mean = rows[126:145].mean(dim=0)
        assert not torch.allclose(node_mean, tile_mean, atol=0.5)
    assert report["mode"] == "family_words" and set(report["input_embedding"]["family_base_norms"]) == {"N", "E", "T", "P"}
    with pytest.raises(ValueError):
        initialize_semantic_token_rows(model, components, setup, seed=42, mode="sideways")


def test_vocab_gaussian_init_matches_the_base_vocabulary_spread():
    vocab, hidden = 512, 8
    token_ids = tuple(range(vocab - 154 - 4, vocab - 4))
    setup = TokenSetup(
        tokens=tuple(f"<T{index}>" for index in range(154)),
        token_ids=token_ids,
        tokenizer_size=vocab - 4,
        model_vocab_size=vocab,
        added_tokens=154,
    )
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=vocab,
        hidden_size=hidden,
    )
    torch.manual_seed(0)
    model = _EmbeddingModel(vocab, hidden)
    report = initialize_semantic_token_rows(model, components, setup, seed=42, mode="vocab_gaussian")
    ids = torch.tensor(token_ids)
    for module in (model.model.language_model.embed_tokens, model.lm_head):
        reference = module.weight[: min(token_ids)]
        rows = module.weight[ids]
        assert torch.allclose(rows.mean(dim=0), reference.mean(dim=0), atol=0.3)
        ratio = rows.std(dim=0) / reference.std(dim=0)
        assert ratio.mean() > 0.7 and ratio.mean() < 1.3
        cosine = torch.nn.functional.normalize(rows - rows.mean(0), dim=1)
        offdiag = (cosine @ cosine.T).fill_diagonal_(0).abs().mean()
        assert offdiag < 0.5
    assert report["noise_scale"] == 1.0 and report["mode"] == "vocab_gaussian"


def test_answer_token_metrics_ignore_prompt_and_trivial_completion_tokens():
    vocab, hidden = 6, 4
    weight = torch.eye(vocab, hidden)
    # Row 0: prompt (-100), answer token 2 (correct), end-of-turn 4, newline 5.
    # Row 1: prompt, answer tokens 1 (correct) and 3 (wrong), end-of-turn 4.
    labels = torch.tensor([[-100, -100, 2, 4, 5], [-100, 1, 3, 4, -100]])
    hidden_states = torch.zeros(2, 5, hidden)
    hidden_states[0, 1, 2] = 1.0  # predicts 2 at the position preceding label 2
    hidden_states[1, 0, 1] = 1.0  # predicts 1
    hidden_states[1, 1, 0] = 1.0  # predicts 0, expected 3

    metrics = answer_token_metrics(hidden_states, labels, weight, None, trivial_token_ids=(4, 5))

    assert metrics["answer_token_count"] == 3.0
    assert metrics["answer_token_accuracy"] == pytest.approx(2 / 3)
    assert metrics["answer_row_exact"] == pytest.approx(0.5)


class _FakeTrainableTokensHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_layer = torch.nn.Linear(3, 7, bias=False)
        self.base_layer.requires_grad_(False)
        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged = False
        self.indices = torch.tensor([1, 5])
        self.delta = torch.nn.Parameter(self.base_layer.weight[self.indices].detach().clone())

    def get_base_layer(self) -> torch.nn.Module:
        return self.base_layer

    def get_merged_weights(self, active_adapters: list[str]) -> torch.Tensor:
        assert active_adapters == ["default"]
        return self.base_layer.weight.index_copy(0, self.indices, self.delta)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.get_merged_weights(self.active_adapters))


class _FakeTrainableTokensWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_adapter = _FakeTrainableTokensHead()

    @property
    def weight(self) -> torch.Tensor:
        return self.token_adapter.get_merged_weights(self.token_adapter.active_adapters)


def test_chunked_nll_head_view_matches_peft_forward_and_gradients():
    torch.manual_seed(7)
    head = _FakeTrainableTokensHead()
    view = _ChunkedNLLTrainableTokensHead(head)
    labels = torch.tensor([0, 1, 5, 3])

    direct_hidden = torch.randn(4, 3, requires_grad=True)
    direct_loss = F.cross_entropy(head(direct_hidden).float(), labels)
    direct_loss.backward()
    expected_hidden_grad = direct_hidden.grad.detach().clone()
    expected_delta_grad = head.delta.grad.detach().clone()

    head.delta.grad = None
    chunked_hidden = direct_hidden.detach().clone().requires_grad_(True)
    chunked_loss = F.cross_entropy(F.linear(chunked_hidden, view.weight).float(), labels)
    chunked_loss.backward()

    assert torch.allclose(chunked_loss, direct_loss)
    assert torch.allclose(chunked_hidden.grad, expected_hidden_grad)
    assert torch.allclose(head.delta.grad, expected_delta_grad)
    assert head.base_layer.weight.grad is None


def test_chunked_nll_head_view_respects_disabled_adapters():
    head = _FakeTrainableTokensHead()
    head.disable_adapters = True

    view_weight = _ChunkedNLLTrainableTokensHead(head).weight

    assert view_weight is head.base_layer.weight


def test_chunked_nll_head_view_supports_lora_trainable_tokens_wrapper():
    wrapper = _FakeTrainableTokensWrapper()
    view = _ChunkedNLLTrainableTokensHead(wrapper)

    assert torch.equal(view.weight, wrapper.weight)
    assert view.bias is None


class _FakeCollator:
    def __call__(self, examples):
        return {"input_ids": torch.tensor([[1], [2]])}


def test_spatial_target_collator_selects_correct_or_control_bbox():
    setup = TokenSetup(
        tokens=("<N00>",),
        token_ids=(17,),
        tokenizer_size=18,
        model_vocab_size=128,
        added_tokens=1,
    )
    target = {
        "token": "<N00>",
        "entity_type": "node",
        "bbox": [0.1, 0.2, 0.3, 0.4],
        "center": [0.2, 0.3],
        "control_bbox": [0.5, 0.6, 0.7, 0.8],
    }
    examples = [{"spatial_targets": [target]}, {"spatial_targets": []}]

    correct = SpatialTargetCollator(_FakeCollator(), setup, target_mode="correct")(examples)
    shuffled = SpatialTargetCollator(_FakeCollator(), setup, target_mode="shuffled")(examples)

    assert correct["spatial_target_token_ids"].tolist() == [17, 0]
    assert correct["spatial_target_mask"].tolist() == [True, False]
    assert correct["spatial_target_bboxes"][0].tolist() == pytest.approx(target["bbox"])
    assert shuffled["spatial_target_bboxes"][0].tolist() == pytest.approx(target["control_bbox"])


def test_soft_patch_objective_localizes_matching_post_merger_feature():
    pooled = torch.eye(4)
    grids = torch.tensor([[1, 4, 4]])
    token_embeddings = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    bboxes = torch.tensor([[0.0, 0.0, 0.49, 0.49]])
    mask = torch.tensor([True])

    target, xs, ys = soft_patch_target(bboxes[0], grid_height=2, grid_width=2)
    loss, accuracy, count = spatial_patch_loss(
        pooled,
        grids,
        token_embeddings,
        bboxes,
        mask,
        merge_size=2,
        temperature=0.07,
    )

    assert target.shape == xs.shape == ys.shape == (4,)
    assert target.sum().item() == pytest.approx(1.0)
    assert torch.isfinite(loss)
    assert accuracy.item() == 1.0
    assert count == 1
