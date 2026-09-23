import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evals.catan_board_bench.tokens import semantic_recognition_token_inventory
from sft.ms_swift_core import (
    ModelComponentPaths,
    attach_model_components,
    audit_optimizer_coverage,
    audit_trainable_scope,
    load_semantic_token_inventory,
    module_name_for_instance,
    prepare_semantic_tokens,
)


class FakeTokenizer:
    def __init__(self, *, existing: Sequence[str] = ()) -> None:
        self.vocabulary = {f"base-{index}": index for index in range(128)}
        for token in existing:
            self.vocabulary[token] = len(self.vocabulary)
        self.all_special_ids = [0, 1]

    def __len__(self) -> int:
        return len(self.vocabulary)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocabulary)

    def add_tokens(self, tokens: Sequence[str], special_tokens: bool = False) -> int:
        assert special_tokens is False
        added = 0
        for token in tokens:
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)
                added += 1
        return added

    def convert_tokens_to_ids(self, tokens: Sequence[str]) -> list[int]:
        return [self.vocabulary[token] for token in tokens]

    def encode(self, token: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [self.vocabulary[token]]


class FakeModel(torch.nn.Module):
    def __init__(self, *, tied: bool = False) -> None:
        super().__init__()
        self.input = torch.nn.Embedding(128, 8)
        self.output = torch.nn.Linear(8, 128, bias=False)
        if tied:
            self.output.weight = self.input.weight
        self.config = SimpleNamespace(tie_word_embeddings=tied)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.input

    def get_output_embeddings(self) -> torch.nn.Linear:
        return self.output

    def resize_token_embeddings(
        self, size: int, *, pad_to_multiple_of: int, mean_resizing: bool
    ) -> torch.nn.Embedding:
        assert mean_resizing is False
        padded = ((size + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        old_input = self.input.weight.detach()
        old_output = self.output.weight.detach()
        self.input = torch.nn.Embedding(padded, old_input.shape[1])
        self.output = torch.nn.Linear(old_output.shape[1], padded, bias=False)
        with torch.no_grad():
            self.input.weight[: old_input.shape[0]].copy_(old_input)
            self.output.weight[: old_output.shape[0]].copy_(old_output)
        return self.input


class AuditModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.language_model.layers[0].q_proj = torch.nn.Module()
        self.model.language_model.layers[0].q_proj.lora_A = torch.nn.ParameterDict(
            {"default": torch.nn.Parameter(torch.ones(4, 8))}
        )
        self.model.language_model.layers[0].q_proj.lora_B = torch.nn.ParameterDict(
            {"default": torch.nn.Parameter(torch.ones(8, 4))}
        )
        self.model.language_model.embed_tokens = torch.nn.Module()
        self.model.language_model.embed_tokens.token_adapter = torch.nn.Module()
        self.model.language_model.embed_tokens.token_adapter.trainable_tokens_delta = (
            torch.nn.ParameterDict(
                {"default": torch.nn.Parameter(torch.ones(154, 8))}
            )
        )
        self.lm_head = torch.nn.Linear(8, 16, bias=False)
        self.lm_head.token_adapter = torch.nn.Module()
        self.lm_head.token_adapter.trainable_tokens_delta = torch.nn.ParameterDict(
            {"default": torch.nn.Parameter(torch.ones(154, 8))}
        )
        self.model.language_model.base_weight = torch.nn.Parameter(
            torch.ones(8, 8), requires_grad=False
        )
        self.model.visual = torch.nn.Module()
        self.model.visual.block = torch.nn.Linear(8, 8)
        self.model.visual.merger = torch.nn.Linear(8, 8)
        self.lm_head.requires_grad_(False)
        self.lm_head.token_adapter.trainable_tokens_delta["default"].requires_grad_(True)
        self.model_meta = SimpleNamespace(
            model_arch=SimpleNamespace(
                language_model=["model.language_model"],
                vision_tower=["model.visual"],
                aligner=["model.visual.merger"],
                lm_head="lm_head",
            )
        )
        self._catan_semantic_token_setup = SimpleNamespace(
            token_ids=tuple(range(154)),
            as_dict=lambda: {"token_ids": list(range(154))},
        )
        attach_model_components(
            self,
            ModelComponentPaths(
                architecture="fixture",
                input_embedding="model.language_model.embed_tokens",
                output_head="lm_head",
                language=("model.language_model",),
                vision=("model.visual",),
                aligner=("model.visual.merger",),
                vocab_size=16,
                hidden_size=8,
            ),
        )


def test_semantic_inventory_loader_requires_exact_bidirectional_contract(tmp_path: Path) -> None:
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps(semantic_recognition_token_inventory()))

    assert load_semantic_token_inventory(path)["counts"]["total"] == 154

    payload = semantic_recognition_token_inventory()
    payload["tokens"] = payload["tokens"][:-1]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exact 154-token"):
        load_semantic_token_inventory(path)


def test_prepare_semantic_tokens_adds_regular_atomic_rows_and_freezes_base_output() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()

    setup = prepare_semantic_tokens(
        tokenizer,
        model,
        semantic_recognition_token_inventory(),
    )

    assert setup.added_tokens == 154
    assert setup.token_ids == tuple(range(128, 282))
    assert setup.tokenizer_size == 282
    assert setup.model_vocab_size == 384
    assert model.get_input_embeddings().weight.requires_grad
    assert not model.get_output_embeddings().weight.requires_grad
    assert not set(setup.token_ids) & set(tokenizer.all_special_ids)


def test_input_embedding_module_name_is_exact_and_unique() -> None:
    model = FakeModel()

    assert module_name_for_instance(model, model.get_input_embeddings()) == "input"
    with pytest.raises(ValueError, match=r"found \[\]"):
        module_name_for_instance(model, torch.nn.Embedding(2, 2))


def test_prepare_semantic_tokens_rejects_tied_or_partial_tokenizers() -> None:
    with pytest.raises(ValueError, match="requires untied"):
        prepare_semantic_tokens(
            FakeTokenizer(),
            FakeModel(tied=True),
            semantic_recognition_token_inventory(),
        )

    tokens = semantic_recognition_token_inventory()["tokens"]
    with pytest.raises(ValueError, match="only part"):
        prepare_semantic_tokens(
            FakeTokenizer(existing=tokens[:1]),
            FakeModel(),
            semantic_recognition_token_inventory(),
        )


def test_scope_audit_accepts_only_visual_aligner_atlas_rows_and_language_lora(tmp_path: Path) -> None:
    model = AuditModel()

    manifest = audit_trainable_scope(
        model,
        language_lora=True,
        output_path=tmp_path / "scope.json",
    )

    assert manifest["errors"] == []
    assert manifest["groups"]["atlas_input_rows"]["tensors"] == 1
    assert manifest["groups"]["atlas_output_rows"]["tensors"] == 1
    assert manifest["groups"]["vision"]["parameters"] > 0
    assert manifest["groups"]["aligner"]["parameters"] > 0
    assert manifest["groups"]["language_lora"]["parameters"] > 0
    assert json.loads((tmp_path / "scope.json").read_text())["schema"].endswith("/v2")


def test_scope_audit_rejects_base_language_and_output_updates(tmp_path: Path) -> None:
    model = AuditModel()
    model.model.language_model.base_weight.requires_grad_(True)
    model.lm_head.weight.requires_grad_(True)

    with pytest.raises(RuntimeError, match="forbidden trainable group"):
        audit_trainable_scope(
            model,
            language_lora=True,
            output_path=tmp_path / "scope.json",
        )

    assert json.loads((tmp_path / "scope.json").read_text())["errors"]


def test_optimizer_coverage_requires_every_trainable_exactly_once(tmp_path: Path) -> None:
    model = AuditModel()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)

    report = audit_optimizer_coverage(
        model,
        optimizer,
        output_path=tmp_path / "optimizer.json",
    )

    assert report["errors"] == []
    assert report["covered_tensors"] == report["trainable_tensors"]

    incomplete = torch.optim.AdamW(parameters[:-1], lr=1e-4)
    with pytest.raises(RuntimeError, match="omitted"):
        audit_optimizer_coverage(model, incomplete)
