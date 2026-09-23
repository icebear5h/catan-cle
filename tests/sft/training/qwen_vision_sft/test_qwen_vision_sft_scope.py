"""Token patching and trainable scope audits."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sft.qwen_series_vision_sft import (
    VISION_LANGUAGE_LORA,
    VISION_ONLY,
)
from sft.scripts.eval.eval_qwen_vl_adapter import (
    normalize_non_lora_state_dict,
)
from sft.scripts.train.train_qwen_series_with_catan_tokens import (
    _audit_trainable_parameters,
    _language_token_target_names,
    _patch_peft_for_catan_tokens,
    _patch_peft_for_catan_tokens_only,
)

from .support import _AuditModel, _Parameter, _Tokenizer, _TokenModel


def test_catan_token_targets_include_untied_output_rows() -> None:
    assert _language_token_target_names(_TokenModel(tied=False)) == [
        "model.language_model.embed_tokens",
        "lm_head",
    ]
    assert _language_token_target_names(_TokenModel(tied=True)) == [
        "model.language_model.embed_tokens"
    ]


def test_language_lora_patch_adds_input_and_output_token_rows() -> None:
    captured: dict[str, object] = {}

    def get_peft_model(
        model: object, config: object, *args: object, **kwargs: object
    ) -> object:
        captured["config"] = config
        return model

    upstream = SimpleNamespace(get_peft_model=get_peft_model)
    config = SimpleNamespace(trainable_token_indices=None)
    model = _TokenModel(tied=False)
    _patch_peft_for_catan_tokens(upstream, _Tokenizer(), ["<N00>", "<RED>"])

    assert upstream.get_peft_model(model, config) is model
    assert config.trainable_token_indices == {
        "model.language_model.embed_tokens": [100, 101],
        "lm_head": [100, 101],
    }


def test_token_only_patch_replaces_lora_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class TrainableTokensConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_peft = SimpleNamespace(TrainableTokensConfig=TrainableTokensConfig)
    original_import = __import__(
        "sft.scripts.train.train_qwen_series_with_catan_tokens", fromlist=["importlib"]
    ).importlib.import_module

    def import_module(name: str) -> object:
        if name == "peft":
            return fake_peft
        return original_import(name)

    wrapper = __import__("sft.scripts.train.train_qwen_series_with_catan_tokens", fromlist=["importlib"])
    monkeypatch.setattr(wrapper.importlib, "import_module", import_module)

    def get_peft_model(
        model: object, config: object, *args: object, **kwargs: object
    ) -> object:
        captured["config"] = config
        return model

    upstream = SimpleNamespace(get_peft_model=get_peft_model)
    model = _TokenModel(tied=False)
    _patch_peft_for_catan_tokens_only(upstream, _Tokenizer(), ["<N00>"])

    assert upstream.get_peft_model(model, object()) is model
    assert captured["config"].kwargs == {
        "target_modules": ["model.language_model.embed_tokens", "lm_head"],
        "token_indices": [100],
        "init_weights": True,
    }


def test_trainable_scope_audit_accepts_both_intended_profiles(tmp_path: Path) -> None:
    shared = [
        ("base_model.model.model.visual.blocks.0.weight", _Parameter(10)),
        ("base_model.model.model.visual.merger.weight", _Parameter(5)),
        (
            "base_model.model.model.language_model.embed_tokens.token_adapter."
            "trainable_tokens_delta.default",
            _Parameter(3),
        ),
        (
            "base_model.model.lm_head.token_adapter.trainable_tokens_delta.default",
            _Parameter(3),
        ),
    ]
    vision_only = _audit_trainable_parameters(
        _AuditModel(shared),
        VISION_ONLY,
        tmp_path / "vision-only",
    )
    assert vision_only["errors"] == []

    joint = _audit_trainable_parameters(
        _AuditModel(
            shared
            + [
                (
                    "base_model.model.model.language_model.layers.0.q_proj."
                    "lora_A.default.weight",
                    _Parameter(4),
                )
            ]
        ),
        VISION_LANGUAGE_LORA,
        tmp_path / "joint",
    )
    assert joint["errors"] == []
    assert joint["groups"]["language_lora"]["parameters"] == 4


def test_trainable_scope_audit_rejects_language_base_updates(tmp_path: Path) -> None:
    model = _AuditModel(
        [
            ("model.visual.blocks.0.weight", _Parameter(10)),
            ("model.visual.merger.weight", _Parameter(5)),
            ("model.language_model.embed_tokens.token_adapter.weight", _Parameter(3)),
            ("lm_head.token_adapter.weight", _Parameter(3)),
            ("model.language_model.layers.0.weight", _Parameter(100)),
        ]
    )

    with pytest.raises(RuntimeError, match="base language parameters must remain frozen"):
        _audit_trainable_parameters(model, VISION_ONLY, tmp_path)

    recorded = json.loads((tmp_path / "trainable_parameters.json").read_text())
    assert recorded["errors"]


def test_non_lora_state_prefixes_match_base_qwen_model() -> None:
    normalized = normalize_non_lora_state_dict(
        {
            "base_model.model.model.visual.blocks.0.weight": "vision",
            "base_model.model.model.visual.merger.weight": "merger",
        }
    )

    assert normalized == {
        "model.visual.blocks.0.weight": "vision",
        "model.visual.merger.weight": "merger",
    }
