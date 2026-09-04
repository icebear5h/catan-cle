import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import sft.modal_qwen_series_vision_train as modal_vision_train
from evals.catan_board_bench.tokens import recognition_token_inventory
from sft.qwen_series_vision_sft import (
    DEFAULT_27B_MODEL_ID,
    H200_PROFILE,
    L40S_PROFILE,
    VISION_LANGUAGE_LORA,
    VISION_ONLY,
    VisionSftConfig,
    build_vision_sft_command,
    default_run_name,
    fingerprint_training_dataset,
    launch_identity,
    load_recognition_token_inventory,
    validate_hardware_profile,
)
from sft.scripts.eval_qwen_vl_adapter import (
    evaluation_image,
    evaluation_metadata,
    expected_text,
    load_non_lora_adapter_weights,
    normalize_non_lora_state_dict,
    user_text,
)
from sft.scripts.train_qwen_series_with_catan_tokens import (
    _audit_trainable_parameters,
    _language_token_target_names,
    _patch_peft_for_catan_tokens,
    _patch_peft_for_catan_tokens_only,
)


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _command(profile: str) -> list[str]:
    return build_vision_sft_command(
        VisionSftConfig(profile=profile),
        train_json="/data/train.json",
        image_folder="/data/images",
        output_dir="/runs/output",
    )


def test_joint_vision_sft_command_matches_full_tower_pattern():
    command = _command(VISION_LANGUAGE_LORA)

    assert _option(command, "--bits") == "16"
    assert _option(command, "--freeze_llm") == "True"
    assert _option(command, "--freeze_vision_tower") == "False"
    assert _option(command, "--freeze_merger") == "False"
    assert _option(command, "--enable_reasoning") == "False"
    assert _option(command, "--vision_lr") == "1e-06"
    assert _option(command, "--merger_lr") == "1e-05"
    assert _option(command, "--lora_enable") == "True"
    assert "--catan_token_adapter_only" not in command


def test_vision_only_uses_adapter_lifecycle_without_language_lora():
    config = VisionSftConfig(profile=VISION_ONLY)
    command = build_vision_sft_command(
        config,
        train_json="/data/train.json",
        image_folder="/data/images",
        output_dir="/runs/output",
    )

    assert "--catan_token_adapter_only" in command
    assert _option(command, "--lora_enable") == "True"
    assert config.language_lora is False
    assert config.as_manifest_dict()["freeze_llm"] is True


def test_full_epoch_command_omits_max_steps():
    config = VisionSftConfig(max_steps=None, num_train_epochs=2)
    command = build_vision_sft_command(
        config,
        train_json="train.json",
        image_folder="images",
        output_dir="output",
    )

    assert "--max_steps" not in command
    assert _option(command, "--num_train_epochs") == "2"
    assert default_run_name(config).endswith("epochs-2")


def test_invalid_profile_and_known_undersized_hardware_fail_closed():
    with pytest.raises(ValueError, match="Unsupported vision SFT profile"):
        VisionSftConfig(profile="unknown")
    with pytest.raises(ValueError, match="not admitted on one L40S"):
        validate_hardware_profile(DEFAULT_27B_MODEL_ID, L40S_PROFILE)

    validate_hardware_profile(DEFAULT_27B_MODEL_ID, H200_PROFILE)


def test_dataset_fingerprint_hashes_referenced_images():
    dataset = Path("artifacts/fixtures/sft/modal_vision_sft_smoke/train.jsonl")

    first = fingerprint_training_dataset(dataset)
    second = fingerprint_training_dataset(dataset)

    assert first == second
    assert first["rows"] == 4
    assert first["unique_images"] == 4
    assert first["max_prompt_characters"] < 512
    assert first["max_answer_characters"] == len("<BLUE> <SETTLEMENT>")
    assert len(first["images"]) == 4
    assert len(first["combined_sha256"]) == 64


def test_replay_fingerprint_uses_explicit_image_root_and_exact_token_inventory(
    tmp_path: Path,
):
    annotations = tmp_path / "annotations"
    images = tmp_path / "images"
    annotations.mkdir()
    images.mkdir()
    (images / "board.png").write_bytes(b"board-image")
    dataset = annotations / "train.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "image": "board.png",
                "conversations": [
                    {"from": "human", "value": "<image>\n<N00><Q_NODE_OCCUPANCY>"},
                    {"from": "gpt", "value": "<A_NODE_OCCUPANCY_EMPTY>"},
                ],
            }
        )
        + "\n"
    )
    inventory_path = tmp_path / "trainable_tokens.json"
    inventory_path.write_text(json.dumps(recognition_token_inventory()))

    fingerprint = fingerprint_training_dataset(
        dataset,
        image_root=images,
        token_inventory=inventory_path,
    )

    assert fingerprint["rows"] == 1
    assert fingerprint["unique_images"] == 1
    assert fingerprint["image_root"] == str(images)
    assert fingerprint["annotation_root"] == str(annotations)
    assert fingerprint["token_inventory"]["counts"]["total"] == 220
    assert load_recognition_token_inventory(inventory_path)["tokens"] == (
        recognition_token_inventory()["tokens"]
    )
    with pytest.raises(FileNotFoundError):
        fingerprint_training_dataset(
            dataset,
            image_root=tmp_path / "wrong-images",
            token_inventory=inventory_path,
        )


def test_vision_command_carries_remote_token_inventory():
    command = build_vision_sft_command(
        VisionSftConfig(profile=VISION_ONLY),
        train_json="/data/train.json",
        image_folder="/data/images",
        output_dir="/runs/output",
        token_inventory="/data/trainable_tokens.json",
    )

    assert _option(command, "--catan_token_inventory") == ("/data/trainable_tokens.json")


def test_dataset_fingerprint_requires_one_pair_and_one_image(tmp_path: Path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps({"id": "text-only", "messages": []}) + "\n")

    with pytest.raises(ValueError, match="exactly one user/assistant pair"):
        fingerprint_training_dataset(dataset)


def test_dataset_fingerprint_rejects_long_generation_targets(tmp_path: Path):
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "long-answer",
                "image": "image.png",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Read the board."},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "x" * 513}],
                    },
                ],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="short-answer limit"):
        fingerprint_training_dataset(dataset)


def test_spatial_eval_variants_blank_or_occlude_equal_regions(tmp_path: Path):
    image_path = tmp_path / "board.png"
    Image.new("RGB", (100, 100), (255, 255, 255)).save(image_path)
    row = {
        "image": str(image_path),
        "row_id": "probe",
        "task_type": "neutral_probe_token_return",
        "entity_type": "node",
        "spatial_targets": [
            {
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "control_bbox": [0.7, 0.7, 0.8, 0.8],
            }
        ],
    }

    blank = evaluation_image(
        row, variant="blank", shuffled_images=None, occlusion_margin=0.0
    )
    target = evaluation_image(
        row, variant="target_occlusion", shuffled_images=None, occlusion_margin=0.0
    )
    control = evaluation_image(
        row, variant="control_occlusion", shuffled_images=None, occlusion_margin=0.0
    )

    assert blank.getpixel((50, 50)) == (127, 127, 127)
    assert target.getpixel((15, 15)) == (42, 42, 42)
    assert target.getpixel((75, 75)) == (255, 255, 255)
    assert control.getpixel((75, 75)) == (42, 42, 42)
    assert evaluation_metadata(row, image_variant="target_occlusion") == {
        "entity_type": "node",
        "task_type": "neutral_probe_token_return",
        "eval_variant": "target_occlusion",
    }


def test_launch_identity_is_order_independent():
    assert launch_identity({"a": 1, "b": 2}) == launch_identity({"b": 2, "a": 1})


class _Module:
    def __init__(self, weight=None):
        self.weight = weight if weight is not None else object()


class _TokenModel:
    def __init__(self, *, tied: bool):
        input_weight = object()
        output_weight = input_weight if tied else object()
        self.embed = _Module(input_weight)
        self.output = _Module(output_weight)

    def named_modules(self):
        return [
            ("", self),
            ("model.language_model.embed_tokens", self.embed),
            ("lm_head", self.output),
        ]

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.output


def test_catan_token_targets_include_untied_output_rows():
    assert _language_token_target_names(_TokenModel(tied=False)) == [
        "model.language_model.embed_tokens",
        "lm_head",
    ]
    assert _language_token_target_names(_TokenModel(tied=True)) == [
        "model.language_model.embed_tokens"
    ]


class _Tokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, tokens):
        return list(range(100, 100 + len(tokens)))


def test_language_lora_patch_adds_input_and_output_token_rows():
    captured = {}

    def get_peft_model(model, config, *args, **kwargs):
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


def test_token_only_patch_replaces_lora_config(monkeypatch):
    captured = {}

    class TrainableTokensConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_peft = SimpleNamespace(TrainableTokensConfig=TrainableTokensConfig)
    original_import = __import__(
        "sft.scripts.train_qwen_series_with_catan_tokens", fromlist=["importlib"]
    ).importlib.import_module

    def import_module(name):
        if name == "peft":
            return fake_peft
        return original_import(name)

    wrapper = __import__("sft.scripts.train_qwen_series_with_catan_tokens", fromlist=["importlib"])
    monkeypatch.setattr(wrapper.importlib, "import_module", import_module)

    def get_peft_model(model, config, *args, **kwargs):
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


class _Parameter:
    def __init__(self, count: int, requires_grad: bool = True):
        self.count = count
        self.requires_grad = requires_grad

    def numel(self):
        return self.count


class _AuditModel:
    config = SimpleNamespace(tie_word_embeddings=False)

    def __init__(self, parameters):
        self.parameters = parameters

    def named_parameters(self):
        return iter(self.parameters)


def test_trainable_scope_audit_accepts_both_intended_profiles(tmp_path: Path):
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


def test_trainable_scope_audit_rejects_language_base_updates(tmp_path: Path):
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


def test_non_lora_state_prefixes_match_base_qwen_model():
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


def test_atomic_qwen_conversation_is_eval_compatible():
    row = {
        "image": "board.png",
        "conversations": [
            {"from": "human", "value": "<image>\n<E00_01><Q_EDGE_OWNER>"},
            {"from": "gpt", "value": "<A_EDGE_OWNER_EMPTY>"},
        ],
    }

    assert user_text(row) == "<E00_01><Q_EDGE_OWNER>"
    assert expected_text(row) == "<A_EDGE_OWNER_EMPTY>"


def test_new_vision_adapter_requires_non_lora_state(tmp_path: Path):
    (tmp_path / "launch_manifest.json").write_text(
        json.dumps({"schema": "catan_qwen_vision_sft_launch_identity/v1"})
    )

    with pytest.raises(FileNotFoundError, match="missing required non-LoRA"):
        load_non_lora_adapter_weights(object(), tmp_path, object())


def test_remote_launch_manifest_is_idempotent(monkeypatch, tmp_path: Path):
    commits = {"cache": 0, "runs": 0}
    calls = []

    monkeypatch.setattr(
        modal_vision_train,
        "hf_cache",
        SimpleNamespace(commit=lambda: commits.__setitem__("cache", commits["cache"] + 1)),
    )
    monkeypatch.setattr(
        modal_vision_train,
        "sft_runs",
        SimpleNamespace(commit=lambda: commits.__setitem__("runs", commits["runs"] + 1)),
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output_dir = Path(_option(command, "--output_dir"))
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "adapter_config.json",
            "tokenizer_config.json",
            "trainable_parameters.json",
        ):
            (output_dir / name).write_text("{}\n")
        (output_dir / "adapter_model.safetensors").write_bytes(b"adapter")
        (output_dir / "non_lora_state_dict.bin").write_bytes(b"vision")

    monkeypatch.setattr(modal_vision_train.subprocess, "run", fake_run)
    config = VisionSftConfig(
        profile=VISION_ONLY,
        model_id="Qwen/Qwen3-VL-4B-Instruct",
    )
    dataset = {
        "source_sha256": "a" * 64,
        "combined_sha256": "b" * 64,
        "rows": 4,
        "unique_images": 4,
        "max_prompt_characters": 128,
        "max_answer_characters": 20,
        "annotation_root": "artifacts/annotations",
        "image_root": "artifacts/images",
        "token_inventory": {
            "reference": "tokens.json",
            "sha256": "c" * 64,
            "counts": {"atlas": 154, "query": 6, "answer": 60, "total": 220},
        },
    }
    kwargs = {
        "hardware": L40S_PROFILE,
        "train_json": "/data/train.json",
        "image_folder": "/data/images",
        "output_dir": str(tmp_path / "run"),
        "token_inventory": "/data/trainable_tokens.json",
        "config_payload": asdict(config),
        "dataset": dataset,
    }

    first = modal_vision_train._run_remote(**kwargs)
    second = modal_vision_train._run_remote(**kwargs)

    assert first["status"] == "completed"
    assert second["status"] == "already_completed"
    assert len(calls) == 1
    assert commits == {"cache": 1, "runs": 1}
    manifest = json.loads((tmp_path / "run/launch_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["identity"] == first["identity"]
