"""Config, resume policy, row validation, and collation."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sft.scripts.train import train_trl_catan_vision as training

from .support import TextRow, text_row, text_tokenizer


def test_config_keeps_vision_defaults_and_requires_explicit_text_contract() -> None:
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
def test_text_resume_is_explicitly_rejected_and_cross_mode_vision_resume_fails(tmp_path: Path, saved_mode: str) -> None:
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
def test_text_rows_reject_media_controls_and_malformed_pairs(row: TextRow) -> None:
    with pytest.raises(ValueError):
        training._message_pair(row, line_number=7, input_mode="text")


def test_native_collation_masks_exact_boundary_and_supervises_eot_even_when_pad_equals_eot() -> None:
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


def test_native_boundary_mismatch_and_missing_eot_are_rejected() -> None:
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


def test_long_board_text_has_exact_budget_and_never_uses_image_or_patch_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
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
