"""Offline launch-contract checks; use real corpus rows and a real tiny tokenizer."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Protocol, cast

import pytest
from transformers import PreTrainedTokenizerFast

from evals.catan_board_bench.tokens import atlas_tokens
from sft.cartesian_eval import SCHEMA as CARTESIAN_SCHEMA
from sft.cartesian_eval.scaling import SCHEMA as SCALING_SCHEMA
from sft.cartesian_eval.shorthand import SCHEMA as SHORTHAND_SCHEMA
from sft.miles_eval.contracts import (
    compact,
    json_list,
    json_object,
    parse_json,
    score,
    text,
)
from sft.miles_eval.data import read_panel
from sft.miles_eval.preflight import TokenizerView, check_token_lengths, serving_files
from sft.miles_eval.run import (
    MILES_REVISION,
    launch,
    miles_arguments,
    panel_spec,
    prepare,
    prepared_panels,
)
from sft.paths import PROJECT_ROOT


class FastTokenizerFactory(Protocol):
    def __call__(self, *, tokenizer_file: str, unk_token: str) -> PreTrainedTokenizerFast: ...


def prepare_real_row(tmp_path: Path) -> Path:
    corpus = PROJECT_ROOT / "artifacts/generated/sft/symbolic_board_v2/test.jsonl"
    source = tmp_path / "source.jsonl"
    source.write_text(corpus.read_text().splitlines()[0] + "\n")
    output = tmp_path / "prepared"
    prepare([("direction", source)], output)
    return output


def tiny_tokenizer(tmp_path: Path, *, add_atlas: bool) -> PreTrainedTokenizerFast:
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text(compact({
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
        "normalizer": None, "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None, "decoder": None,
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"},
    }))
    factory = cast("FastTokenizerFactory", PreTrainedTokenizerFast)
    tokenizer = factory(tokenizer_file=str(tokenizer_file), unk_token="[UNK]")
    if add_atlas:
        tokenizer.add_tokens(atlas_tokens())
    tokenizer.chat_template = (
        "{{ 'thinking' if enable_thinking else 'direct' }} "
        "{% for message in messages %}{{message.role}} {{message.content}} {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    return tokenizer


def test_prepared_manifest_and_single_gpu_eval_flags(tmp_path: Path) -> None:
    directory = prepare_real_row(tmp_path)
    manifest = parse_json((directory / "manifest.json").read_text())
    assert manifest["miles_revision"] == MILES_REVISION
    panels = prepared_panels(directory)
    args = miles_arguments(tmp_path / "checkpoint", panels, tmp_path / "output")
    for flag, value in {"--num-rollout": "0", "--eval-interval": "1", "--eval-num-gpus": "0",
                        "--rollout-num-gpus": "1", "--rollout-num-gpus-per-engine": "1",
                        "--eval-temperature": "0", "--eval-top-k": "1",
                        "--n-samples-per-eval-prompt": "1", "--eval-max-response-len": "512"}.items():
        assert args[args.index(flag) + 1] == value
    assert "--debug-rollout-only" in args and "--disable-rollout-global-dataset" in args
    assert args[args.index("--apply-chat-template-kwargs") + 1] == '{"enable_thinking":false}'
    assert "--eval-max-prompt-len" not in args  # Miles must not silently filter the panel.
    assert "--save" not in args and "--load" not in args
    assert args[args.index("--custom-rm-path") + 1] == "sft.miles_eval.hooks.reward"
    assert panel_spec(f"direction={panels['direction']}") == ("direction", panels["direction"])
    panels["direction"].write_text("{}\n")
    with pytest.raises(ValueError, match="changed"):
        prepared_panels(directory)


def test_dry_run_is_reproducible_and_starts_no_runtime(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    directory = prepare_real_row(tmp_path)
    launch(tmp_path / "miles checkout", tmp_path / "model", directory, tmp_path / "results", execute=False)
    printed = shlex.split(capsys.readouterr().out)
    assert printed[1:4] == ["-m", "sft.miles_eval.run", "run"]
    assert printed[-1] == "--execute"
    assert printed[printed.index("--miles-root") + 1] == str(tmp_path / "miles checkout")
    assert not (tmp_path / "results").exists()


def test_adapter_checkpoint_fails_before_model_load(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="merged/exported"):
        serving_files(tmp_path)


def test_real_tokenizer_budget_includes_header_and_full_completion(tmp_path: Path) -> None:
    directory = prepare_real_row(tmp_path)
    panels = prepared_panels(directory)
    tokenizer = tiny_tokenizer(tmp_path, add_atlas=True)
    row = read_panel(panels["direction"])[0]
    view = cast("TokenizerView", tokenizer)
    prompt = view.apply_chat_template(json_list(row["prompt"]), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    assert isinstance(prompt, str) and prompt.startswith("direct ")
    budget = len(tokenizer.encode(prompt, add_special_tokens=False)) + 8
    audit = check_token_lengths(view, panels, context=budget, new_tokens=8)
    assert len(json_list(audit["atlas_token_ids"])) == 154
    assert '"enable_thinking":false' in compact(audit)
    with pytest.raises(ValueError, match="truncation"):
        check_token_lengths(view, panels, context=budget - 1, new_tokens=8)


@pytest.mark.parametrize(("relative_path", "schema"), [
    ("cartesian_eval_v2/eval.jsonl", CARTESIAN_SCHEMA),
    ("cartesian_h_eval_v1/eval.jsonl", SHORTHAND_SCHEMA),
    ("cartesian_scaling_eval_v1/scaled_h.jsonl", SCALING_SCHEMA),
    ("cartesian_scaling_eval_v1/integer_xy.jsonl", SCALING_SCHEMA),
])
def test_cartesian_panel_uses_stock_tokenizer_and_exact_original_scoring(
    tmp_path: Path, relative_path: str, schema: str,
) -> None:
    source = PROJECT_ROOT / "artifacts/generated/sft" / relative_path
    directory = tmp_path / "cartesian"
    prepare([("cartesian", source)], directory)
    panels = prepared_panels(directory)
    tokenizer = tiny_tokenizer(tmp_path, add_atlas=False)
    view = cast("TokenizerView", tokenizer)
    before = tokenizer.get_vocab()
    audit = check_token_lengths(view, panels, context=4096, new_tokens=512)
    assert audit["atlas_token_ids"] == [] and audit["requires_atlas_tokenizer"] is False
    assert len(json_list(audit["token_lengths"])) == 200 and tokenizer.get_vocab() == before
    for row in read_panel(panels["cartesian"]):
        catan = json_object(json_object(row["metadata"])["catan"])
        metadata = json_object(catan["metadata"])
        gold = text(row["label"])
        assert metadata["schema"] == schema
        assert score(gold, gold, metadata)["correct"] is True
        assert score(gold, "unrequested explanation", metadata)["correct"] is False
    old_source = PROJECT_ROOT / "artifacts/generated/sft/coordinate_comparison_v1/paired.jsonl"
    legacy = tmp_path / "legacy"
    prepare([("legacy", old_source)], legacy)
    with pytest.raises(ValueError, match="atlas token"):
        check_token_lengths(view, prepared_panels(legacy), context=4096, new_tokens=512)
