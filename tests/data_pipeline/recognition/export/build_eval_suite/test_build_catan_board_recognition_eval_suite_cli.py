"""CLI overwrite, preflight, and answer-only policy."""
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scripts.board_recognition.build_catan_board_recognition_eval_suite import (
    build_rows,
    main,
    read_jsonl,
    write_jsonl,
)
from sft.analysis.behavior_diagnostics import records_fingerprint

from .support import JsonRow


@pytest.mark.parametrize("answer_only", [False, True])
def test_cli_spatial_only_and_overwrite_policy(tmp_path: Path, supplement: tuple[Path, list[JsonRow], list[JsonRow]], monkeypatch: pytest.MonkeyPatch, answer_only: bool) -> None:
    source, _, _ = supplement
    output: Any = tmp_path / "eval.jsonl"
    smoke = tmp_path / "smoke.jsonl"
    argv = [
        "build_eval",
        "--root",
        str(tmp_path / "missing-base"),
        "--supplement-dir",
        str(source),
        "--spatial-only",
        "--output",
        str(output),
        "--smoke-output",
        str(smoke),
        "--smoke-rows",
        "1",
        *(["--answer-only"] if answer_only else []),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 0
    assert len(read_jsonl(output)) == 2
    assert all(
        (row["metadata"].get("prompt_variant") == "answer_only") == answer_only
        for row in read_jsonl(output)
    )
    assert len(read_jsonl(smoke)) == 1
    before = output.read_bytes(), smoke.read_bytes()
    with pytest.raises(FileExistsError, match="pass --overwrite"):
        main()
    assert (output.read_bytes(), smoke.read_bytes()) == before
    monkeypatch.setattr(sys, "argv", [*argv, "--overwrite"])
    assert main() == 0
    assert (output.read_bytes(), smoke.read_bytes()) == before


def test_cli_preflights_smoke_output_before_writing(tmp_path: Path, supplement: tuple[Path, list[JsonRow], list[JsonRow]], monkeypatch: pytest.MonkeyPatch) -> None:
    source, _, _ = supplement
    output = tmp_path / "new.jsonl"
    smoke = tmp_path / "existing.jsonl"
    write_jsonl(smoke, [{"id": "existing"}])
    before = smoke.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_eval",
            "--supplement-dir",
            str(source),
            "--spatial-only",
            "--output",
            str(output),
            "--smoke-output",
            str(smoke),
        ],
    )
    with pytest.raises(FileExistsError, match="pass --overwrite"):
        main()
    assert not output.exists()
    assert smoke.read_bytes() == before


@pytest.mark.parametrize("overwrite", [False, True])
def test_cli_rejects_same_output_and_smoke_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overwrite: bool) -> None:
    output = tmp_path / "eval.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_eval",
            "--output",
            str(output),
            "--smoke-output",
            str(output),
            *(["--overwrite"] if overwrite else []),
        ],
    )
    with pytest.raises(ValueError, match="different paths"):
        main()
    assert not output.exists()


def test_answer_only_requires_spatial_only_before_reads_or_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    missing_root = tmp_path / "missing"
    with pytest.raises(ValueError, match="--answer-only requires --spatial-only"):
        build_rows(missing_root, "validation", answer_only=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_eval",
            "--answer-only",
            "--overwrite",
            "--output",
            str(missing_root / "eval.jsonl"),
            "--smoke-output",
            str(missing_root / "smoke.jsonl"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--answer-only requires --spatial-only" in capsys.readouterr().err
    assert not missing_root.exists()


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_answer_only_is_neutral_and_does_not_mutate_sources(
    tmp_path: Path, supplement: tuple[Path, list[JsonRow], list[JsonRow]], monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    source, rows, audits = supplement
    audits[2].update(task_type=f"tile_adjacent_{answer}", answer=answer)
    rows[2]["messages"][1]["content"] = answer
    sources = {source / "validation.jsonl": rows, source / "audit/validation.jsonl": audits}
    before_sources = deepcopy(sources)
    monkeypatch.setattr(
        "scripts.board_recognition.build_catan_board_recognition_eval_suite.read_jsonl", sources.__getitem__
    )
    original: Any = build_rows(tmp_path, "validation", supplement_dir=source, spatial_only=True)
    before_original = deepcopy(original)
    control: Any = build_rows(
        tmp_path, "validation", supplement_dir=source, spatial_only=True, answer_only=True
    )
    assert sources == before_sources
    assert original == before_original
    instructions: Any = [
        "Answer with only one of the two tokens shown in the question.",
        "Answer with exactly one word: yes or no.",
    ]
    for base, row, instruction in zip(original, control, instructions, strict=True):
        assert row == {
            **base,
            "id": base["id"] + ":answer_only",
            "metadata": {**base["metadata"], "prompt_variant": "answer_only"},
            "messages": [
                {
                    **base["messages"][0],
                    "content": base["messages"][0]["content"] + "\n" + instruction,
                },
                *base["messages"][1:],
            ],
        }
        assert row["messages"][0] is not base["messages"][0]
    assert records_fingerprint(original) != records_fingerprint(control)
