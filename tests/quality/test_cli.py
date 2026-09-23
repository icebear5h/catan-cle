"""CLI output caps, fail-closed Git handling, and real checker integration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from scripts.quality.cli import main


def test_output_limit_preserves_full_evaluation(
    git_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("z.py", "a.py"):
        (git_repo / name).write_text("pass\n" * 301, encoding="utf-8")
    assert main(["--structure-only", "--json", "--limit", "1"], root=git_repo) == 1
    limited = json.loads(capsys.readouterr().out)
    assert limited["structure"]["violation_count"] == 2
    assert limited["structure"]["line_counts"] == {"a.py": 301, "z.py": 301}
    assert limited["detail_count"] == 2
    assert limited["omitted_details"] == 1
    assert [item["path"] for item in limited["details"]] == ["a.py"]
    assert limited["tools"] == []

    assert main(["--structure-only", "--json"], root=git_repo) == 1
    complete = json.loads(capsys.readouterr().out)
    assert complete["structure"] == limited["structure"]
    assert [item["path"] for item in complete["details"]] == ["a.py", "z.py"]
    assert main(["--structure-only", "--limit", "0"], root=git_repo) == 1
    text = capsys.readouterr().out
    assert "2 detail lines omitted; all checks were evaluated" in text
    assert "a.py: max-lines" not in text


def test_git_errors_fail_closed_instead_of_falling_back(
    git_repo: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--structure-only", "--json"], root=tmp_path) == 2
    outside = json.loads(capsys.readouterr().out)
    assert outside["ok"] is False
    assert "git rev-parse" in outside["error"]

    (git_repo / ".git/index").write_bytes(b"corrupt index")
    assert main(["--structure-only", "--json"], root=git_repo) == 2
    corrupt = json.loads(capsys.readouterr().out)
    assert corrupt["ok"] is False
    assert "git ls-files" in corrupt["error"]


@pytest.mark.skipif(
    importlib.util.find_spec("mypy") is None or importlib.util.find_spec("ruff") is None,
    reason="real checker integration requires the dev dependencies",
)
def test_full_cli_runs_real_ruff_and_strict_mypy_after_structure_failure(
    git_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (git_repo / "pyproject.toml").write_text(
        '[tool.ruff.lint]\nselect = ["E", "F", "ANN"]\n',
        encoding="utf-8",
    )
    left = git_repo / "left/worker.py"
    right = git_repo / "right/worker.py"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_text("def missing(value):\n    return value\n" + "\n" * 299, encoding="utf-8")
    right.write_text('def wrong() -> int:\n    return "wrong"\n', encoding="utf-8")

    assert main(["--json", "--limit", "0"], root=git_repo) == 1
    failing = json.loads(capsys.readouterr().out)
    assert failing["structure"]["violation_count"] == 1
    assert [(tool["name"], tool["returncode"]) for tool in failing["tools"]] == [
        ("ruff", 1),
        ("mypy", 1),
    ]
    assert failing["details"] == []
    assert failing["omitted_details"] > 1
    for tool in failing["tools"]:
        assert tool["command"][:3] == [sys.executable, "-m", tool["name"]]

    left.write_text("def identity(value: int) -> int:\n    return value\n", encoding="utf-8")
    right.write_text("def correct() -> int:\n    return 1\n", encoding="utf-8")
    assert main(["--json"], root=git_repo) == 0
    passing = json.loads(capsys.readouterr().out)
    assert passing["ok"] is True
    assert [tool["returncode"] for tool in passing["tools"]] == [0, 0]
