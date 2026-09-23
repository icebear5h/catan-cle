"""Semantic dependency protection, including real repository manifest previews."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.quality.dependencies import check_change

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "pyproject.toml"


def test_real_manifest_allows_tool_configuration_and_formatting() -> None:
    old = MANIFEST.read_text(encoding="utf-8")
    new = old + '\n[tool.quality_test_fixture]\nenabled = true\ndate = 2026-09-21\n'
    assert check_change(old, new) == ()
    assert check_change(old, "# formatting only\n" + old.replace("\n", "\r\n")) == ()


def test_real_manifest_rejects_runtime_extra_and_build_dependencies() -> None:
    old = MANIFEST.read_text(encoding="utf-8")
    for before, after, field in (
        ('"numpy>=1.24.0"', '"numpy>=2.0.0"', "project.dependencies"),
        ('"modal>=0.68.0"', '"modal>=1.0.0"', "project.optional-dependencies"),
        ('requires = ["hatchling"]', 'requires = ["setuptools"]', "build-system.requires"),
    ):
        assert before in old
        assert check_change(old, old.replace(before, after)) == (field,)


@pytest.mark.parametrize(
    ("field", "source"),
    [
        ("project.dependencies", '[project]\ndependencies = ["numpy"]\n'),
        ("project.optional-dependencies", '[project.optional-dependencies]\ndev = ["ruff"]\n'),
        ("dependency-groups", '[dependency-groups]\ndev = ["ruff", {include-group = "lint"}]\n'),
        ("build-system.requires", '[build-system]\nrequires = ["hatchling"]\n'),
        ("tool.uv.sources", '[tool.uv.sources]\nnumpy = {git = "https://example.org/numpy"}\n'),
        ("tool.uv.override-dependencies", '[tool.uv]\noverride-dependencies = ["numpy"]\n'),
        ("tool.uv.constraint-dependencies", '[tool.uv]\nconstraint-dependencies = ["numpy"]\n'),
        ("tool.uv.dev-dependencies", '[tool.uv]\ndev-dependencies = ["ruff"]\n'),
        ("tool.uv.exclude-dependencies", '[tool.uv]\nexclude-dependencies = ["numpy"]\n'),
        (
            "tool.uv.build-constraint-dependencies",
            '[tool.uv]\nbuild-constraint-dependencies = ["numpy<2"]\n',
        ),
        (
            "tool.uv.extra-build-dependencies",
            '[tool.uv.extra-build-dependencies]\npackage = [{requirement = "numpy"}]\n',
        ),
    ],
)
def test_every_protected_field_rejects_addition_removal_and_nested_changes(
    field: str, source: str
) -> None:
    assert check_change("", source) == (field,)
    assert check_change(source, "") == (field,)
    replacement = source.replace("numpy", "pandas").replace("ruff", "mypy")
    replacement = replacement.replace("hatchling", "setuptools")
    assert check_change(source, replacement) == (field,)


def test_toml_dotted_keys_and_inline_tables_are_compared_semantically() -> None:
    old = 'tool.uv.sources = {foo = {git = "https://example.org/foo", rev = "abc"}}\n'
    new = '[tool.uv.sources.foo]\nrev = "abc"\ngit = "https://example.org/foo"\n'
    assert check_change(old, new) == ()
    assert check_change(old, new.replace('"abc"', '"def"')) == ("tool.uv.sources",)
    assert check_change("", "[project]\ndependencies = []\n") == ("project.dependencies",)


def test_bom_and_line_endings_do_not_hide_dependency_changes() -> None:
    old = '\ufeff[project]\r\ndependencies = ["numpy"]\r\n'
    assert check_change(old, '[project]\ndependencies = ["numpy"]\n') == ()
    assert check_change(old, old.replace("numpy", "pandas")) == ("project.dependencies",)


@pytest.mark.parametrize("marker", ["dependencies", "optional-dependencies"])
def test_dynamic_dependency_markers_are_protected_but_other_metadata_is_editable(marker: str) -> None:
    plain = '[project]\ndynamic = ["version"]\n'
    dynamic = f'[project]\ndynamic = ["version", "{marker}"]\n'
    assert check_change(plain, dynamic) == ("project.dynamic",)
    assert check_change(dynamic, plain) == ("project.dynamic",)
    assert check_change("", plain) == ()
    assert check_change(dynamic, dynamic.replace('"version"', '"description"')) == ()
    assert check_change(dynamic, f'[project]\ndynamic = ["{marker}", "version"]\n') == ()


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "[]",
        '{"old": ""}',
        '{"old": false, "new": ""}',
        '{"old": "", "new": "", "extra": true}',
        json.dumps({"old": "[broken", "new": ""}),
        json.dumps({"old": "", "new": "[broken"}),
        json.dumps({"old": "", "new": "project = 1"}),
        json.dumps({"old": "", "new": '[project]\ndynamic = "dependencies"'}),
        json.dumps({"old": "", "new": '[project]\ndynamic = [1]'}),
    ],
)
def test_cli_fails_closed_on_invalid_input(payload: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.quality.dependencies"],
        input=payload, cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "Invalid dependency preview" in result.stderr


def test_cli_exit_status_for_real_manifest_changes() -> None:
    old = MANIFEST.read_text(encoding="utf-8")
    for new, code in (
        (old + "\n# safe config edit\n", 0),
        (old.replace('"numpy>=1.24.0"', '"numpy>=2.0.0"'), 1),
    ):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.quality.dependencies"],
            input=json.dumps({"old": old, "new": new}), cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == code, result.stderr
