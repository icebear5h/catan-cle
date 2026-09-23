"""Boundary, scope, and filesystem safety checks against actual Git inventories."""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.quality import check_structure, collect_inventory


def _write(root: Path, name: str, contents: str = "pass\n") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True)


def test_physical_line_boundary_and_documentation_scope(git_repo: Path) -> None:
    source = _write(git_repo, "new source/code.py", "pass\r\n" * 299 + "pass")
    _write(git_repo, "docs/guide.md", "documentation\n" * 301)
    passing = check_structure(collect_inventory(git_repo))
    assert passing.line_counts == {"new source/code.py": 300}
    assert passing.violations == ()
    assert "docs" not in passing.folder_counts

    source.write_bytes(b"pass\r" * 300 + b"pass")
    failing = check_structure(collect_inventory(git_repo))
    assert failing.line_counts == {"new source/code.py": 301}
    assert [(item.path, item.rule) for item in failing.violations] == [
        ("new source/code.py", "max-lines")
    ]


def test_folder_boundary_counts_all_direct_authored_files(git_repo: Path) -> None:
    _write(git_repo, "feature/code.ts", "export {};\n")
    for number in range(14):
        _write(git_repo, f"feature/note{number:02d}.md", "documentation\n" * 301)
    _write(git_repo, "feature/nested/child.py")
    passing = check_structure(collect_inventory(git_repo))
    assert passing.folder_counts["feature"] == 15
    assert passing.violations == ()

    _write(git_repo, "feature/settings.json", "{}\n")
    failing = check_structure(collect_inventory(git_repo))
    assert failing.folder_counts["feature"] == 16
    assert [(item.path, item.rule) for item in failing.violations] == [("feature", "max-files")]


def test_git_scope_deleted_ignored_untracked_and_ordering(git_repo: Path) -> None:
    _write(git_repo, ".gitignore", "ignored*.py\n")
    removed = _write(git_repo, "gone.py")
    _write(git_repo, "z tracked.py")
    _write(git_repo, "ignored tracked.py")
    _git(git_repo, "add", "--", ".")
    _git(git_repo, "add", "-f", "--", "ignored tracked.py")
    removed.unlink()
    _write(git_repo, "ignored untracked.py", "pass\n" * 301)
    _write(git_repo, "a untracked.js", "const value = 1;\n")
    inventory = collect_inventory(git_repo)
    assert inventory.files == (".gitignore", "a untracked.js", "ignored tracked.py", "z tracked.py")
    assert inventory.code_files == ("a untracked.js", "ignored tracked.py", "z tracked.py")
    assert inventory.python_files == ("ignored tracked.py", "z tracked.py")
    assert inventory.deleted_count == 1
    assert collect_inventory(git_repo) == inventory


def test_exclusions_unknown_source_roots_and_dotfiles(git_repo: Path) -> None:
    excluded = (
        ".venv/hidden.py",
        "web/node_modules/dependency.ts",
        "build/generated.js",
        "artifacts/generated.py",
        "data/input.py",
        "references/demo.py",
        "reports/report.html",
        "logs/debug.py",
        "output/plot.py",
        "evals/catan_board_bench/datasets/fixture.py",
        "playground/frontend/public/assets/bundle.js",
        "pkg/__pycache__/cached.py",
        "env/lib.py",
    )
    for name in excluded:
        _write(git_repo, name, "pass\n" * 301)
    included = (".opencode/plugin.ts", "cle/env/core.py", "surprise/new.mjs", "docs/example.py")
    for name in reversed(included):
        _write(git_repo, name, "\n" * 301)
    _git(git_repo, "add", "--", ".")
    inventory = collect_inventory(git_repo)
    assert inventory.code_files == tuple(sorted(included))
    assert inventory.excluded_count == len(excluded)
    result = check_structure(inventory)
    assert [item.path for item in result.violations] == sorted(included)


def test_declared_source_roots_are_governed_without_direct_code(git_repo: Path) -> None:
    for folder in (".", ".opencode", "scripts", "docs"):
        for number in range(16):
            _write(git_repo, f"{folder}/note{number:02d}.md", "notes\n")
    result = check_structure(collect_inventory(git_repo))
    assert result.folder_counts == {".": 16, ".opencode": 16, "scripts": 16}
    assert [(item.path, item.rule) for item in result.violations] == [
        (".", "max-files"),
        (".opencode", "max-files"),
        ("scripts", "max-files"),
    ]


def test_nested_authored_directories_are_not_blanket_excluded(git_repo: Path) -> None:
    sources = (
        "src/data/loader.py",
        "cle/cache/policy.py",
        "src/reports/render.py",
        "src/artifacts/writer.py",
        "src/references/lookup.py",
        "src/logs/handler.py",
        "src/output/formatter.py",
        "src/vendor/client.py",
        "src/third_party/adapter.py",
        "src/deps/resolve.py",
    )
    for name in sources:
        _write(git_repo, name, "pass\n" * 301)
    _git(git_repo, "add", "--", ".")

    inventory = collect_inventory(git_repo)
    assert inventory.python_files == tuple(sorted(sources))
    assert inventory.excluded_count == 0
    result = check_structure(inventory)
    assert result.line_counts == dict.fromkeys(sources, 301)
    assert result.folder_counts == {str(Path(name).parent): 1 for name in sources}
    assert [(item.path, item.rule) for item in result.violations] == [
        (name, "max-lines") for name in sorted(sources)
    ]


def test_gitlinks_and_symlinks_never_follow_targets(git_repo: Path, tmp_path: Path) -> None:
    outside = _write(tmp_path, "outside/secret.py", "pass\n" * 301)
    tracked = _write(git_repo, "swapped/secret.py")
    _git(git_repo, "add", "--", ".")
    tracked.unlink()
    tracked.parent.rmdir()
    tracked.parent.symlink_to(outside.parent, target_is_directory=True)
    (git_repo / "external.py").symlink_to(outside)
    (git_repo / "broken.txt").symlink_to(tmp_path / "missing")
    _write(git_repo, "external module/hidden.py", "pass\n" * 301)
    _git(
        git_repo, "update-index", "--add", "--cacheinfo", "160000," + "1" * 40 + ",external module"
    )

    inventory = collect_inventory(git_repo)
    assert inventory.submodules == ("external module",)
    assert inventory.symlinks == ("broken.txt", "external.py", "swapped")
    assert inventory.blocked_paths == ("swapped/secret.py",)
    assert inventory.code_files == ()
    result = check_structure(inventory)
    assert result.line_counts == {}
    assert result.folder_counts["."] == 3
    assert [(item.path, item.rule) for item in result.violations] == [
        ("external.py", "source-symlink"),
        ("swapped/secret.py", "unsafe-path"),
    ]
