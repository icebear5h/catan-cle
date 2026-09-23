import assert from "node:assert/strict";
import test from "node:test";
import { classifyCommit, guardDependencyCommand } from "../lib/shell.js";

test("standalone commit classification respects quoted message arguments", () => {
  for (const command of [
    'git commit -m "repair checks"', "git commit --amend --no-edit", "git commit -am 'repair'",
    "git commit --message='message with ; && | characters'", "git commit -m 'literal $HOME'",
    "git commit -m 'document pip install and git commit'", "git commit -m '--no-verify'",
  ]) assert.equal(classifyCommit(command), true, command);
  for (const command of ["git status", "git log --grep commit", "git diff", "uv lock"]) {
    assert.equal(classifyCommit(command), false, command);
  }
});

test("commits reject chains, control flow, substitutions, wrappers and alternate index/path modes", () => {
  for (const command of [
    'git add . && git commit -m repair', 'git commit -m repair && git status',
    'git status; git commit -m repair', 'git commit -m repair\ngit status',
    'git commit -m repair | cat', '(git commit -m repair)',
    'if true; then git commit -m repair; fi', 'git commit -m "$(git add .)"',
    'git commit -m `date`', 'git commit -m $MESSAGE', 'git commit -m repair > log',
    'git commit -m repair # comment', 'env GIT_INDEX_FILE=/tmp/index git commit -m repair',
    'GIT_WORK_TREE=/tmp git commit -m repair', 'git -C /tmp commit -m repair',
    'git --git-dir=/tmp/repo commit -m repair', 'git --work-tree /tmp commit -m repair',
    'git commit --only source.py', 'git commit --include source.py', 'git commit source.py',
    'git commit --pathspec-from-file=paths', 'git commit -p', 'sh -c "git commit -m repair"',
    'git commit -m "unterminated', 'git commit -m repair &',
  ]) assert.throws(() => classifyCommit(command), /standalone|Unsupported/, command);
});

test("ordinary verification bypasses are rejected", () => {
  for (const command of [
    "git commit --no-verify -m repair", "git commit -n -m repair", "git commit -anm repair",
    "git -c core.hooksPath=/dev/null commit -m repair",
    "git -ccore.hooksPath=/dev/null commit -m repair",
  ]) assert.throws(() => classifyCommit(command), /blocked/, command);
});

test("ordinary pip mutations are rejected even inside uv run", () => {
  for (const command of [
    "pip install numpy", "pip3 uninstall numpy", "python3 -m pip install numpy",
    "uv pip sync requirements.txt", "uv pip install numpy", "uv pip uninstall numpy",
    "uv run pip install numpy", "uv run python -m pip uninstall numpy",
    "uv run --no-sync python3.12 -m pip --quiet install numpy",
    "/tmp/venv/bin/pip3.12 --disable-pip-version-check install numpy",
    "true && pip install numpy",
  ]) assert.throws(() => guardDependencyCommand(command), /package mutation/, command);
});

test("obvious manifest shell writes are rejected, ordinary reads and uv commands pass", () => {
  for (const command of [
    "echo data > pyproject.toml", "cat data >> uv.lock", "tee pyproject.toml",
    "cp other pyproject.toml", "mv uv.lock backup", "rm uv.lock", "touch uv.lock",
    "sed -i '' s/a/b/ pyproject.toml", "perl -pi -e 's/a/b/' pyproject.toml",
    "sed --in-place s/a/b/ pyproject.toml", "git restore pyproject.toml",
    "git checkout HEAD -- uv.lock", "curl -o uv.lock https://example.org/lock",
    "uv pip compile pyproject.toml --output-file uv.lock",
    "python -c 'from pathlib import Path; Path(\"pyproject.toml\").write_text(\"x\")'",
    "python -c 'open(\"uv.lock\", \"w\").write(\"x\")'",
    'node -e \'fs.writeFileSync("uv.lock", "x")\'',
  ]) assert.throws(() => guardDependencyCommand(command), /Shell writes/, command);
  for (const command of [
    "cat pyproject.toml", "git diff -- uv.lock", "sed -n '1,20p' pyproject.toml",
    "python -c 'from pathlib import Path; print(Path(\"pyproject.toml\").read_text())'",
    "uv add numpy", "uv remove numpy", "uv lock", "uv sync --extra dev",
    "uv run python -m pip list", "uv pip list", "pip show numpy", "uv pip compile pyproject.toml",
  ]) assert.doesNotThrow(() => guardDependencyCommand(command), command);
});
