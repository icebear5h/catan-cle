import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { cp, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import plugin from "../plugins/local-quality.js";
import { OUTPUT_LIMIT, requireSuccess, run } from "../lib/process.js";
import { requireAlignedIndex } from "../lib/quality.js";

const source = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const git = (root, ...args) => execFileSync("git", args, { cwd: root, encoding: "utf8" });

async function fixture(t, checker = true) {
  const root = await mkdtemp(join(tmpdir(), "quality plugin "));
  t.after(() => rm(root, { recursive: true, force: true }));
  git(root, "init", "-q");
  await writeFile(join(root, ".gitignore"), ".venv/\n__pycache__/\n.mypy_cache/\n.ruff_cache/\n");
  if (checker) {
    await mkdir(join(root, "scripts"));
    await cp(join(source, "scripts/quality"), join(root, "scripts/quality"), {
      recursive: true, filter: (path) => !path.includes("__pycache__"),
    });
    await symlink(join(source, ".venv"), join(root, ".venv"));
  }
  const hooks = await plugin({ directory: root, worktree: root });
  const before = (tool, args) => hooks["tool.execute.before"]({ tool, sessionID: "s", callID: "c" }, { args });
  const after = async (tool) => {
    const output = { output: "original tool output", title: "original title", metadata: { kept: true } };
    await hooks["tool.execute.after"]({ tool, sessionID: "s", callID: "c", args: {} }, output);
    return output;
  };
  return { root, before, after };
}

test("after hooks append actual strict structural debt with bounded details; repairs remain possible", async (t) => {
  const { root, before, after } = await fixture(t);
  await mkdir(join(root, "debt"));
  for (let i = 0; i < 16; i++) await writeFile(join(root, `debt/source${i}.py`), "pass\n".repeat(301));
  for (const tool of ["edit", "functions.write", "apply_patch", "functions.bash"]) {
    const output = await after(tool);
    assert.match(output.output, /^original tool output\n\n\[local-quality: structure-only, exit 1\]/);
    assert.match(output.output, /301 physical lines > 300/);
    assert.match(output.output, /16 direct authored files > 15/);
    assert.match(output.output, /detail lines omitted; all checks were evaluated/);
    assert.match(output.output, /Ruff\/mypy skipped: --structure-only/);
    assert.ok(output.output.length <= OUTPUT_LIMIT + 30);
    assert.equal(output.title, "original title");
    assert.deepEqual(output.metadata, { kept: true });
  }
  await before("functions.edit", { filePath: "debt/source0.py", oldString: "pass", newString: "" });
  await before("bash", { command: "git status" });
  assert.equal((await after("read")).output, "original tool output");
});

test("before hook rejects unsafe shell forms before invoking a missing checker", async (t) => {
  const { before } = await fixture(t, false);
  await assert.rejects(before("functions.bash", { command: "git add . && git commit -m repair" }), /standalone/);
  await assert.rejects(before("bash", { command: "git commit --no-verify -m repair" }), /blocked/);
  await assert.rejects(before("bash", { command: "uv run python -m pip install numpy" }), /package mutation/);
});

test("index gate rejects untracked and unstaged files, foreign repos and hidden index flags", async (t) => {
  const { root } = await fixture(t, false);
  await assert.rejects(requireAlignedIndex(root, root), /untracked/);
  await writeFile(join(root, "source.py"), "value = 1\n");
  git(root, "add", ".");
  await requireAlignedIndex(root, root);
  await writeFile(join(root, "source.py"), "value = 2\n");
  await assert.rejects(requireAlignedIndex(root, root), /unstaged tracked/);
  git(root, "add", ".");
  git(root, "update-index", "--assume-unchanged", "source.py");
  await assert.rejects(requireAlignedIndex(root, root), /assume-unchanged/);
  git(root, "update-index", "--no-assume-unchanged", "source.py");
  git(root, "update-index", "--skip-worktree", "source.py");
  await assert.rejects(requireAlignedIndex(root, root), /skip-worktree/);
  git(root, "update-index", "--no-skip-worktree", "source.py");
  const other = await fixture(t, false);
  await assert.rejects(requireAlignedIndex(root, other.root), /checked worktree/);
  await mkdir(join(root, "subdir"));
  await requireAlignedIndex(root, join(root, "subdir"));
  await writeFile(join(root, "outside-subdir.py"), "value = 1\n");
  await assert.rejects(requireAlignedIndex(root, join(root, "subdir")), /untracked/);
});

test("missing checker blocks aligned commits and is appended after tools", async (t) => {
  const { root, before, after } = await fixture(t, false);
  git(root, "add", ".");
  await assert.rejects(before("bash", { command: "git commit -m repair" }), /checker missing/);
  assert.match((await after("write")).output, /checker missing/);
});

test("full pre-commit CLI runs Ruff and strict mypy and blocks real staged structural debt", async (t) => {
  const { root, before } = await fixture(t);
  await writeFile(join(root, "broken.py"), 'def missing(value):\n    return value\n' + "\n".repeat(299));
  git(root, "add", ".");
  await assert.rejects(before("bash", { command: "git commit -m repair" }), (error) => {
    assert.match(error.message, /strict repository-wide quality gate failed/);
    assert.match(error.message, /301 physical lines > 300/);
    assert.match(error.message, /ruff: exit/);
    assert.match(error.message, /mypy: exit 1/);
    assert.doesNotMatch(error.message, /--structure-only/);
    return true;
  });
  assert.equal(await readFile(join(root, "broken.py"), "utf8"), 'def missing(value):\n    return value\n' + "\n".repeat(299));
});

test("clean staged worktree can pass the full pre-commit gate without executing a commit", async (t) => {
  const { root, before } = await fixture(t);
  await writeFile(join(root, "good.py"), "value: int = 1\n");
  git(root, "add", ".");
  await before("functions.bash", { command: "git commit -m repair" });
  assert.equal(git(root, "rev-list", "--all", "--count").trim(), "0");
});

test("checker process failures, timeout and excessive output fail closed", async () => {
  const missing = await run("/nonexistent/quality-checker", []);
  assert.throws(() => requireSuccess(missing, "blocked"), /Cannot execute/);
  const timed = await run(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { timeout: 30 });
  assert.throws(() => requireSuccess(timed, "blocked"), /Timed out/);
  const noisy = await run(process.execPath, ["-e", 'process.stdout.write("x".repeat(100000))']);
  assert.throws(() => requireSuccess(noisy, "blocked"), /truncated/);
  assert.ok(noisy.text.length <= OUTPUT_LIMIT + 30);
});
