import assert from "node:assert/strict";
import { cp, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { guardManifest, previewEdit } from "../lib/manifests.js";

const source = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), "quality manifests "));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, "scripts/quality"), { recursive: true });
  await cp(join(source, "scripts/quality/dependencies.py"), join(root, "scripts/quality/dependencies.py"));
  await symlink(join(source, ".venv"), join(root, ".venv"));
  const manifest = await readFile(join(source, "pyproject.toml"), "utf8");
  await writeFile(join(root, "pyproject.toml"), manifest);
  await writeFile(join(root, "uv.lock"), "test lock\n");
  return { root, manifest, guard: (tool, args) => guardManifest(tool, args, root, root) };
}

test("edit previews use pinned host exact matching and native replaceAll dollar semantics", () => {
  assert.throws(() => previewEdit("a = 1", { oldString: "a=1", newString: "a=2" }), /exact match/);
  assert.throws(() => previewEdit("xx", { oldString: "x", newString: "y" }), /ambiguous/);
  assert.throws(() => previewEdit("abc", { oldString: "", newString: "xyz" }), /nonempty/);
  assert.equal(previewEdit("xx", { oldString: "x", newString: "$&", replaceAll: true }), "xx");
  assert.equal(previewEdit("abc", { oldString: "b", newString: "$&" }), "a$&c");
  for (const [replacement, expected] of [["$$", "a$c"], ["$`", "aac"], ["$'", "acc"], ["$1", "a$1c"]]) {
    assert.equal(previewEdit("abc", { oldString: "b", newString: replacement, replaceAll: true }), expected);
    assert.equal(previewEdit("abc", { oldString: "b", newString: replacement }), `a${replacement}c`);
  }
});

test("edit previews preserve BOM and normalize search/replacement to the host line ending", () => {
  assert.equal(previewEdit('\ufeffa = 1\r\nb = 2\r\n', {
    oldString: 'a = 1\nb = 2', newString: 'a = 2\nb = 2',
  }), '\ufeffa = 2\r\nb = 2\r\n');
  assert.equal(previewEdit('a = 1\nb = 2\n', {
    oldString: 'a = 1\r\nb = 2', newString: '\ufeffa = 2\r\nb = 2',
  }), '\ufeffa = 2\nb = 2\n');
  assert.throws(() => previewEdit('\ufeffa = 1', {
    oldString: '\ufeffa = 1', newString: 'a = 2',
  }), /exact match/);
});

test("real manifest edit/write preview allows configuration but rejects dependency fields", async (t) => {
  const { root, manifest, guard } = await fixture(t);
  await guard("edit", { filePath: "pyproject.toml", oldString: 'version = "0.1.0"', newString: 'version = "0.1.1"' });
  await guard("write", { filePath: "pyproject.toml", content: manifest + "\n[tool.quality_fixture]\nenabled = true\n" });
  await assert.rejects(guard("edit", {
    filePath: "pyproject.toml", oldString: '"numpy>=1.24.0"', newString: '"numpy>=2.0.0"',
  }), /project.dependencies/);
  await assert.rejects(guard("write", {
    filePath: "pyproject.toml", content: manifest + '\n[dependency-groups]\ncheck = ["ruff"]\n',
  }), /dependency-groups/);
  await assert.rejects(guard("write", { filePath: "pyproject.toml", content: "[invalid" }), /Invalid dependency/);
  await assert.rejects(guard("edit", { filePath: "pyproject.toml", oldString: "no match", newString: "x" }), /exact match/);
  assert.equal(await readFile(join(root, "pyproject.toml"), "utf8"), manifest, "previews never write files");
});

test("direct lockfile edits and symlink aliases to either protected file are blocked", async (t) => {
  const { root, guard } = await fixture(t);
  await symlink("pyproject.toml", join(root, "alias.txt"));
  await symlink("uv.lock", join(root, "lock-alias.txt"));
  await symlink(root, join(root, "directory-alias"));
  for (const path of ["uv.lock", "lock-alias.txt", "directory-alias/uv.lock"]) {
    await assert.rejects(guard("write", { filePath: path, content: "anything" }), /Direct uv.lock/);
  }
  await assert.rejects(guard("write", { filePath: "alias.txt", content: "" }), /Dependency edit blocked/);
  await guard("write", { filePath: "ordinary.py", content: "pass\n" });
});

test("mixed CRLF/LF duplicate regression inspects the dependency span selected by the host", async (t) => {
  const { root, guard } = await fixture(t);
  const source = '[project]\r\ndependencies = [\r\n  "numpy",\r\n]\r\n' +
    '[tool.example]\nnote = [\n  "numpy",\n]\n';
  await writeFile(join(root, "pyproject.toml"), source);
  await assert.rejects(guard("edit", {
    filePath: "pyproject.toml", oldString: '  "numpy",\n]', newString: '  "pandas",\n]',
  }), /project.dependencies/);
});

test("BOM manifests permit configuration edits and reject dependencies, including dollar expansion", async (t) => {
  const { root, guard } = await fixture(t);
  const source = '\ufeff[project]\ndependencies = ["numpy"]\n[tool.example]\ntext = "alpha"\n';
  await writeFile(join(root, "pyproject.toml"), source);
  await guard("edit", { filePath: "pyproject.toml", oldString: 'text = "alpha"', newString: 'text = "beta"' });
  await guard("write", { filePath: "pyproject.toml", content: source.replace('"alpha"', '"beta"') });
  await guard("edit", { filePath: "pyproject.toml", oldString: '"numpy"', newString: "$&", replaceAll: true });
  await assert.rejects(guard("edit", {
    filePath: "pyproject.toml", oldString: "alpha", newString: "$'", replaceAll: true,
  }), /Invalid dependency/);
  await assert.rejects(guard("edit", {
    filePath: "pyproject.toml", oldString: '"numpy"', newString: '"$&"', replaceAll: true,
  }), /Invalid dependency/);
  await assert.rejects(guard("write", {
    filePath: "pyproject.toml", content: source.replace('"numpy"', '"pandas"'),
  }), /project.dependencies/);
});

test("missing dependency checker fails closed while ordinary repair edits stay available", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "missing dependency checker "));
  t.after(() => rm(root, { recursive: true, force: true }));
  await writeFile(join(root, "pyproject.toml"), "# no dependencies\n");
  await assert.rejects(guardManifest("write", {
    filePath: "pyproject.toml", content: "# config change\n",
  }, root, root), /unavailable checker/);
  await guardManifest("edit", { filePath: "repair.py" }, root, root);
});
