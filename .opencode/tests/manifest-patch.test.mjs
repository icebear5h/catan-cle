import assert from "node:assert/strict";
import { cp, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import plugin from "../plugins/local-quality.js";
import { parsePatch, previewManifestPatch } from "../lib/manifest-patch.js";

const source = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const manifest = '[project]\ndependencies = ["numpy"]\n[tool.ruff]\nline-length = 100\n';
const config = '@@\n [tool.ruff]\n-line-length = 100\n+line-length = 99';
const dependency = '@@\n [project]\n-dependencies = ["numpy"]\n+dependencies = ["pandas"]';
const patch = (...operations) => ["*** Begin Patch", ...operations, "*** End Patch"].join("\n");

async function fixture(t, content = manifest) {
  const root = await mkdtemp(join(tmpdir(), "manifest patch "));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, "scripts/quality"), { recursive: true });
  await cp(join(source, "scripts/quality/dependencies.py"), join(root, "scripts/quality/dependencies.py"));
  await symlink(join(source, ".venv"), join(root, ".venv"));
  await writeFile(join(root, "pyproject.toml"), content);
  await writeFile(join(root, "uv.lock"), "lock\n");
  const hooks = await plugin({ directory: root, worktree: root });
  const guard = (args) => hooks["tool.execute.before"]({ tool: "functions.apply_patch" }, { args });
  return { root, guard };
}

test("ordinary manifest config patches are admitted and rewritten to exact full-file replacement", async (t) => {
  const { root, guard } = await fixture(t);
  for (const separator of ["", " ", "\t", "  \t"]) {
    const args = { patchText: patch(`*** Update File:${separator}pyproject.toml\n${config}`) };
    await guard(args);
    assert.equal(args.patchText, patch('*** Update File: pyproject.toml\n@@\n' +
      '-[project]\n-dependencies = ["numpy"]\n-[tool.ruff]\n-line-length = 100\n' +
      '+[project]\n+dependencies = ["numpy"]\n+[tool.ruff]\n+line-length = 99'));
  }
  assert.equal(await readFile(join(root, "pyproject.toml"), "utf8"), manifest);
  await assert.rejects(guard({ patchText: patch(`*** Update File:pyproject.toml\n${dependency}`) }), /project.dependencies/);
});

test("multiple exact hunks and mixed file operations survive manifest rewrite", async (t) => {
  const old = manifest + '[tool.example]\ntext = "first"\n';
  const { guard } = await fixture(t, old);
  const other = '*** Add File: ordinary.txt\n+*** Update File:uv.lock\n+*** Move to:pyproject.toml';
  const args = { patchText: patch(other,
    `*** Update File: pyproject.toml\n${config}\n@@ [tool.example]\n-text = "first"\n+text = "$& $$"`,
    '*** Update File: ordinary.py\n@@\n-old\n+new') };
  await guard(args);
  const operations = parsePatch(args.patchText);
  assert.equal(operations.length, 3);
  assert.equal(operations[0].type, "Add File");
  assert.deepEqual(operations[0].body, ["+*** Update File:uv.lock", "+*** Move to:pyproject.toml"]);
  assert.ok(operations[1].body.includes('+text = "$& $$"'), "patch replacements are literal");
  assert.deepEqual(operations[2].body, ["@@", "-old", "+new"]);
});

test("all operations must pass; duplicate and symlink-aliased manifest targets are rejected", async (t) => {
  const { root, guard } = await fixture(t);
  await mkdir(join(root, "nested"));
  await writeFile(join(root, "nested/pyproject.toml"), manifest);
  await symlink("pyproject.toml", join(root, "alias.txt"));
  const valid = `*** Update File: pyproject.toml\n${config}`;
  const args = { patchText: patch(valid, `*** Update File: nested/pyproject.toml\n${config}`) };
  await guard(args);
  assert.equal(parsePatch(args.patchText).length, 2);
  for (const second of [
    `*** Update File: nested/pyproject.toml\n${dependency}`, '*** Add File:uv.lock\n+x',
    `*** Update File: pyproject.toml\n${config}`, `*** Update File: alias.txt\n${config}`,
  ]) {
    const original = patch(valid, second), denied = { patchText: original };
    await assert.rejects(guard(denied), /project.dependencies|uv.lock|duplicate\/aliased/);
    assert.equal(denied.patchText, original, "failed batches must not modify tool arguments");
  }
  await assert.rejects(guard({ patchText: patch(`*** Update File:alias.txt\n${dependency}`) }), /project.dependencies/);
});

test("lock operations and manifest add/delete/move are protected regardless of header spacing", async (t) => {
  const { guard } = await fixture(t);
  for (const separator of ["", " ", "\t"]) {
    for (const body of [
      `*** Add File:${separator}uv.lock\n+x`, `*** Delete File:${separator}uv.lock`,
      `*** Update File:${separator}uv.lock\n@@\n-lock\n+changed`,
      `*** Update File:note.txt\n*** Move to:${separator}uv.lock\n@@\n-a\n+b`,
      `*** Add File:${separator}pyproject.toml\n+x`, `*** Delete File:${separator}pyproject.toml`,
      `*** Update File:note.txt\n*** Move to:${separator}pyproject.toml\n@@\n-a\n+b`,
      `*** Update File:${separator}pyproject.toml\n*** Move to:note.txt\n${config}`,
    ]) await assert.rejects(guard({ patchText: patch(body) }), /uv.lock|add\/delete\/move/);
  }
});

test("malformed or hidden operation syntax fails closed; prefixed headers remain content", async (t) => {
  const { guard } = await fixture(t);
  for (const body of [
    '*** Add File:\n+x', '*** Add File uv.lock\n+x', '*** Add File :uv.lock\n+x',
    '***\tAdd File:uv.lock\n+x', '*** Move to:pyproject.toml',
    `*** Update File:pyproject.toml\n*** Move to:\n${config}`,
    '*** Begin Patch\n*** Add File:note.txt\n+x', '*** End Patch\n*** Add File:uv.lock\n+x',
    '*** Add File:note.txt\nunprefixed ignored text\n*** Add File:uv.lock\n+x',
    '*** Update File:note.txt\n@@\n *** End Patch\n-old\n+new',
  ]) await assert.rejects(guard({ patchText: patch(body) }), /patch rejected/);
  await guard({ patchText: patch('*** Add File:note.txt\n+*** Add File:uv.lock\n+*** End Patch') });
  await guard({ patchText: patch('*** Update File:note.txt\n@@\n-*** Add File:uv.lock\n+*** Move to:pyproject.toml') });
});

test("fuzzy, ambiguous, context-free and out-of-order manifest hunks are rejected", () => {
  for (const body of [
    ['@@', '-line-length=100', '+line-length = 99'],
    ['@@', '- line-length = 100', '+line-length = 99'],
    ['@@', '+[tool.new]', '+enabled = true'],
    ['@@', ' [project]'],
    ['@@ missing anchor', '-line-length = 100', '+line-length = 99'],
    ['@@', '-line-length = 100', '+line-length = 99', '@@', '-[project]', '+[project]'],
  ]) assert.throws(() => previewManifestPatch(manifest, body), /exact context|old context|missing update/);
  const duplicate = '[a]\nx = 1\n[b]\nx = 1\n';
  assert.throws(() => previewManifestPatch(duplicate, ['@@', '-x = 1', '+x = 2']), /ambiguous/);
  assert.throws(() => previewManifestPatch(manifest, ['@@', '-[project]', '+[project]', '*** End of File']), /EOF/);
});

test("CRLF/BOM patches rewrite raw full-file context; normalized duplicate spans cannot redirect changes", async (t) => {
  const old = '\ufeff' + manifest.replaceAll('\n', '\r\n');
  const { guard } = await fixture(t, old);
  const args = { patchText: patch(`*** Update File:\tpyproject.toml\n${config}`).replaceAll('\n', '\r\n') };
  await guard(args);
  assert.ok(args.patchText.includes('-[project]\r\n-dependencies = ["numpy"]\r\n'));
  assert.ok(args.patchText.includes('+\ufeff[project]\r\n+dependencies = ["numpy"]\r\n'));
  assert.ok(args.patchText.includes('+line-length = 99\r\n*** End Patch'));
  await assert.rejects(guard({ patchText: patch(`*** Update File:pyproject.toml\n${dependency}`) }), /project.dependencies/);
  const mixed = '[project]\r\ndependencies = [\r\n  "numpy",\r\n]\r\n' +
    '[tool.example]\nnote = [\n  "numpy",\n]\n';
  assert.throws(() => previewManifestPatch(mixed, ['@@', '-  "numpy",', '+  "pandas",', ' ]']), /ambiguous/);
});
