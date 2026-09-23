import { access, lstat, readFile, readlink, realpath } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import { parsePatch, previewManifestPatch, renderManifestUpdate, renderOperation } from "./manifest-patch.js";
import { preserveBom, previewEdit } from "./manifest-text.js";
import { requireSuccess, run } from "./process.js";

export { previewEdit } from "./manifest-text.js";

export async function canonicalPath(path, depth = 0) {
  if (depth > 40) throw new Error(`Cannot resolve path: ${path}`);
  try {
    return await realpath(path);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    try {
      if ((await lstat(path)).isSymbolicLink()) {
        return canonicalPath(resolve(dirname(path), await readlink(path)), depth + 1);
      }
    } catch (missing) {
      if (missing.code !== "ENOENT") throw missing;
    }
    const parent = dirname(path);
    if (parent === path) throw error;
    return resolve(await canonicalPath(parent, depth + 1), basename(path));
  }
}

export async function protectedKind(path, directory, root) {
  const requested = resolve(directory, path);
  const canonical = await canonicalPath(requested);
  for (const name of ["uv.lock", "pyproject.toml"]) {
    if (basename(requested) === name || basename(canonical) === name ||
        canonical === await canonicalPath(resolve(root, name))) return name;
  }
  return null;
}

async function checkPreview(old, next, root) {
  try {
    await access(resolve(root, "scripts/quality/dependencies.py"));
  } catch (error) {
    throw new Error(`pyproject.toml preview rejected: unavailable checker (${error.message}).`);
  }
  // The stdlib preview validates TOML itself; uv must not reject incomplete project
  // metadata before a nondependency repair can reach that comparison.
  requireSuccess(await run("uv", ["run", "--no-project", "--no-sync", "python", "-m", "scripts.quality.dependencies"], {
    cwd: root, input: JSON.stringify({ old, new: next }),
  }), "pyproject.toml preview rejected (dependency changes or unavailable checker).");
}

async function guardPatch(args, directory, root) {
  const operations = parsePatch(args.patchText ?? args.patch);
  const paths = new Set(), rendered = [];
  let hasManifest = false;
  for (const operation of operations) {
    const kind = await protectedKind(operation.path, directory, root);
    const destination = operation.move && await protectedKind(operation.move, directory, root);
    if (kind === "uv.lock" || destination === "uv.lock") {
      throw new Error("apply_patch cannot touch uv.lock; use uv lock/sync/add/remove.");
    }
    for (const path of [operation.path, ...(operation.move ? [operation.move] : [])]) {
      const canonical = await canonicalPath(resolve(directory, path));
      if (paths.has(canonical)) throw new Error("Patch has duplicate/aliased target paths; combine hunks in one Update File.");
      paths.add(canonical);
    }
    if (!kind && !destination) { rendered.push(renderOperation(operation)); continue; }
    if (operation.type !== "Update File" || operation.move) {
      throw new Error("Manifest patches support Update File only; add/delete/move are blocked.");
    }
    hasManifest = true;
    const old = await readFile(resolve(directory, operation.path), "utf8");
    const next = previewManifestPatch(old, operation.body);
    await checkPreview(old, next, root);
    rendered.push(renderManifestUpdate(operation.path, old, next));
  }
  // Do not mutate arguments until every operation has passed. Rewriting excludes
  // host fuzzy matching from admitted manifest updates; external races remain out of scope.
  if (hasManifest) {
    args.patchText = ["*** Begin Patch", ...rendered, "*** End Patch"].join("\n");
    if ("patch" in args) args.patch = args.patchText;
  }
}

export async function guardManifest(tool, args, directory, root) {
  if (tool === "apply_patch") {
    return guardPatch(args, directory, root);
  }
  const path = args.filePath ?? args.file_path ?? args.path;
  if (typeof path !== "string") throw new Error(`Cannot inspect ${tool} file path.`);
  const kind = await protectedKind(path, directory, root);
  if (!kind) return;
  if (kind === "uv.lock") throw new Error("Direct uv.lock edits are blocked; use uv lock/sync/add/remove.");
  let old;
  try {
    old = await readFile(resolve(directory, path), "utf8");
  } catch (error) {
    if (error.code !== "ENOENT" || tool !== "write") throw error;
    old = "";
  }
  if (tool === "write" && typeof args.content !== "string") {
    throw new Error("pyproject.toml write requires full string content.");
  }
  const next = tool === "write" ? preserveBom(old, args.content) : previewEdit(old, args);
  await checkPreview(old, next, root);
}
