import { access } from "node:fs/promises";
import { resolve } from "node:path";
import { canonicalPath } from "./manifests.js";
import { requireSuccess, run } from "./process.js";

export async function quality(root, structureOnly) {
  try {
    await access(resolve(root, "scripts/quality/__main__.py"));
  } catch (error) {
    return { code: 2, text: `Quality checker missing/unreadable: ${error.message}`, truncated: false };
  }
  const args = ["run", "--no-sync", "python", "-m", "scripts.quality"];
  if (structureOnly) args.push("--structure-only");
  args.push("--limit", structureOnly ? "12" : "40");
  return run("uv", args, { cwd: root, timeout: structureOnly ? 30_000 : 120_000 });
}

export async function requireAlignedIndex(root, cwd) {
  const override = Object.keys(process.env).find((name) =>
    /^GIT_(?:INDEX|DIR$|WORK_TREE$|COMMON_DIR$|OBJECT|ALTERNATE|CONFIG|NAMESPACE)/.test(name));
  if (override) throw new Error(`Commit blocked: alternate Git environment ${override}.`);
  const top = requireSuccess(await run("git", ["rev-parse", "--show-toplevel"], { cwd }),
    "Cannot identify commit repository.");
  if (await canonicalPath(top.replace(/\n$/, "")) !== await canonicalPath(root)) {
    throw new Error("Commit workdir must belong to the plugin's checked worktree.");
  }
  // ls-files otherwise limits its inventory to a requested subdirectory.
  const git = async (...args) => run("git", args, { cwd: root, maxOutput: 4 * 1024 * 1024 });
  const flags = requireSuccess(await git("ls-files", "-v", "-z"), "Cannot inspect Git index flags.");
  if (flags.split("\0").some((line) => /^[a-zS] /.test(line))) {
    throw new Error("Commit blocked: clear assume-unchanged/skip-worktree index flags before checking.");
  }
  requireSuccess(await git("diff", "--quiet", "--no-ext-diff", "--ignore-submodules=none", "--"),
    "Commit blocked: unstaged tracked changes (or Git failure). Stage complete repairs first.");
  const untracked = requireSuccess(await git("ls-files", "--others", "--exclude-standard", "-z"),
    "Cannot inspect untracked files.");
  if (untracked) throw new Error("Commit blocked: nonignored untracked files remain. Stage or intentionally ignore them.");
  const conflicts = requireSuccess(await git("ls-files", "--unmerged", "-z"), "Cannot inspect index conflicts.");
  if (conflicts) throw new Error("Commit blocked: unresolved index conflicts.");
}
