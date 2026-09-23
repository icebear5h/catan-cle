import { resolve } from "node:path";
import { canonicalPath, guardManifest } from "../lib/manifests.js";
import { bounded, requireSuccess } from "../lib/process.js";
import { quality, requireAlignedIndex } from "../lib/quality.js";
import { classifyCommit, guardDependencyCommand } from "../lib/shell.js";

// Bounded workflow guard, not a sandbox or an installed Git hook. Shell aliases,
// scripts, other tools/processes and races can evade it; metadata is not uv provenance.
// Reports are advisory after edits so a red repository can always be repaired.
export default async ({ directory, worktree }) => {
  const root = await canonicalPath(resolve(worktree || directory));
  const base = resolve(directory);
  const name = (tool) => tool.replace(/^functions\./, "");
  const mutations = new Set(["edit", "write", "apply_patch"]);
  return {
    "tool.execute.before": async (input, output) => {
      const tool = name(input.tool);
      const args = output.args;
      if (mutations.has(tool)) await guardManifest(tool, args, base, root);
      if (tool !== "bash") return;
      if (typeof args.command !== "string") throw new Error("Cannot inspect bash command.");
      guardDependencyCommand(args.command);
      if (!classifyCommit(args.command)) return;
      const cwd = resolve(base, args.workdir || ".");
      await requireAlignedIndex(root, cwd);
      requireSuccess(await quality(root, false), "Commit blocked: strict repository-wide quality gate failed.");
      await requireAlignedIndex(root, cwd);
    },
    "tool.execute.after": async (input, output) => {
      const tool = name(input.tool);
      if (!mutations.has(tool) && tool !== "bash") return;
      let report;
      try {
        const result = await quality(root, true);
        report = `[local-quality: structure-only, exit ${result.code}]\n${result.text}`;
      } catch (error) {
        report = `[local-quality: unavailable]\n${error.message}`;
      }
      output.output = `${output.output ?? ""}\n\n${bounded(report)}`;
    },
  };
};
