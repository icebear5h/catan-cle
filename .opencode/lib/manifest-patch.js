import { normalizeLF, preserveBom, withoutBom } from "./manifest-text.js";

const fail = (message) => { throw new Error(`Manifest patch rejected: ${message}`); };

// Parse a strict subset of v1.18.31 patch/index.ts. Unlike its permissive parser,
// reject ignored/malformed lines. Only unprefixed headers are operations: +*** is data.
export function parsePatch(patch) {
  if (typeof patch !== "string") fail("patchText must be a string");
  const lines = normalizeLF(patch).trim().split("\n");
  if (lines[0] !== "*** Begin Patch" || lines.at(-1) !== "*** End Patch") {
    fail("expected a single Begin/End envelope, without wrappers");
  }
  const operations = [];
  let i = 1;
  while (i < lines.length - 1) {
    const header = /^\*\*\* (Add File|Update File|Delete File):(.*)$/.exec(lines[i++]);
    if (!header || !header[2].trim()) fail("malformed or unsupported operation header");
    const operation = { type: header[1], path: header[2].trim(), move: null, body: [] };
    if (lines[i]?.startsWith("*** Move to:")) {
      if (operation.type !== "Update File") fail("Move to requires Update File");
      operation.move = lines[i++].slice("*** Move to:".length).trim();
      if (!operation.move) fail("empty Move to path");
    }
    while (i < lines.length - 1 && !lines[i].startsWith("***")) {
      const line = lines[i++];
      if (["*** Begin Patch", "*** End Patch"].includes(line.trim())) {
        fail("ambiguous envelope marker in context; use narrower context");
      }
      if (operation.type === "Delete File" ||
          (operation.type === "Add File" ? !line.startsWith("+") : !/^(?:@@|[ +\-])/.test(line))) {
        fail("unsupported body line; use prefixed context/add/remove lines");
      }
      operation.body.push(line);
    }
    if (operation.type === "Update File" && lines[i] === "*** End of File") {
      operation.body.push(lines[i++]);
    }
    if ([operation.path, operation.move ?? ""].some((path) => /[\r\n\0]/.test(path))) {
      fail("invalid path characters");
    }
    operations.push(operation);
  }
  if (!operations.length) fail("empty patch");
  return operations;
}

function uniqueMatch(lines, pattern, start) {
  let found = -1;
  for (let i = start; i <= lines.length - pattern.length; i++) {
    if (!pattern.every((line, offset) => line === lines[i + offset])) continue;
    if (found !== -1) fail("ambiguous exact context; include more surrounding lines");
    found = i;
  }
  if (found === -1) fail("missing exact context; fuzzy matching is not supported");
  return found;
}

function fileLines(text) {
  const lines = text.split("\n");
  if (lines.at(-1) === "") lines.pop();
  return lines;
}

export function previewManifestPatch(source, body) {
  const lines = fileLines(normalizeLF(withoutBom(source)));
  const replacements = [];
  let cursor = 0, i = 0;
  while (i < body.length) {
    if (!/^@@(?: |$)/.test(body[i])) fail("manifest updates require @@ exact-context hunks");
    const anchor = body[i++].slice(2).trim();
    if (anchor) cursor = uniqueMatch(lines, [anchor], cursor) + 1;
    const before = [], after = [];
    let changed = false;
    while (i < body.length && !body[i].startsWith("@@") && body[i] !== "*** End of File") {
      const line = body[i++], prefix = line[0];
      if (![" ", "+", "-"].includes(prefix)) fail("unsupported hunk line");
      if (prefix !== "+") before.push(line.slice(1));
      if (prefix !== "-") after.push(line.slice(1));
      changed ||= prefix !== " ";
    }
    if (!before.length || !changed) fail("each hunk needs old context and a change");
    const index = uniqueMatch(lines, before, cursor);
    cursor = index + before.length;
    if (body[i] === "*** End of File") {
      if (++i !== body.length || cursor !== lines.length) fail("End of File context is not at EOF");
    }
    replacements.push({ index, length: before.length, after });
  }
  if (!replacements.length) fail("missing update hunks");
  const next = [...lines];
  for (const replacement of replacements.reverse()) {
    next.splice(replacement.index, replacement.length, ...replacement.after);
  }
  if (next.at(-1) !== "") next.push("");
  const ending = source.includes("\r\n") ? "\r\n" : "\n";
  return preserveBom(source, next.join(ending));
}

export function renderOperation(operation) {
  return [`*** ${operation.type}: ${operation.path}`,
    ...(operation.move ? [`*** Move to: ${operation.move}`] : []), ...operation.body].join("\n");
}

export function renderManifestUpdate(path, source, next) {
  // Host searches raw lines (CR included), strips the source BOM and ensures LF
  // termination. Full-file old context forces its exact first-pass match at 0.
  const oldLines = fileLines(withoutBom(source));
  if (!oldLines.length) fail("empty manifest requires a full-content write");
  return [`*** Update File: ${path}`, "@@",
    ...oldLines.map((line) => `-${line}`), ...fileLines(next).map((line) => `+${line}`)].join("\n");
}
