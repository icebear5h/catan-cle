// Pinned semantics: opencode v1.18.31 tool/edit.ts, tool/write.ts and util/bom.ts.
// Exact matching is required here; the host's fuzzy fallback is never admitted.
export const withoutBom = (text) => text.startsWith("\ufeff") ? text.slice(1) : text;
export const normalizeLF = (text) => text.replaceAll("\r\n", "\n");

export function preserveBom(source, replacement) {
  const next = withoutBom(replacement);
  const bom = source.startsWith("\ufeff") || replacement.startsWith("\ufeff");
  // Host Bom.split followed by Bom.join (which itself strips once).
  return (bom ? "\ufeff" : "") + withoutBom(next);
}

export function previewEdit(source, args) {
  const { oldString, newString, replaceAll = false } = args;
  if (typeof oldString !== "string" || !oldString || typeof newString !== "string" ||
      typeof replaceAll !== "boolean") {
    throw new Error("pyproject.toml edits require nonempty exact oldString and string newString.");
  }
  const content = withoutBom(source);
  const ending = content.includes("\r\n") ? "\r\n" : "\n";
  const convert = (text) => normalizeLF(text).replaceAll("\n", ending);
  const search = convert(oldString), replacement = convert(newString);
  if (search === replacement) throw new Error("No changes after host line-ending normalization.");
  const index = content.indexOf(search);
  if (index < 0 || (!replaceAll && index !== content.lastIndexOf(search))) {
    throw new Error("pyproject.toml edit has missing or ambiguous exact match after host normalization.");
  }
  // Native replaceAll expands $$, $&, $` and $'; single replacement is literal.
  const next = replaceAll ? content.replaceAll(search, replacement)
    : content.slice(0, index) + replacement + content.slice(index + search.length);
  return preserveBom(source, next);
}
