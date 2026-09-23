import { basename } from "node:path";

// Deliberately a small ordinary-command lexer, not a shell parser or sandbox.
// Wrappers, aliases, generated commands and concurrent mutations are outside its guarantee.
export function shellWords(command) {
  const words = [];
  let value = "", quote = "", active = false, unsafe = false;
  const flush = () => {
    if (active) words.push({ value, control: false });
    value = ""; active = false;
  };
  for (let i = 0; i < command.length; i++) {
    const char = command[i];
    if (quote) {
      if (char === quote) { quote = ""; continue; }
      if (quote === '"' && (char === "$" || char === "`")) unsafe = true;
      if (quote === '"' && char === "\\" && /[\$`"\\\n]/.test(command[i + 1] ?? "")) {
        value += command[++i];
      } else value += char;
      continue;
    }
    if (char === "'" || char === '"') { quote = char; active = true; }
    else if (char === "\\") {
      active = true;
      if (i + 1 === command.length || command[i + 1] === "\n") unsafe = true;
      value += command[++i] ?? "";
    } else if (/\s/.test(char)) {
      flush();
      if (char === "\n") words.push({ value: char, control: true });
    } else if (/[;&|<>(){}]/.test(char)) {
      flush(); words.push({ value: char, control: true });
    } else {
      active = true; value += char;
      if (/[$`*?\[~#]/.test(char)) unsafe = true;
    }
  }
  flush();
  return { words, unsafe: unsafe || Boolean(quote) };
}

function gitSubcommand(words, index) {
  const withValue = new Set(["-c", "-C", "--git-dir", "--work-tree", "--namespace", "--config-env"]);
  for (let i = index + 1; i < words.length && !words[i].control; i++) {
    const word = words[i].value;
    if (withValue.has(word)) { i++; continue; }
    if (word.startsWith("-")) continue;
    return word;
  }
  return "";
}

function validateCommitOptions(values) {
  const flags = new Set([
    "--amend", "--no-edit", "--allow-empty", "--allow-empty-message", "--signoff",
    "--no-signoff", "--reset-author", "--verbose", "--quiet", "--dry-run", "--all",
    "--no-gpg-sign", "--gpg-sign", "--edit", "--no-status", "--status",
    "-a", "-s", "-v", "-q", "-e", "-S",
  ]);
  const valued = new Set([
    "-m", "--message", "-F", "--file", "--author", "--date", "--cleanup", "--trailer",
    "--fixup", "--squash", "-C", "--reuse-message", "-c", "--reedit-message",
  ]);
  for (let i = 0; i < values.length; i++) {
    const word = values[i];
    if (word === "--no-verify" || /^-[asvqe]*n/.test(word)) {
      throw new Error("git commit --no-verify/-n is blocked.");
    }
    if (flags.has(word)) continue;
    if (valued.has(word)) {
      if (++i >= values.length) throw new Error(`Missing commit option value: ${word}`);
      continue;
    }
    if (word.startsWith("--") && valued.has(word.split("=", 1)[0]) && word.includes("=")) continue;
    if (/^--gpg-sign=|^-S.+|^-[asvqe]*[mF].+/.test(word)) continue;
    if (/^-[asvqe]*[mF]$/.test(word)) {
      if (++i >= values.length) throw new Error(`Missing commit option value: ${word}`);
      continue;
    }
    if (/^-[asvqe]+$/.test(word)) continue;
    throw new Error(`Unsupported commit option/pathspec: ${word}. Use a plain staged git commit.`);
  }
}

export function classifyCommit(command) {
  const { words, unsafe } = shellWords(command);
  const obvious = words.some((word, index) => basename(word.value) === "git" &&
    gitSubcommand(words, index) === "commit") ||
    words.some((word) => /\bgit\s+(?:-[^\n;|&]*\s+)?commit\b/.test(word.value));
  if (!obvious) return false;
  if (words.some((word) => /^core\.hookspath(?:=|$)/i.test(word.value) ||
      /^-ccore\.hookspath=/i.test(word.value))) {
    throw new Error("git core.hooksPath overrides are blocked for commits.");
  }
  if (unsafe || words.some((word) => word.control) ||
      words[0]?.value !== "git" || words[1]?.value !== "commit") {
    throw new Error("Use one standalone git commit command; split chains/control commands into separate calls. " +
      "Git prefixes, alternate paths/indexes and shell expansions are not supported.");
  }
  validateCommitOptions(words.slice(2).map((word) => word.value));
  return true;
}

export function guardDependencyCommand(command) {
  const { words } = shellWords(command);
  for (let i = 0; i < words.length; i++) {
    if (!/^pip(?:3(?:\.\d+)?)?$/.test(basename(words[i].value))) continue;
    for (let j = i + 1; j < words.length && !words[j].control; j++) {
      if (["install", "uninstall", "sync"].includes(words[j].value)) {
        throw new Error("pip/uv pip package mutation is blocked, including inside uv run. Use uv add/remove/sync.");
      }
    }
  }
  if (!/\b(?:pyproject\.toml|uv\.lock)\b/.test(command)) return;
  const mutation = words.some((word) => [
    "tee", "cp", "mv", "rm", "touch", "truncate", "dd", "install", "rsync", "patch", "ed", "ex",
  ].includes(basename(word.value)));
  const gitWrite = words.some((word, index) => basename(word.value) === "git" &&
    ["restore", "checkout", "reset", "apply"].includes(gitSubcommand(words, index)));
  const outputFile = words.some((word, index) =>
    /^(?:-o|-O|--output|--output-file)$/.test(word.value) &&
      /(?:pyproject\.toml|uv\.lock)$/.test(words[index + 1]?.value ?? "") ||
    /^(?:-o|-O|--output=|--output-file=).*(?:pyproject\.toml|uv\.lock)$/.test(word.value));
  const scriptWrite = /\b(?:write_text|write_bytes|writeFile(?:Sync)?|appendFile(?:Sync)?|truncate|unlink|rename)\b|\.write\s*\(|\bopen\s*\([^\n]*["'][wax][b+t]*["']/.test(command);
  const inPlace = /\b(?:sed|perl)\b[^\n;&|]*\s(?:-[a-zA-Z]*i|--in-place\b)/.test(command);
  if (command.includes(">") || mutation || gitWrite || outputFile || scriptWrite || inPlace) {
    throw new Error("Shell writes mentioning pyproject.toml/uv.lock are blocked. " +
      "Use edit/write for nonpackage config or uv add/remove/lock/sync for dependencies.");
  }
}
