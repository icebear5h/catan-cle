import { spawn } from "node:child_process";

export const OUTPUT_LIMIT = 16_384;

export function bounded(text, limit = OUTPUT_LIMIT) {
  return text.length <= limit ? text : `${text.slice(0, limit)}\n[output truncated]`;
}

// No shell interpolation; a timeout kills the whole checker process group on Unix.
export function run(command, args, {
  cwd, input, timeout = 30_000, env = process.env, maxOutput = OUTPUT_LIMIT,
} = {}) {
  return new Promise((resolve) => {
    const child = spawn(command, args, {
      cwd, env, detached: process.platform !== "win32", stdio: ["pipe", "pipe", "pipe"],
    });
    let text = "";
    let failure = "";
    let truncated = false;
    const capture = (chunk) => {
      const decoded = chunk.toString("utf8");
      const remaining = maxOutput - text.length;
      truncated ||= decoded.length > remaining;
      text += decoded.slice(0, remaining);
    };
    child.stdout.on("data", capture);
    child.stderr.on("data", capture);
    child.on("error", (error) => { failure = `Cannot execute ${command}: ${error.message}`; });
    child.stdin.on("error", (error) => {
      if (error.code !== "EPIPE") failure = `Cannot send checker input: ${error.message}`;
    });
    const timer = setTimeout(() => {
      failure = `Timed out after ${timeout}ms: ${command}`;
      try {
        if (process.platform !== "win32" && child.pid) process.kill(-child.pid, "SIGKILL");
        else child.kill("SIGKILL");
      } catch (error) {
        if (error.code !== "ESRCH") child.kill("SIGKILL");
      }
    }, timeout);
    child.on("close", (code, signal) => {
      clearTimeout(timer);
      const suffix = truncated ? "\n[output truncated]" : "";
      resolve({ code: failure ? 2 : (code ?? 2), text: bounded(
        [text + suffix, failure || (signal ? `Terminated by ${signal}` : "")]
          .filter(Boolean).join("\n"), maxOutput,
      ), truncated });
    });
    child.stdin.end(input);
  });
}

export function requireSuccess(result, context) {
  if (result.code !== 0 || result.truncated) {
    throw new Error(`${context}\n${bounded(result.text || `Checker exited ${result.code}`)}`);
  }
  return result.text;
}
