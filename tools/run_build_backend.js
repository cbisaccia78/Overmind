const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const repoRoot = path.resolve(__dirname, "..");
const buildScript = path.join(repoRoot, "tools", "build_backend.py");

function resolveConfiguredPython() {
  const configured = process.env.OVERMIND_BUILD_PYTHON;
  if (configured && configured.trim()) {
    return { command: configured.trim(), argsPrefix: [] };
  }

  const venvCandidates =
    process.platform === "win32"
      ? [path.join(repoRoot, ".venv", "Scripts", "python.exe"), path.join(repoRoot, ".venv", "python.exe")]
      : [path.join(repoRoot, ".venv", "bin", "python"), path.join(repoRoot, ".venv", "bin", "python3")];

  for (const candidate of venvCandidates) {
    if (fs.existsSync(candidate)) {
      return { command: candidate, argsPrefix: [] };
    }
  }

  if (process.platform === "win32") {
    return { fallbackLaunchers: [{ command: "py", argsPrefix: ["-3"] }, { command: "python", argsPrefix: [] }] };
  }

  return { fallbackLaunchers: [{ command: "python3", argsPrefix: [] }, { command: "python", argsPrefix: [] }] };
}

function runWithLauncher(command, argsPrefix) {
  const result = spawnSync(command, [...argsPrefix, buildScript], {
    cwd: repoRoot,
    env: process.env,
    stdio: "inherit",
  });

  return result;
}

const runtime = resolveConfiguredPython();

if (runtime.command) {
  const result = runWithLauncher(runtime.command, runtime.argsPrefix);
  process.exit(result.status ?? 1);
}

for (const launcher of runtime.fallbackLaunchers) {
  const result = runWithLauncher(launcher.command, launcher.argsPrefix);
  if (!result.error) {
    process.exit(result.status ?? 1);
  }

  if (result.error.code !== "ENOENT") {
    throw result.error;
  }
}

throw new Error("Unable to find a Python 3 interpreter. Set OVERMIND_BUILD_PYTHON to continue.");
