const { app, BrowserWindow, dialog } = require("electron");
const { spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");

const PORT = Number.parseInt(process.env.OVERMIND_PORT || "8765", 10);
const HOST = process.env.OVERMIND_HOST || "127.0.0.1";
const BASE_URL = `http://${HOST}:${PORT}`;

let mainWindow;
let backendProcess;
let closing = false;

function resolveBundledBackendPath() {
  const binaryName = process.platform === "win32" ? "overmind-backend.exe" : "overmind-backend";
  const bundledPath = path.join(process.resourcesPath, "backend", binaryName);
  if (fs.existsSync(bundledPath)) {
    return bundledPath;
  }
  return null;
}

function resolvePythonCommand() {
  if (process.env.OVERMIND_PYTHON) {
    return process.env.OVERMIND_PYTHON;
  }

  const workspaceRoot = path.resolve(__dirname, "..");
  const venvPython = path.join(workspaceRoot, ".venv", "bin", "python");
  if (fs.existsSync(venvPython)) {
    return venvPython;
  }

  return "python3";
}

function startBackend() {
  const workspaceRoot = path.resolve(__dirname, "..");
  const userDataDir = app.getPath("userData");
  const defaultWorkspace = path.join(userDataDir, "workspace");
  fs.mkdirSync(defaultWorkspace, { recursive: true });

  const bundledBackendPath = app.isPackaged ? resolveBundledBackendPath() : null;
  const command = bundledBackendPath || resolvePythonCommand();
  const args = bundledBackendPath
    ? ["--host", HOST, "--port", String(PORT)]
    : ["-m", "uvicorn", "app.main:app", "--host", HOST, "--port", String(PORT)];

  backendProcess = spawn(command, args, {
    cwd: workspaceRoot,
    env: {
      ...process.env,
      OVERMIND_WORKSPACE: process.env.OVERMIND_WORKSPACE || defaultWorkspace,
      OVERMIND_DB: process.env.OVERMIND_DB || path.join(userDataDir, "overmind.db"),
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  backendProcess.stdout.on("data", (chunk) => {
    process.stdout.write(`[overmind-api] ${chunk}`);
  });

  backendProcess.stderr.on("data", (chunk) => {
    process.stderr.write(`[overmind-api] ${chunk}`);
  });

  backendProcess.on("exit", (code, signal) => {
    if (closing) {
      return;
    }
    dialog.showErrorBox(
      "Overmind backend stopped",
      `The Python backend exited unexpectedly (code: ${String(code)}, signal: ${String(signal)}).`
    );
    app.quit();
  });
}

async function waitForBackendReady(timeoutMs = 15000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    try {
      const response = await fetch(`${BASE_URL}/health`);
      if (response.ok) {
        return;
      }
    } catch {
      // Backend not ready yet.
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("Timed out waiting for Overmind backend to start.");
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 980,
    minHeight: 700,
    title: "Overmind",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
}

async function bootstrap() {
  startBackend();
  await waitForBackendReady();
  createWindow();
  await mainWindow.loadURL(BASE_URL);
}

function stopBackend() {
  if (!backendProcess) {
    return;
  }
  closing = true;
  backendProcess.kill("SIGTERM");
  backendProcess = undefined;
}

app.whenReady().then(async () => {
  try {
    await bootstrap();
  } catch (error) {
    dialog.showErrorBox("Overmind startup failed", String(error));
    app.quit();
  }
});

app.on("window-all-closed", () => {
  stopBackend();
  app.quit();
});

app.on("before-quit", () => {
  stopBackend();
});
